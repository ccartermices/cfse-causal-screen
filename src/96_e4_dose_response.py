# -*- coding: utf-8 -*-
"""E4 — 剂量-精度曲线：线性化在多大的干预下开始失效？

回应审稿意见 C04（R2-M2 与 R4-M1 共同提出，全篇第二重要的一条）。

问题
----
一阶预测器 pred_f 就是 dB_f 在零点的 Taylor 一阶项，而被预测的效应本身是
中位 |dB| = 0.0195、基线 margin 3.2--11.3 logits，即**相对扰动仅 0.2--0.6%**。
在这个尺度上，一阶近似必然准确，所以「rho = 0.725」有多少来自「扰动小」，
稿件没有分离。

反方向的证据其实就在稿件自己的 §4.6：加性近似与实测联合效应的 rho 只有 0.566。
即**干预一旦变大，线性化就崩**。

设计
----
把消融改成部分消融 z'_f = (1 - alpha) * z_f，alpha = 0.25 ... 5：
  alpha = 1    完整消融（现稿口径）
  alpha > 1    反向放大（沿同一方向推得更远）
  alpha < 1    部分消融（扰动更小）

关键性质：pred_f 对 alpha 是**线性**的（delta_f ∝ alpha），而实测效应不是。
因此 rho(alpha) 的偏离就是曲率的直接读数，不需要另建二阶项估计器。

同时报告：
  - 相对位移范数 ||delta_f|| / ||h||，把「扰动有多小」变成可比较量
  - 按 |pred| 三分位分层的 rho(alpha)（高 |pred| 档曲率最可能显著）
  - Pythia 与晚层（已知线性化失败侧）的对照

用法
----
  python 96_e4_dose_response.py --list
  python 96_e4_dose_response.py --all
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import time
import traceback
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from model_utils import (load_config, resolve_model_cfg, model_needs_fp8,  # noqa: E402
                         dequantize_fp8_model)

_m27 = import_module("27_h6_semantic_injection")
_m36 = import_module("36_causal_validity")
_m57 = import_module("57_resolution_fix")
_m77 = import_module("77_grid_heldout")

find_decoder_layers = _m27.find_decoder_layers
load_model_for_generation = _m27.load_model_for_generation
spearman = _m77.spearman

ROOT = Path(__file__).resolve().parent
REPO = ROOT
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_e4_dose"

# 覆盖正常侧与已知失败侧（Pythia 基座 / 晚层线性化更差）
DEFAULT_COMBOS = [
    ("qwen3.5-2b", 16, "ioi"),      # 可测侧
    ("tinyllama-1.1b", 21, "ioi"),  # 中等侧
    ("pythia-1.4b", 16, "ioi"),     # 已知失败侧
]

ALPHAS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
N_FEATURES = 60
BATCH = 16


def load_model(name):
    cfg = load_config(ROOT / "config.json")
    _, mc = resolve_model_cfg(cfg, name)
    model_path = REPO / mc["local_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"{name}: {model_path}")
    if model_needs_fp8(mc, model_path):
        try:
            from huggingface_hub import constants as _hf_const
            _hf_const.HF_HUB_OFFLINE = True
        except Exception:
            pass
    model, tok = load_model_for_generation(REPO, mc, model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model_needs_fp8(mc, model_path):
        n_deq = dequantize_fp8_model(model)
        print(f"  [fp8] 就地反量化模块数 = {n_deq}", flush=True)
    model.eval()
    return model, tok


def partial_ablate_hook(sae, mean_t, std_t, feat_idx, alpha):
    """部分消融 z'_f = (1 - alpha) * z_f。

    alpha=1 -> 置零（现稿口径）；alpha=0 -> 不干预；alpha>1 -> 反向放大。

    注意 36.make_feature_hook 在 mode="ablate" 时**忽略** alpha（coef 恒为 -s_f），
    只有 mode="steer" 才把 alpha 用进系数 coef=(alpha-1)*s_f。故这里必须走 steer：
      steer(alpha_p=1-alpha)  =>  coef = (1-alpha-1)*s_f = -alpha*s_f
    与「剂量 alpha 倍的消融方向位移」一致。
    """
    return _m36.make_feature_hook(sae, mean_t, std_t, int(feat_idx), "steer",
                                  1.0 - float(alpha))



def run_combo(m, layer, task, args, device, model=None, tok=None):
    out_dir = OUT / m / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_p = out_dir / f"{task}.npz"
    json_p = out_dir / f"{task}.json"
    if npz_p.exists() and json_p.exists() and not args.force:
        print(f"  [跳过] {m} L{layer} {task}", flush=True)
        return json.load(open(json_p, encoding="utf-8"))

    if model is None or tok is None:
        model, tok = load_model(m)
    layers_mod = find_decoder_layers(model)
    lmap = _m36.calibrate_layer_map(
        model, tok, layers_mod, device,
        "When Mary and John went to the store, John gave a drink to")
    block_idx = lmap[str(layer)]["block_idx"]
    layer_mod = layers_mod[block_idx]

    loaded = _m36.load_per_layer_sae(ROOT, m, layer, device)
    if loaded is None:
        print(f"  [缺 SAE] {m} L{layer}", flush=True)
        return None
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    built = _m36.TASK_BUILDERS[task](tok, args.n_prompts, seed=42)
    items = built[0] if isinstance(built, tuple) else built
    if not items:
        print(f"  [任务不可构造] {m} {task}", flush=True)
        return None

    feats = _m77.loaded_features(m, layer, task)
    if not feats:
        print(f"  [缺候选特征] {m} L{layer} {task}", flush=True)
        return None
    cand = [int(x["feature_idx"]) for x in feats][:N_FEATURES]

    print(f"  [{m} L{layer} {task}] prompts={len(items)} features={len(cand)}",
          flush=True)

    # 基线
    b0_32, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, None,
                                        device, BATCH)
    b0_32 = np.asarray(b0_32, dtype=np.float64)

    # 各 alpha 的效应矩阵
    mats = {}
    t0 = time.time()
    for a in ALPHAS:
        E = np.zeros((len(cand), len(items)), dtype=np.float64)
        for i, f in enumerate(cand):
            hk = partial_ablate_hook(sae, mean_t, std_t, f, a)
            v, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, hk,
                                           device, BATCH)
            E[i] = np.asarray(v, dtype=np.float64) - b0_32
        mats[a] = E
        print(f"    alpha={a:<5} 中位|dB|={np.median(np.abs(E)):.5f} "
              f"p90={np.percentile(np.abs(E), 90):.5f} ({time.time()-t0:.0f}s)",
              flush=True)

    # 逐 prompt 的 pred（与 77 同口径）
    W = sae.decoder.weight.detach().float()
    Wsel = W.index_select(1, torch.tensor(cand, dtype=torch.long, device=device))
    Wsc = Wsel * std_t.view(-1, 1)
    pred_p, _ = _m77.per_prompt_pred(model, tok, items, layer_mod, sae,
                                     mean_t, std_t, Wsc, cand, device, BATCH)
    pred = np.abs(pred_p.mean(axis=0))   # |pred| 在全体 prompt 上的聚合
    pred_signed = pred_p.mean(axis=0)

    # rho(alpha)：|pred| 与 |dB| 的秩相关；并按 |pred| 三分位分层
    rows = []
    for a in ALPHAS:
        E = mats[a]
        c = np.abs(E.mean(axis=1))
        rho_all = spearman(pred, c)
        # 分层
        order = np.argsort(-pred, kind="mergesort")
        third = len(order) // 3
        hi = order[:third]
        mid = order[third:2 * third]
        lo = order[2 * third:]
        rows.append({
            "alpha": a,
            "rho_all": float(rho_all),
            "rho_hi_pred": float(spearman(pred[hi], c[hi])),
            "rho_mid_pred": float(spearman(pred[mid], c[mid])),
            "rho_lo_pred": float(spearman(pred[lo], c[lo])),
            "median_abs_effect": float(np.median(np.abs(E))),
            "p90_abs_effect": float(np.percentile(np.abs(E), 90)),
            "sign_acc": float(np.mean(np.sign(pred_signed)
                                      == np.sign(E.mean(axis=1)))),
            # b0 是逐 prompt 的 (n_prompt,)，E 是 (n_feat, n_prompt)；
            # 广播后逐 (特征, prompt) 判断该消融是否把答案推向另一侧。
            "flip_rate": float(np.mean(np.sign(b0_32[None, :] + E)
                                       != np.sign(b0_32[None, :]))),
        })

    # 曲率读数：rho 相对 alpha=1 的偏离
    rho_a1 = next(r["rho_all"] for r in rows if r["alpha"] == 1.0)
    for r in rows:
        r["rho_drop_vs_alpha1"] = float(rho_a1 - r["rho_all"])

    # 线性预测：pred 随 alpha 线性，实测不线性 -> 报告逐 alpha 的相对误差
    np.savez_compressed(npz_p,
                        **{f"E_a{a}": mats[a].astype(np.float32) for a in ALPHAS},
                        pred=pred.astype(np.float32),
                        pred_signed=pred_signed.astype(np.float32),
                        cand=np.asarray(cand, dtype=np.int32),
                        b0=b0_32.astype(np.float32))
    rec = {
        "schema": "cfse_e4_dose/v1",
        "model": m, "layer": layer, "task": task,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_prompts": len(items), "n_features": len(cand),
        "alphas": ALPHAS,
        "curve": rows,
        "rho_alpha1": float(rho_a1),
        "rho_alpha5": float(next(r["rho_all"] for r in rows if r["alpha"] == 5.0)),
        "rho_drop_total": float(rho_a1 - next(r["rho_all"] for r in rows
                                              if r["alpha"] == 5.0)),
        "b0_mean": float(b0_32.mean()),
        "npz": str(npz_p.relative_to(ROOT)),
    }
    json_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    rho(a=1)={rho_a1:.4f} -> rho(a=5)={rec['rho_alpha5']:.4f} "
          f"(降 {rec['rho_drop_total']:+.4f})", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="E4：剂量-精度曲线")
    ap.add_argument("--combo", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        for c in DEFAULT_COMBOS:
            print(":".join(map(str, c)))
        return

    if args.combo:
        m, l, t = args.combo.split(":")
        combos = [(m, int(l), t)]
    else:
        combos = DEFAULT_COMBOS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for m, l, t in combos:
        print(f"[E4] === {m} L{l} {t} ===", flush=True)
        model = tok = None
        try:
            model, tok = load_model(m)
            r = run_combo(m, l, t, args, device, model, tok)
            if r:
                results.append(r)
            del model
            model = None
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
        finally:
            model = None
            torch.cuda.empty_cache()

    print("\n" + "=" * 74)
    print(f"{'combo':30s} " + " ".join(f"a={a:<5}" for a in ALPHAS))
    print("-" * 74)
    for r in results:
        vals = {row["alpha"]: row["rho_all"] for row in r["curve"]}
        print(f"{r['model']}|{r['layer']}|{r['task']:18s} " +
              " ".join(f"{vals[a]:6.3f}" for a in ALPHAS))
    print("=" * 74)
    (OUT / "e4_summary.json").write_text(json.dumps(
        {"results": results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
