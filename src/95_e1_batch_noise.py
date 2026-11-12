# -*- coding: utf-8 -*-
"""E1 — 批噪声与精度：效应矩阵是否可辨识？

回应审稿意见 C12（R6-M1，全篇唯一可能推翻实证结论的一条）。

问题
----
附录 E 写「effects are measured with batched scoring (batch size 16).
Single-prompt and batched scores agree to at most 0.5 logits, and most prompt
pairs agree to within 0.12」。而 §7.1 报告 float32 下中位 |dB| = 0.0195。

  批量噪声上界 0.5 logit  =  中位效应的 25 倍
  典型批量噪声 0.12 logit =  中位效应的  6 倍

如果效应比测量抖动还小一个量级，效应矩阵的秩次就主要由抖动决定。
本脚本直接测量这件事，并给出**裁决**。

设计
----
对若干组合，在 {bf16, float32} x {batch 1, 8, 16, 32} 八种配置下重算效应矩阵，
逐元素互比，得到：
  1. 配置间逐元素相关矩阵（8x8）
  2. 每特征的跨配置标准差 sigma_meas(f)，及其与 |dB_f| 的比值
  3. 可辨识比 = 中位 sigma_meas / 中位 |dB|   <-- 关键裁决量
  4. rel（split-half 可靠性）在各配置下的变化

裁决标准
--------
  典型偏差 <= 0.002 且 逐元素相关 >= 0.99  -> 通过，效应可辨识
  逐元素相关 < 0.9 或 典型偏差 > 0.02      -> 不通过，效应不可辨识

用法
----
  python 95_e1_batch_noise.py --list
  python 95_e1_batch_noise.py --combo qwen3.5-2b:16:ioi
  python 95_e1_batch_noise.py --all
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
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

ROOT = Path(__file__).resolve().parent
REPO = ROOT
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_e1_batchnoise"

# 覆盖 κ 的高中低 + 已知失败侧
DEFAULT_COMBOS = [
    ("qwen3.5-2b", 16, "ioi"),      # kappa ~0.32，grid 的主样本
    ("qwen2.5-7b-instruct", 18, "gt"),  # kappa ~0.72
    ("tinyllama-1.1b", 21, "ioi"),  # kappa ~0.85
    ("pythia-1.4b", 16, "ioi"),     # kappa ~0.09，预期失败侧
]

BATCHES = [8, 16, 32]
BATCH_REF = 1                 # 参考配置：只对少量特征跑，代价高但足以估噪声
N_FEATURES = 24               # 批量配置跑满的特征数
N_FEATURES_REF = 8            # batch=1 参考配置只跑这么多特征
N_PROMPTS = 100               # 用 100 个 prompt：批量组成变了正是要测的东西
PASS_CORR = 0.99
PASS_DEV = 0.002
FAIL_CORR = 0.90
FAIL_DEV = 0.02


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


def spearman(a, b):
    return _m77.spearman(a, b)


def effect_matrix(model, tok, items, layer_mod, sae, mean_t, std_t, cand,
                  device, batch_size, dtype_mode, b0=None):
    """在给定 batch_size 与精度口径下重算效应矩阵 (n_feat, n_prompt)。

    dtype_mode='f32' 用 lm_head 输入在 float32 下重算 logit 差；
    dtype_mode='bf16' 用模型原生 logits。

    b0 若已给出则复用，否则现算（每个配置只算一次，不要每特征重算）。
    """
    n_f, n_p = len(cand), len(items)
    E = np.zeros((n_f, n_p), dtype=np.float64)
    if b0 is None:
        if dtype_mode == "f32":
            b0, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, None,
                                            device, batch_size)
        else:
            b0, _ = _m36.batch_behavior(model, tok, items, layer_mod, None,
                                        device, batch_size)
        b0 = np.asarray(b0, dtype=np.float64)
    for i, f in enumerate(cand):
        hk = _m36.make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        if dtype_mode == "f32":
            v, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, hk,
                                           device, batch_size)
        else:
            v, _ = _m36.batch_behavior(model, tok, items, layer_mod, hk,
                                       device, batch_size)
        E[i] = np.asarray(v, dtype=np.float64) - b0
    return E


def rel_of(E, n_splits=20, seed0=4242):
    """split-half 可靠性（折叠口径，与 §7.2 一致），多 split 取均值。"""
    n_feat, n_prompt = E.shape
    h = n_prompt // 2
    vals = []
    for s in range(n_splits):
        rng = np.random.RandomState(seed0 + 1000 * s)
        perm = rng.permutation(n_prompt)
        sel, ev = perm[:h], perm[h:]
        c_sel = np.abs(E[:, sel].mean(axis=1))
        c_ev = np.abs(E[:, ev].mean(axis=1))
        r = spearman(c_sel, c_ev)
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def run_combo(m, layer, task, args, device, model=None, tok=None):
    tag = f"{m}:{layer}:{task}"
    out_dir = OUT / m / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_p = out_dir / f"{task}_e1.npz"
    json_p = out_dir / f"{task}_e1.json"
    if npz_p.exists() and json_p.exists() and not args.force:
        print(f"  [跳过] {tag}", flush=True)
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

    print(f"  [{tag}] prompts={len(items)} features={len(cand)}", flush=True)

    mats = {}
    t0 = time.time()
    # ---- 批量配置：跑满 N_FEATURES 个特征 ----
    for mode in ("bf16", "f32"):
        for bs in BATCHES:
            key = f"{mode}_b{bs}"
            mats[key] = effect_matrix(model, tok, items, layer_mod, sae, mean_t,
                                      std_t, cand, device, bs, mode)
            el = time.time() - t0
            print(f"    {key:10s} 中位|dB|={np.median(np.abs(mats[key])):.6f} "
                  f"({el:.0f}s)", flush=True)

    # ---- batch=1 参考配置：只跑前 N_FEATURES_REF 个特征 ----
    # batch=1 的代价是 batch=16 的 16 倍，故参考配置只用一半 prompt；
    # 比较时只取参考配置也覆盖的那部分 prompt。
    n_ref_prompt = len(items) // 2
    items_ref = items[:n_ref_prompt]
    cand_ref = cand[:N_FEATURES_REF]
    for mode in ("bf16", "f32"):
        key = f"{mode}_b1"
        b0_ref = None
        mats[key] = effect_matrix(model, tok, items_ref, layer_mod, sae, mean_t,
                                  std_t, cand_ref, device, 1, mode, b0=b0_ref)
        print(f"    {key:10s} 中位|dB|={np.median(np.abs(mats[key])):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- 统一裁剪到共有形状 ----
    # batch=1 参考配置只覆盖前一半 prompt、前 N_FEATURES_REF 个特征，
    # 故所有比较必须先截断到共有的 (特征, prompt) 形状，否则元素无法一一对应。
    keys = list(mats.keys())
    min_feat = min(mats[k].shape[0] for k in keys)
    min_prompt = min(mats[k].shape[1] for k in keys)
    n_ref_p = min_prompt
    for k in keys:
        mats[k] = mats[k][:min_feat, :min_prompt]

    # ---- 跨 dtype 偏差（同一 batch）----
    cross = {}
    for bs in BATCHES + [1]:
        a, b = f"bf16_b{bs}", f"f32_b{bs}"
        if a not in mats or b not in mats:
            continue
        d = mats[a] - mats[b]
        cross[f"b{bs}"] = {
            "median_abs_diff": float(np.median(np.abs(d))),
            "p90_abs_diff": float(np.percentile(np.abs(d), 90)),
            "max_abs_diff": float(np.abs(d).max()),
            "elem_corr": float(np.corrcoef(mats[a].ravel(),
                                          mats[b].ravel())[0, 1]),
        }

    # ---- 配置间逐元素相关 ----
    corr = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            n = min_feat
            x, y = mats[a][:n].ravel(), mats[b][:n].ravel()
            if x.std() < 1e-15 or y.std() < 1e-15:
                corr[i, j] = 1.0 if i == j else float("nan")
            else:
                corr[i, j] = float(np.corrcoef(x, y)[0, 1])

    # ---- 可辨识比（只比较同特征数的前 N_FEATURES_REF 行）----
    ident = {}
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if j <= i:
                continue
            n = min_feat
            d = mats[a][:n] - mats[b][:n]
            sigma = float(np.std(d))
            med_e = float(np.median(np.abs(mats[a][:n])))
            ident[f"{a}_vs_{b}"] = {
                "sigma_meas": sigma,
                "max_abs_diff": float(np.abs(d).max()),
                "median_abs_effect": med_e,
                "identifiability_ratio": sigma / max(med_e, 1e-12),
            }

    # ---- 每个配置的 rel 与量化统计 ----
    per_cfg = {}
    for k in keys:
        E = mats[k]
        per_cfg[k] = {
            "median_abs_E": float(np.median(np.abs(E))),
            "p90_abs_E": float(np.percentile(np.abs(E), 90)),
            "frac_exact_zero": float((np.abs(E) < 1e-12).mean()),
            "n_unique": int(len(np.unique(np.round(E, 9)))),
            "rel": rel_of(E),
            "n_features": int(E.shape[0]),
        }

    # ---- 裁决：同 dtype 下 batch 变化引入的噪声 vs 中位效应 ----
    ref = mats["f32_b16"]
    same_dtype_devs = []
    for bs in BATCHES:
        if bs == 16:
            continue
        d = mats[f"f32_b{bs}"] - ref
        same_dtype_devs.append((float(np.median(np.abs(d))),
                                float(np.percentile(np.abs(d), 90)),
                                float(np.abs(d).max()),
                                float(np.corrcoef(mats[f"f32_b{bs}"].ravel(),
                                                   ref.ravel())[0, 1])))
    # batch=1 参考：直接量测「单条 vs 批量」的偏差
    n_ref = min(mats["f32_b1"].shape[0], ref.shape[0])
    d_b1 = mats["f32_b1"][:n_ref] - ref[:n_ref]
    b1_dev_median = float(np.median(np.abs(d_b1)))
    b1_dev_p90 = float(np.percentile(np.abs(d_b1), 90))
    b1_dev_max = float(np.abs(d_b1).max())
    b1_corr = float(np.corrcoef(mats["f32_b1"][:n_ref, :n_ref_p].ravel(),
                                ref[:n_ref, :n_ref_p].ravel())[0, 1])

    med_dev = float(np.median([x[0] for x in same_dtype_devs]))
    p90_dev = float(np.median([x[1] for x in same_dtype_devs]))
    max_dev = float(np.max([x[2] for x in same_dtype_devs]))
    min_corr = float(np.min([x[3] for x in same_dtype_devs]))

    med_effect = float(np.median(np.abs(ref)))
    # 裁决量：单条 vs 批量的中位偏差（最直接对应审稿意见的 0.5 / 0.12）
    dev_for_verdict = max(b1_dev_median, med_dev)
    corr_for_verdict = min(b1_corr, min_corr)
    if corr_for_verdict < FAIL_CORR or dev_for_verdict > FAIL_DEV:
        verdict = "FAIL"
        reason = (f"单条 vs 批量的中位偏差 {b1_dev_median:.2e}（或批量间 {med_dev:.2e}）"
                  f"超阈值 {FAIL_DEV}，或逐元素相关 {corr_for_verdict:.4f} < {FAIL_CORR}"
                  f" -> 效应不可辨识")
    elif dev_for_verdict <= PASS_DEV and corr_for_verdict >= PASS_CORR:
        verdict = "PASS"
        reason = (f"单条 vs 批量的中位偏差 {b1_dev_median:.2e} <= {PASS_DEV}，"
                  f"逐元素相关 {corr_for_verdict:.4f} >= {PASS_CORR} -> 效应可辨识")
    else:
        verdict = "MARGINAL"
        reason = (f"单条 vs 批量的中位偏差 {b1_dev_median:.2e}、逐元素相关 "
                  f"{corr_for_verdict:.4f}，落在通过与失败之间")

    np.savez_compressed(npz_p,
                        **{k: v.astype(np.float32) for k, v in mats.items()},
                        min_feat=np.asarray([min_feat]),
                        min_prompt=np.asarray([min_prompt]),
                        cand=np.asarray(cand, dtype=np.int32))
    rec = {
        "schema": "cfse_e1_batchnoise/v1",
        "model": m, "layer": layer, "task": task,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_prompts": len(items), "n_features": len(cand),
        "configs": keys,
        "corr_matrix": corr.tolist(),
        "identifiability": ident,
        "per_config": per_cfg,
        "same_dtype_vs_b16": {
            "median_abs_dev": med_dev, "p90_abs_dev": p90_dev,
            "max_abs_dev": max_dev, "min_elem_corr": min_corr,
            "per_batch": {f"b{bs}": {"median_abs_dev": same_dtype_devs[i][0],
                                     "p90_abs_dev": same_dtype_devs[i][1],
                                     "max_abs_dev": same_dtype_devs[i][2],
                                     "elem_corr": same_dtype_devs[i][3]}
                          for i, bs in enumerate(bs for bs in BATCHES if bs != 16)},
        },
        "batch1_vs_batch16_f32": {
            "n_features_compared": int(n_ref), "n_prompts_compared": int(n_ref_p),
            "median_abs_dev": b1_dev_median, "p90_abs_dev": b1_dev_p90,
            "max_abs_dev": b1_dev_max, "elem_corr": b1_corr,
            "note": "batch=1 只跑前 N_FEATURES_REF 个特征，用于直接量测审稿意见所指的单条 vs 批量偏差",
        },
        "cross_dtype": cross,
        "median_abs_effect_f32_b16": med_effect,
        "identifiability_ratio_f32_b16": float(
            np.std(mats["f32_b16"][:n_ref, :n_ref_p]
                   - mats["f32_b1"][:n_ref, :n_ref_p])) / max(med_effect, 1e-12),
        "verdict": verdict, "reason": reason,
        "npz": str(npz_p.relative_to(ROOT)),
    }
    json_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    -> {verdict}: {reason}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="E1：批噪声与精度，效应矩阵是否可辨识")
    ap.add_argument("--combo", type=str, default=None, help="model:layer:task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.list:
        for c in DEFAULT_COMBOS:
            print(":".join(map(str, c)))
        return

    if args.combo:
        m, l, t = args.combo.split(":")
        combos = [(m, int(l), t)]
    elif args.all:
        combos = DEFAULT_COMBOS
    else:
        combos = DEFAULT_COMBOS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for m, l, t in combos:
        print(f"[E1] === {m} L{l} {t} ===", flush=True)
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
            print(f"  [失败] {m} L{l} {t}", flush=True)
        finally:
            model = None
            torch.cuda.empty_cache()

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print(f"{'combo':30s} {'中位|dB|':>10s} {'批偏差':>10s} {'相关':>8s} {'verdict':>9s}")
    print("-" * 70)
    for r in results:
        s = r["same_dtype_vs_b16"]
        print(f"{r['model']}|{r['layer']}|{r['task']:22s} "
              f"{r['median_abs_effect_f32_b16']:10.5f} {s['median_abs_dev']:10.2e} "
              f"{s['min_elem_corr']:8.4f} {r['verdict']:>9s}")
    print("=" * 70)
    verdicts = [r["verdict"] for r in results]
    overall = ("FAIL" if "FAIL" in verdicts else
               "MARGINAL" if "MARGINAL" in verdicts else
               "PASS" if verdicts else "N/A")
    print(f"总体裁决：{overall}")
    (OUT / "e1_summary.json").write_text(json.dumps(
        {"results": results, "overall": overall,
         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
