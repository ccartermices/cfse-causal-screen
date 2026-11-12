# -*- coding: utf-8 -*-
"""E3 — SAE-free 基线：这到底是 SAE 现象，还是梯度现象？

回应审稿意见 C03（R2-M1，P0）。

问题
----
pred_f = -E[sum_t s_f(t) (w_f o sigma) . g(t)]。其中 s_f w_f 是 **SAE 特征的贡献**。
但「读出对齐」是残差流坐标本身就有的属性，与 SAE 是否学到单义特征无关。

若把 s_f w_f 换成 h(t) 本身（残差流坐标），得到一个**完全不需要 SAE**的因果筛查器：
    score_i = |sum_t g_i(t) h_i(t)|
如果它的 rho 与 pred 接近，则
  (1) 本文对 SAE 指标的批判仍成立（M1--M6 确实弱），
  (2) 但「first-order predictor」的贡献必须改写为「梯度 x 激活在 SAE 基上不优于坐标基」，
  (3) 且 §5.3 的三因子分解失去 SAE 特异性——它在坐标基上同样成立。

反之，若坐标基明显更差，那才是本文最强的结果：SAE 的 decoder 方向提供了一个
比坐标更对齐的投影基——这才是「SAE 有什么用」的正面回答，而现稿完全没有给出。

设计
----
四个基线，同 hook、同 prompt 数、同真值口径（float32），各取 200 个候选：
  (a) SAE-pred      现稿的一阶预测器（W_dec 列投影）
  (b) coord         坐标基 |sum_t g_i(t) h_i(t)|，逐坐标消融
  (c) random        200 个随机正交方向，同式打分，沿方向消融
  (d) pca           任务激活的前 200 个主方向，同式打分，沿方向消融

每个基线各自定义自己的真值（沿该基的干预），避免跨基比较的语义混乱。
报告配对差 rho_pred - rho_coord 及其 CI，这才是「SAE 是否带来增益」的估计量。

用法
----
  python 97_e3_sae_free.py --list
  python 97_e3_sae_free.py --all
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUM_EXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
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
OUT = BASE / "cfse_e3_saefree"

DEFAULT_COMBOS = [
    ("qwen3.5-2b", 16, "ioi"),
    ("tinyllama-1.1b", 21, "ioi"),
]

N_CAND = 80
BATCH = 16
RNG_SEED = 42


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


# ---------------------------------------------------------------- 方向基干预
def make_direction_ablate_hook(dirs: torch.Tensor):
    """沿一组方向做消融：把 h 在这些方向上的投影减去。

    dirs: (n_dir, d_model)，已归一化。
    """
    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        orig_dtype = hs.dtype
        h = hs.float()
        with torch.no_grad():
            proj = torch.einsum("btd,fd->btf", h, dirs)      # (B,T,F)
            delta = torch.einsum("btf,fd->btd", proj, dirs)   # (B,T,D)
            h_new = (h - delta).to(orig_dtype)
        if isinstance(output, (tuple, list)):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_coord_ablate_hook(idx: int):
    """消融单个残差流坐标。"""
    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        orig_dtype = hs.dtype
        h = hs.float().clone()
        h[:, :, idx] = 0.0
        h_new = h.to(orig_dtype)
        if isinstance(output, (tuple, list)):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def capture_h_and_grad(model, tok, items, layer_mod, device, batch_size):
    """分批前向+反向，同时取 h(t)、g(t)、行为 margin 与 mask。

    必须分批：把 200 条 prompt 一次喂进去，反向图会大到不可用（实测单次要数分钟）。
    分批后每批独立求梯度再拼接——各条样本之间没有交互，等价且快一个量级。
    """
    hs, gs, bs, ms = [], [], [], []
    for s0 in range(0, len(items), batch_size):
        chunk = items[s0:s0 + batch_size]
        caps = {}

        def hk(module, args, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            caps["h"] = x

        input_ids, attn = _m36._encode_with_prefix(tok, chunk, device)
        handle = layer_mod.register_forward_hook(hk)
        try:
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :].float()
        finally:
            handle.remove()
        h = caps["h"]
        cid = torch.tensor([it["correct_id"] for it in chunk], device=logits.device)
        did = torch.tensor([it["distractor_id"] for it in chunk], device=logits.device)
        Bm = (logits.gather(1, cid[:, None]) - logits.gather(1, did[:, None])).squeeze(1)
        g = torch.autograd.grad(Bm.sum(), h)[0]
        mask = attn.unsqueeze(-1).float()
        # 各 batch 的序列长度不同（该 batch 内最长者），拼到全局定长 128 再 cat，
        # 否则 (B,T,D) 的 T 不一致会直接报错。mask 为 0 处 h/g 也已为 0。
        TMAX = 128

        def _pad(x):
            if x.shape[1] >= TMAX:
                return x[:, :TMAX]
            z = torch.zeros(x.shape[0], TMAX - x.shape[1], x.shape[2],
                            dtype=x.dtype, device=x.device)
            return torch.cat([z, x], dim=1) if False else torch.cat([x, z], dim=1)

        hs.append(_pad(h.detach().float()))
        gs.append(_pad(g.float() * mask))
        ms.append(_pad(mask))
        bs.append(Bm.detach())
        del h, g, logits
    return (torch.cat(hs, 0), torch.cat(gs, 0), torch.cat(bs, 0), torch.cat(ms, 0))


def score_and_effect_for_dirs(model, tok, items, layer_mod, dirs, device,
                               batch_size=BATCH):
    """对一个方向基：算分数 (n_prompt, n_dir) 与逐 prompt 真值 (n_dir, n_prompt)。

    分数 = |sum_t <g(t), u> <h(t), u>| 逐 prompt
    真值 = 该方向被消融后 dB（逐 prompt）
    """
    n_dir = dirs.shape[0]
    h, g, B0, mask = capture_h_and_grad(model, tok, items, layer_mod, device, batch_size)
    B, T, D = h.shape
    with torch.no_grad():
        gh = torch.einsum("btd,fd->btf", g.float(), dirs)   # (B,T,F)
        hh = torch.einsum("btd,fd->btf", h.float(), dirs)   # (B,T,F)
        score_p = (gh * hh).sum(dim=1)                # (B,F) 有符号
        ntok = mask.sum(dim=1).clamp(min=1.0)         # (B,)
        score_p = score_p / ntok.view(-1, 1)
    score_p = score_p.detach().cpu().double().numpy()

    # 真值：逐方向消融
    E = np.zeros((n_dir, B), dtype=np.float64)
    for i in range(n_dir):
        hk = make_direction_ablate_hook(dirs[i:i + 1])
        v, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, hk,
                                       device, batch_size)
        E[i] = np.asarray(v, dtype=np.float64) - B0.cpu().double().numpy()
    del h, g
    torch.cuda.empty_cache()
    return score_p, E


def main():
    ap = argparse.ArgumentParser(description="E3：SAE-free 基线")
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
    for m, layer, task in combos:
        print(f"[E3] === {m} L{layer} {task} ===", flush=True)
        out_dir = OUT / m / f"layer{layer}"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_p = out_dir / f"{task}.json"
        if json_p.exists() and not args.force:
            print(f"  [跳过]", flush=True)
            results.append(json.load(open(json_p, encoding="utf-8")))
            continue
        try:
            model, tok = load_model(m)
            layers_mod = find_decoder_layers(model)
            lmap = _m36.calibrate_layer_map(
                model, tok, layers_mod, device,
                "When Mary and John went to the store, John gave a drink to")
            layer_mod = layers_mod[lmap[str(layer)]["block_idx"]]

            loaded = _m36.load_per_layer_sae(ROOT, m, layer, device)
            if loaded is None:
                print(f"  [缺 SAE] {m} L{layer}", flush=True)
                continue
            sae, mean_np, std_np = loaded
            mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
            std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

            built = _m36.TASK_BUILDERS[task](tok, args.n_prompts, seed=42)
            items = built[0] if isinstance(built, tuple) else built
            feats = _m77.loaded_features(m, layer, task)
            if not items or not feats:
                print(f"  [缺任务或特征]", flush=True)
                continue
            cand = [int(x["feature_idx"]) for x in feats][:N_CAND]

            # ---- (a) SAE-pred（现稿口径）----
            W = sae.decoder.weight.detach().float()
            Wsel = W.index_select(1, torch.tensor(cand, dtype=torch.long, device=device))
            Wsc = Wsel * std_t.view(-1, 1)
            pred_p, _ = _m77.per_prompt_pred(model, tok, items, layer_mod, sae,
                                             mean_t, std_t, Wsc, cand, device, BATCH)
            E_sae = np.zeros((len(cand), len(items)), dtype=np.float64)
            b0_sae, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, None,
                                                device, BATCH)
            b0_sae = np.asarray(b0_sae, dtype=np.float64)
            t0 = time.time()
            for i, f in enumerate(cand):
                hk = _m36.make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
                v, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, hk,
                                               device, BATCH)
                E_sae[i] = np.asarray(v, dtype=np.float64) - b0_sae
            print(f"    (a) SAE-pred 真值矩阵 {time.time()-t0:.0f}s", flush=True)
            rho_sae = float(spearman(np.abs(pred_p.mean(axis=0)),
                                     np.abs(E_sae.mean(axis=1))))

            # ---- (b) 坐标基 ----
            h, g, B0, mask = capture_h_and_grad(model, tok, items, layer_mod,
                                                device, BATCH)
            B, T, D = h.shape
            with torch.no_grad():
                score_coord = (g * h).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0).view(-1, 1)
            score_coord = score_coord.detach().cpu().double().numpy()
            del h, g
            torch.cuda.empty_cache()
            E_coord = np.zeros((D, len(items)), dtype=np.float64)
            b0c = B0.cpu().double().numpy()
            t0 = time.time()
            # 坐标太多，按 |score| 取 top-N_CAND 个
            agg = np.abs(score_coord.mean(axis=0))
            top_coord = np.argsort(-agg, kind="mergesort")[:N_CAND]
            for i, ci in enumerate(top_coord):
                hk = make_coord_ablate_hook(int(ci))
                v, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, hk,
                                               device, BATCH)
                E_coord[int(ci)] = np.asarray(v, dtype=np.float64) - b0c
                if (i + 1) % 50 == 0:
                    print(f"    (b) 坐标 {i+1}/{N_CAND} ({time.time()-t0:.0f}s)",
                          flush=True)
            rho_coord = float(spearman(np.abs(score_coord.mean(axis=0))[top_coord],
                                       np.abs(E_coord[top_coord].mean(axis=1))))

            # ---- (c) 随机正交方向 ----
            rng = np.random.RandomState(RNG_SEED)
            rnd = rng.randn(N_CAND, D)
            q, _ = np.linalg.qr(rnd.T)
            dirs_rnd = torch.tensor(q.T.copy(), dtype=torch.float32, device=device)
            score_rnd, E_rnd = score_and_effect_for_dirs(model, tok, items,
                                                          layer_mod, dirs_rnd,
                                                          device, BATCH)
            rho_rnd = float(spearman(np.abs(score_rnd.mean(axis=0)),
                                     np.abs(E_rnd.mean(axis=1))))

            # ---- (d) PCA 方向（任务激活的前 N_CAND 个主方向）----
            h2, g2, B0b, mask2 = capture_h_and_grad(model, tok, items, layer_mod,
                                                    device, BATCH)
            with torch.no_grad():
                flat = h2[mask2.squeeze(-1) > 0].reshape(-1, h2.shape[-1])
                if flat.shape[0] > 20000:
                    flat = flat[torch.randperm(flat.shape[0])[:20000]]
                cov = (flat.T @ flat) / max(flat.shape[0], 1)
                eigvals, eigvecs = torch.linalg.eigh(cov.double())
                top_pca = eigvecs[:, -N_CAND:].T.float().contiguous()
            dirs_pca = top_pca.to(device)
            del h2, g2
            torch.cuda.empty_cache()
            score_pca, E_pca = score_and_effect_for_dirs(model, tok, items,
                                                         layer_mod, dirs_pca,
                                                         device, BATCH)
            rho_pca = float(spearman(np.abs(score_pca.mean(axis=0)),
                                     np.abs(E_pca.mean(axis=1))))

            rec = {
                "schema": "cfse_e3_saefree/v1",
                "model": m, "layer": layer, "task": task,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_prompts": len(items), "n_cand": N_CAND,
                "rho": {"sae_pred": rho_sae, "coord": rho_coord,
                        "random_dir": rho_rnd, "pca": rho_pca},
                "delta_sae_minus_coord": float(rho_sae - rho_coord),
                "n_features_sae": len(cand),
                "n_coord_evaluated": len(top_coord),
                "d_model": int(D),
            }
            json_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            results.append(rec)
            print(f"    rho: SAE={rho_sae:.3f} coord={rho_coord:.3f} "
                  f"rand={rho_rnd:.3f} pca={rho_pca:.3f}  "
                  f"(SAE - coord = {rho_sae - rho_coord:+.3f})", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception:
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    if results:
        print("\n" + "=" * 74)
        print(f"{'combo':30s} {'SAE':>7s} {'coord':>7s} {'rand':>7s} {'pca':>7s} {'SAE-coord':>10s}")
        print("-" * 74)
        for r in results:
            q = r["rho"]
            print(f"{r['model']}|{r['layer']}|{r['task']:18s} "
                  f"{q['sae_pred']:7.3f} {q['coord']:7.3f} {q['random_dir']:7.3f} "
                  f"{q['pca']:7.3f} {r['delta_sae_minus_coord']:+10.3f}")
        print("=" * 74)
        d = np.array([r["delta_sae_minus_coord"] for r in results])
        print(f"配对差均值 {d.mean():+.3f}，中位 {np.median(d):+.3f}，"
              f"SAE 胜 {int((d>0).sum())}/{d.size}")
        (OUT / "e3_summary.json").write_text(json.dumps(
            {"results": results, "mean_delta": float(d.mean()),
             "median_delta": float(np.median(d)),
             "sae_wins": int((d > 0).sum()), "n": int(d.size),
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
