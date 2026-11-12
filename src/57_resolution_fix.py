# -*- coding: utf-8 -*-
"""57_resolution_fix.py — 测量分辨率修正: 用 float32 重算被消融 logit

发现
----
53 存盘的逐 prompt 效应矩阵 E 只有 12 个不同取值, 54.9% 恰好为 0,
且全部是 0.0625 的整数倍 —— 这是 bf16 logit 的 ULP (logits 量级 ~16-32 时
ULP = 2^4 * 2^-8 = 0.0625)。36 里的 `logits[...].float()` 只是把已量化的
bf16 值拓宽, 并不恢复分辨率。

后果
----
qwen 的 |效应| 中位数 0.0042, 而量化步长 0.0625 —— 分辨率比 RR ≈ 15,
远超 CFSE G3 门控的 0.1 上限。此时"逐 prompt 效应"绝大多数是 0 或 ±1 ULP,
可靠性低主要来自测量分辨率, 而不是 prompt 异质性。

修正
----
logit 的量化发生在最后一步累加输出, 不是 hidden state。因此只要拿到 lm_head
的输入 h (post-norm hidden) 和权重 W, 在 float32 下做
    logit_v = h.float() @ W[v].float()
即可恢复亚 ULP 分辨率 (h 本身是 bf16, 但 2k+ 维求和给出细粒度结果)。

本脚本对同一批候选特征重算 E_f32, 并与 53 的 E_bf16 逐项对比:
  - 取值个数 / 零点占比 / 是否仍在 0.0625 网格
  - ICC(1)、可靠性曲线、达到 rel=0.8 所需 prompt
若修正有效, ICC 与可靠性应显著上升。

输出: output/phaseStep2/cfse_resolution/{model}/layer{L}/{task}.json/.md
      (同时存 {task}_E_f32.npz 供后续复用)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib

_m49 = importlib.import_module("49_cfse_heldout")
_m36 = importlib.import_module("36_causal_validity")
_m53 = importlib.import_module("53_reliability_curve")

TASK_BUILDERS = _m36.TASK_BUILDERS
capture_sparse = _m36.capture_sparse
find_decoder_layers = _m36.find_decoder_layers
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae
make_feature_hook = _m36.make_feature_hook
load_model_for_generation = _m36.load_model_for_generation
_encode_with_prefix = _m36._encode_with_prefix
from model_utils import (load_config, resolve_model_cfg, model_needs_fp8,
                         dequantize_fp8_model)

ULP_BF16 = 0.0625


def batch_behavior_f32(model, tokenizer, items, layer_mod, hook_fn, device,
                       batch_size=16):
    """与 batch_behavior 同口径, 但两个 logit 在 float32 下重算。"""
    mlp = model.get_output_embeddings()
    W = mlp.weight.detach()
    outs = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
        cap = {}
        h_drop = layer_mod.register_forward_hook(hook_fn) if hook_fn else None

        def pre(mod, args):
            cap["h"] = args[0]

        h_head = mlp.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            if h_drop:
                h_drop.remove()
            h_head.remove()
        h = cap["h"]
        h = (h[0] if isinstance(h, (tuple, list)) else h)[:, -1, :].float()
        cid = torch.tensor([it["correct_id"] for it in chunk], device=h.device)
        did = torch.tensor([it["distractor_id"] for it in chunk], device=h.device)
        wc = W[cid].float()
        wd = W[did].float()
        diff = (h * wc).sum(-1) - (h * wd).sum(-1)
        outs.extend(diff.cpu().tolist())
    return outs, float(np.mean(outs)) if outs else 0.0


def quant_stats(E):
    v = E.ravel()
    k = np.round(v / ULP_BF16)
    return {"n": int(v.size), "n_unique": int(len(np.unique(np.round(v, 6)))),
            "frac_zero": float((np.abs(v) < 1e-12).mean()),
            "frac_on_grid": float((np.abs(k - np.round(k)) < 1e-6).mean()),
            "p50_abs": float(np.percentile(np.abs(v), 50)),
            "p90_abs": float(np.percentile(np.abs(v), 90)),
            "max_abs": float(np.abs(v).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--n_prompts", type=int, default=400)
    ap.add_argument("--n_features", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    rel_dir = base / "output" / "phaseStep2" / "cfse_reliability" / args.model / f"layer{args.layer}"
    npz_bf = rel_dir / f"{args.task}_E.npz"
    if not npz_bf.exists():
        print(f"缺少 bf16 基线 {npz_bf}, 先跑 53")
        return
    zb = np.load(npz_bf)
    E_bf = zb["E"].astype(float)
    cand = [int(x) for x in zb["cand"].tolist()]

    out_dir = base / "output" / "phaseStep2" / "cfse_resolution" / args.model / f"layer{args.layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.task}.json"
    if out_file.exists() and not args.force:
        print(f"[跳过] 已有 {out_file}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = load_config(base / "config.json")
    _, mc = resolve_model_cfg(cfg, args.model)
    model_path = base / mc["local_path"]
    if not model_path.exists():
        for p in (base / "models").glob("*/*"):
            if p.is_dir() and (p / "config.json").exists():
                model_path = p
                break
    if model_needs_fp8(mc, model_path):
        try:
            from huggingface_hub import constants as _hf_const
            _hf_const.HF_HUB_OFFLINE = True
        except Exception:
            pass
        try:
            import huggingface_hub._snapshot_download as _sd
            _sd.read_tree_cache = lambda *a, **k: None
        except Exception:
            pass
    model, tok = load_model_for_generation(base, mc, model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model_needs_fp8(mc, model_path):
        model = dequantize_fp8_model(model)
    model.eval()

    layers_mod = find_decoder_layers(model)
    lmap = calibrate_layer_map(model, tok, layers_mod, device,
                               "When Mary and John went to the store, John gave a drink to")
    block_idx = lmap[str(args.layer)]["block_idx"]
    layer_mod = layers_mod[block_idx]

    loaded = load_per_layer_sae(base, args.model, args.layer, device)
    if loaded is None:
        print("缺少 SAE")
        return
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    items, _ = TASK_BUILDERS[args.task](tok, args.n_prompts, seed=42)
    print(f"[{args.model} L{args.layer} {args.task}] prompts={len(items)} "
          f"features={len(cand)}", flush=True)

    b0_vals, b0 = batch_behavior_f32(model, tok, items, layer_mod, None, device,
                                     args.batch_size)
    b0_arr = np.asarray(b0_vals, dtype=float)
    print(f"  b0(f32)={b0:.5f} sd={b0_arr.std():.5f}", flush=True)

    E = np.zeros((len(cand), len(items)), dtype=float)
    for i, f in enumerate(cand):
        hk = make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        vals, _ = batch_behavior_f32(model, tok, items, layer_mod, hk, device,
                                     args.batch_size)
        E[i] = np.asarray(vals, dtype=float) - b0_arr
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(cand)}", flush=True)

    np.savez_compressed(out_dir / f"{args.task}_E_f32.npz",
                        E=E.astype(np.float32),
                        cand=np.asarray(cand, dtype=np.int32))

    ns = [n for n in (12, 25, 50, 100, 200) if 2 * n <= len(items)]
    curve_f32 = _m53.rel_curve(E, ns)
    vc_f32 = _m53.variance_components(E)
    curve_bf = _m53.rel_curve(E_bf, ns)
    vc_bf = _m53.variance_components(E_bf)

    same = E.shape == E_bf.shape
    corr = None
    if same:
        corr = float(np.corrcoef(E.ravel(), E_bf.ravel())[0, 1])

    res = {"model": args.model, "layer": args.layer, "task": args.task,
           "timestamp": datetime.now().isoformat(), "n_prompts": len(items),
           "n_features": len(cand), "b0_f32": float(b0),
           "b0_sd_f32": float(b0_arr.std()),
           "quant_f32": quant_stats(E), "quant_bf16": quant_stats(E_bf),
           "icc_f32": vc_f32.get("icc"), "icc_bf16": vc_bf.get("icc"),
           "curve_f32": {str(k): v for k, v in curve_f32.items()},
           "curve_bf16": {str(k): v for k, v in curve_bf.items()},
           "n_for_0.8_f32": _m53.spearman_brown_up(
               curve_f32[max(curve_f32)]["rel"], max(curve_f32), 0.8) if curve_f32 else None,
           "n_for_0.8_bf16": _m53.spearman_brown_up(
               curve_bf[max(curve_bf)]["rel"], max(curve_bf), 0.8) if curve_bf else None,
           "corr_E_f32_vs_bf16": corr}
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    qf, qb = res["quant_f32"], res["quant_bf16"]
    L = [f"# 测量分辨率修正 (float32 重算 logit): {args.model} L{args.layer} {args.task}\n",
         f"- prompts={len(items)}, features={len(cand)}；"
         f"bf16 基线来自 53 的 `{args.task}_E.npz`",
         f"- b0(f32) = {b0:.5f} ± {b0_arr.std():.5f}（bf16 口径 b0 见 53）\n",
         "## 效应矩阵的量化结构\n",
         "| 量 | bf16 输出端 | float32 重算 |",
         "|---|---|---|",
         f"| 不同取值个数 | {qb['n_unique']} | **{qf['n_unique']}** |",
         f"| 恰为 0 的比例 | {qb['frac_zero']:.3f} | **{qf['frac_zero']:.3f}** |",
         f"| 落在 0.0625 网格上 | {qb['frac_on_grid']:.3f} | **{qf['frac_on_grid']:.3f}** |",
         f"| \\|E\\| 中位数 | {qb['p50_abs']:.5f} | {qf['p50_abs']:.5f} |",
         f"| \\|E\\| p90 | {qb['p90_abs']:.5f} | {qf['p90_abs']:.5f} |",
         f"| \\|E\\| 最大 | {qb['max_abs']:.4f} | {qf['max_abs']:.4f} |",
         ""]
    if corr is not None:
        L.append(f"- 两种口径逐元素相关 r = {corr:.4f}\n")
    L += ["## 可靠性\n",
          "| n(prompt) | bf16 rel | float32 rel |",
          "|---|---|---|"]
    for n in ns:
        a = (curve_bf.get(n) or {}).get("rel")
        b = (curve_f32.get(n) or {}).get("rel")
        L.append(f"| {n} | {a:.3f} | **{b:.3f}** |" if a is not None and b is not None
                 else f"| {n} | — | — |")
    L.append("")
    L.append(f"- ICC(1)：bf16 **{vc_bf.get('icc'):.4f}** → float32 "
             f"**{vc_f32.get('icc'):.4f}**")
    L.append(f"- 达到 rel=0.8 所需 prompt：bf16 {res['n_for_0.8_bf16']:.0f} → "
             f"float32 **{res['n_for_0.8_f32']:.0f}**")
    (out_dir / f"{args.task}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
