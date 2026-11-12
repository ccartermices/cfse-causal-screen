# -*- coding: utf-8 -*-
"""53_reliability_curve.py — 可靠性-样本量曲线 (CFSE 测量学层)

问题
----
E2b 在 200/200 prompt 上测得逐特征效应的 split-half 可靠性只有 0.26,
而 E2 pilot 在 100/100 上是 0.41。样本量翻倍而可靠性下降, 与 Spearman-Brown
外推矛盾, 说明"可靠性"不是单个数字, 需要知道它随 prompt 数的真实曲线。

做法
----
对同一批 200 个候选特征, 在全部 N 条 prompt 上各做一次消融, 保存**逐 prompt**
行为变化矩阵 E[feature, prompt]。之后所有可靠性与方差分解都可离线重算:
  - 对每个 n, 随机抽两个不相交的 n 条子集, 各自取 |mean(E)| 得 C1/C2, 求 Spearman;
    重复 B 次取均值 -> rel(n)
  - 用 Spearman-Brown 反解达到 rel=0.8 所需的 n
  - 方差分解: 组内(prompt) vs 组间(feature) -> ICC,
    解释为什么可靠性低: 单特征效应本身小, 而跨 prompt 波动大

成本: 200 特征 × N prompts 的前向 (与一次 held-out split 相当), 约 10-15 min。

用法:
  python 53_reliability_curve.py --model qwen3.5-2b --layer 16 --task ioi --n_prompts 400
输出:
  output/phaseStep2/cfse_reliability/{model}/layer{L}/{task}.json/.md
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

TASK_BUILDERS = _m36.TASK_BUILDERS
batch_behavior = _m36.batch_behavior
capture_sparse = _m36.capture_sparse
find_decoder_layers = _m36.find_decoder_layers
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae
make_feature_hook = _m36.make_feature_hook
load_model_for_generation = _m36.load_model_for_generation
from model_utils import (load_config, resolve_model_cfg, model_needs_fp8,
                         dequantize_fp8_model)


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return None
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_brown_up(rel_n, n, target):
    if rel_n is None or not (0 < rel_n < 1) or not (0 < target < 1):
        return None
    if rel_n >= target:
        return 0
    return float(target * n * (1 - rel_n) / (rel_n * (1 - target)))


def rel_curve(E, ns, n_draws=20, seed=0):
    """E: [n_feat, n_prompt] 逐 prompt 效应; 返回 {n: {rel, se, n_pairs}}"""
    rng = np.random.RandomState(seed)
    F, P = E.shape
    out = {}
    for n in ns:
        if 2 * n > P:
            continue
        vals = []
        for _ in range(n_draws):
            perm = rng.permutation(P)
            a, b = perm[:n], perm[n:2 * n]
            c1 = np.abs(E[:, a].mean(axis=1))
            c2 = np.abs(E[:, b].mean(axis=1))
            r = spearman(c1, c2)
            if r is not None:
                vals.append(r)
        if vals:
            out[n] = {"rel": float(np.mean(vals)),
                      "se": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                      "n_pairs": len(vals)}
    return out


def variance_components(E):
    """E[feature, prompt]: 组间/组内方差, ICC(1)."""
    F, P = E.shape
    grand = E.mean()
    feat_means = E.mean(axis=1)
    ss_between = P * ((feat_means - grand) ** 2).sum()
    ss_within = ((E - feat_means[:, None]) ** 2).sum()
    ms_between = ss_between / max(F - 1, 1)
    ms_within = ss_within / max(F * (P - 1), 1)
    icc = ((ms_between - ms_within) /
           (ms_between + (P - 1) * ms_within)) if ms_between > 0 else None
    return {"ms_between": float(ms_between), "ms_within": float(ms_within),
            "icc": float(icc) if icc is not None else None,
            "sd_across_prompts_mean": float(E.std(axis=1).mean()),
            "sd_of_feature_means": float(feat_means.std(ddof=1)),
            "mean_abs_effect": float(np.abs(feat_means).mean())}


def run(args, config, base, model_name, layer, task):
    _, mc = resolve_model_cfg(config, model_name)
    out_dir = (base / "output" / "phaseStep2" / "cfse_reliability" / model_name
               / f"layer{layer}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"
    if out_file.exists() and not args.force:
        print(f"  [跳过] {model_name} L{layer} {task}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = base / mc["local_path"]
    if not model_path.exists():
        for p in (base / "models").glob("*/*"):
            if p.is_dir() and (p / "config.json").exists():
                model_path = p
                break
    print(f"\n[{model_name} L{layer} {task}] 加载基座模型...", flush=True)
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
    block_idx = lmap[str(layer)]["block_idx"]
    layer_mod = layers_mod[block_idx]

    loaded = load_per_layer_sae(base, model_name, layer, device)
    if loaded is None:
        print("  缺少 SAE")
        return
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    items, meta = TASK_BUILDERS[task](tok, args.n_prompts, seed=args.seed)
    print(f"  [{model_name} L{layer} {task}] prompts={len(items)}", flush=True)
    b0_vals, b0 = batch_behavior(model, tok, items, layer_mod, None, device,
                                 args.batch_size)
    b0_arr = np.asarray(b0_vals, dtype=float)

    packs = capture_sparse(model, tok, items, layer_mod, sae, mean_t, std_t,
                           device, args.batch_size)
    S = np.concatenate([p[0] for p in packs], axis=0)
    M = np.concatenate([p[1] for p in packs], axis=0)
    S = S[M]
    mean_act = S.mean(axis=0)
    cand = [int(x) for x in np.argsort(mean_act)[::-1][:args.n_features].tolist()]
    del S

    E = np.zeros((len(cand), len(items)), dtype=float)
    for i, f in enumerate(cand):
        hk = make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        vals, _ = batch_behavior(model, tok, items, layer_mod, hk, device,
                                 args.batch_size)
        E[i] = np.asarray(vals, dtype=float) - b0_arr
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(cand)}", flush=True)

    ns = [12, 25, 50, 100, 200, 400]
    curve = rel_curve(E, [n for n in ns if 2 * n <= len(items)])
    vc = variance_components(E)
    need = {str(t): spearman_brown_up(curve[max(curve)]["rel"], max(curve), t)
            for t in (0.6, 0.7, 0.8, 0.9)} if curve else {}

    out = {"model": model_name, "layer": layer, "task": task,
           "timestamp": datetime.now().isoformat(),
           "n_prompts": len(items), "n_features": len(cand),
           "b0": float(b0), "b0_std": float(b0_arr.std()),
           "curve": {str(k): v for k, v in curve.items()},
           "variance": vc, "n_for_rel": need,
           "E_shape": list(E.shape)}
    # 保存矩阵以便后续任意重算 (float32 压缩)
    np.savez_compressed(out_dir / f"{task}_E.npz",
                        E=E.astype(np.float32),
                        cand=np.asarray(cand, dtype=np.int32))
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    L = [f"# 可靠性-样本量曲线：{model_name} L{layer} {task}\n",
         f"- prompts={len(items)}，候选特征={len(cand)}，b0={b0:.3f}，b0_sd={b0_arr.std():.3f}",
         f"- 逐 prompt 效应 sd（特征内平均）= {vc['sd_across_prompts_mean']:.4f}",
         f"- 特征均值 sd = {vc['sd_of_feature_means']:.4f}，"
         f"平均 |效应| = {vc['mean_abs_effect']:.4f}，ICC(1) = {vc['icc']:.4f}\n",
         "| n(prompt) | rel | se | 说明 |", "|---|---|---|---|"]
    for n, d in curve.items():
        L.append(f"| {n} | {d['rel']:.3f} | {d['se']:.3f} | "
                 f"{'← 半样本' if 2*n == len(items) else ''} |")
    L.append("")
    L.append("达到目标可靠性所需 prompt 数：")
    for t, n in need.items():
        L.append(f"- rel={t}: {'已达' if n == 0 else f'{n:.0f}'} 条")
    (out_dir / f"{task}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--n_prompts", type=int, default=400)
    ap.add_argument("--n_features", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    config = load_config(base / "config.json")
    run(args, config, base, args.model, args.layer, args.task)


if __name__ == "__main__":
    main()
