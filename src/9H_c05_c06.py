# -*- coding: utf-8 -*-
"""E8 — 归因基线的两处收口：KernelSHAP 预算收敛 / 标准 IG

回应审稿意见 C05（R2-M3）与 C06（R2-M4）。

C05 的问题
----------
现稿用 KernelSHAP 在 16 个特征上取 128 个联盟，而 16 特征的全联盟空间是 2^16 = 65,536,
即覆盖 0.2%；没有收敛曲线，也没有种子方差。审稿人因此问：SHAP 的读数是不是预算的产物？

本脚本给出三件事：
  (1) F=8 子集上做**精确 Shapley**（2^8 = 256 个联盟全枚举）作为真值，比较 64/128/256
      三种预算下的 KernelSHAP 与真值的秩一致性；
  (2) F=16（现稿口径）比较 128/512/1024 三种预算的 φ 秩一致性，以及同预算换种子的方差；
  (3) 每个预算下 |φ| 与实测 |ΔB| 的秩相关，看"SHAP 不如一阶预测器"这个结论是否随预算改变。

C06 的问题
----------
现稿 IG 的路径起点是全特征位移（h - R），不是标准 IG 的基线；只有 8 步；投影后
completeness 不成立。审稿人要求：均值基线、32/128 步、并报投影残差占比。

本脚本按标准做法重算：
  - 基线 b = 该层训练集的均值激活（即 SAE 的 mean，与标准化一致）；
  - 路径 h(α) = b + α·(h - b)，α 取 m 等分右端点；
  - 特征 f 的归因 IG_f = Σ_t c_f(t)·(1/m)Σ_k ⟨∇_h B(h(α_k)), u_f⟩，
    c_f 为原点的稀疏激活，u_f = W_sc[:, f]；
  - 于是 Σ_f IG_f 精确等于 B(b + R_F) - B(b)（R_F = Σ_f c_f u_f），completeness 可检验；
  - 另报投影残差占比 ||(h-b) - R_F|| / ||h-b||，量化"用 F 个特征解释位移"漏掉多少。

用法
----
  python 9H_c05_c06.py --list
  python 9H_c05_c06.py --all
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import itertools
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
_m47 = import_module("47_attribution_baselines")
_m77 = import_module("77_grid_heldout")

find_decoder_layers = _m27.find_decoder_layers
load_model_for_generation = _m27.load_model_for_generation
spearman = _m77.spearman
_encode_with_prefix = _m36._encode_with_prefix
load_per_layer_sae = _m36.load_per_layer_sae
_save = _m36._save

ROOT = Path(__file__).resolve().parent
REPO = ROOT
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_e8_shap_ig"
SRC = BASE / "causal_validity"

DEFAULT_COMBOS = [
    ("pythia-1.4b", 16, "ioi"),
    ("qwen3.5-2b", 16, "ioi"),
]

BATCH = 32


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


# ---------------------------------------------------------------- SHAP 部分
def shap_values_for(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc,
                    feat_idx, Z, device, batch_size=BATCH):
    """给定联盟矩阵 Z，返回逐联盟的价值 v(S)（B 的均值）。"""
    idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
    v = np.zeros(Z.shape[0], dtype=np.float64)
    for i in range(Z.shape[0]):
        mask = torch.tensor(Z[i], dtype=torch.float32, device=device)
        hk = _m47.make_multi_ablate_hook(sae, mean_t, std_t, idx_t, mask)
        v[i], _ = _m47.shap_value(model, tok, items, layer_mod, hk, device, batch_size)
    return v


def exact_shapley(F, v, Z):
    """全枚举联盟下的精确 Shapley（Z 为 2^F × F 的 0/1 矩阵）。"""
    from math import factorial
    lookup = {tuple(int(x) for x in row): v[i] for i, row in enumerate(Z)}
    phi = np.zeros(F, dtype=np.float64)
    for i in range(F):
        others = [j for j in range(F) if j != i]
        for r in range(F):
            wgt = factorial(r) * factorial(F - r - 1) / factorial(F)
            for S in itertools.combinations(others, r):
                base = [0] * F
                for j in S:
                    base[j] = 1
                with_i = list(base)
                with_i[i] = 1
                phi[i] += wgt * (lookup[tuple(with_i)] - lookup[tuple(base)])
    return phi


def rank(v):
    from scipy.stats import rankdata
    return rankdata(v)


def do_shap(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc, feat_idx,
            rows, b0_ref, device, seed0, budgets=(128, 512, 1024),
            f_small=8, f_full=16):
    """C05 主体：F=8 精确对照 + F=16 预算收敛 + 换种子方差。"""
    res = {}

    # ---- (1) F=8 精确 Shapley ----
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["M2_mean_act_task"] or -1.0), reverse=True)
    pos8 = order[:f_small]
    idx8 = [feat_idx[i] for i in pos8]
    Z8 = np.array(list(itertools.product([0, 1], repeat=f_small)), dtype=np.float64)
    # 保证第 0 行为空集、最后一行为全联盟
    Z8 = Z8[np.argsort(Z8.sum(axis=1), kind="stable")]
    t0 = time.time()
    v8 = shap_values_for(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc,
                         idx8, Z8, device)
    phi8 = exact_shapley(f_small, v8, Z8)
    v_empty, v_full = float(v8[0]), float(v8[-1])
    res["exact_F8"] = {
        "feature_idx": [int(x) for x in idx8],
        "phi": phi8.tolist(),
        "v_empty": v_empty, "v_full": v_full,
        "n_coalitions": int(len(v8)),
        "efficiency_gap": float(abs(phi8.sum() - (v_full - v_empty))),
        "seconds": time.time() - t0,
    }
    print(f"  [SHAP] F=8 精确: 256 联盟, 效率 gap={res['exact_F8']['efficiency_gap']:.2e}, "
          f"{res['exact_F8']['seconds']:.0f}s", flush=True)

    # ---- F=8 预算对照（与精确值比秩一致性）----
    conv = []
    for K in (64, 128, 256):
        Z, w = _m47.build_coalitions(f_small, K, seed0)
        v = shap_values_for(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc,
                            idx8, Z, device)
        phi = _m47.kernel_shap_exact(Z, v - v[0], w, v[-1] - v[0])
        conv.append({"K": int(Z.shape[0]), "rho_vs_exact": float(spearman(phi, phi8)),
                     "max_abs_err": float(np.max(np.abs(phi - phi8)))})
        print(f"  [SHAP] F=8 K={Z.shape[0]}: rho(φ,φ_exact)={conv[-1]['rho_vs_exact']:.3f}",
              flush=True)
    res["F8_budget"] = conv

    # ---- (2) F=16（现稿口径）预算收敛 + 种子方差 ----
    pos16 = order[:f_full]
    idx16 = [feat_idx[i] for i in pos16]
    eff = np.array([abs(rows[i]["B_ablate"] - b0_ref) for i in pos16], dtype=np.float64)
    phi_by = {}
    for K in budgets:
        Z, w = _m47.build_coalitions(f_full, K, seed0)
        v = shap_values_for(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc,
                            idx16, Z, device)
        phi = _m47.kernel_shap_exact(Z, v - v[0], w, v[-1] - v[0])
        phi_by[K] = phi
        print(f"  [SHAP] F=16 K={Z.shape[0]}: 效率 gap="
              f"{abs(phi.sum() - (v[-1] - v[0])):.2e}", flush=True)
    res["F16_budget"] = {
        f"{a}_vs_{b}": float(spearman(phi_by[a], phi_by[b]))
        for a, b in ((128, 512), (512, 1024), (128, 1024)) if a in phi_by and b in phi_by
    }
    res["F16_rho_vs_effect"] = {str(K): float(spearman(np.abs(phi_by[K]), eff))
                                for K in phi_by}
    res["F16_n_eff"] = int(len(eff))

    # 同预算换种子（K=128，3 个种子）
    seeds = [seed0, seed0 + 1, seed0 + 2]
    phis = []
    for sd in seeds:
        Z, w = _m47.build_coalitions(f_full, 128, sd)
        v = shap_values_for(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc,
                            idx16, Z, device)
        phis.append(_m47.kernel_shap_exact(Z, v - v[0], w, v[-1] - v[0]))
    pair = [float(spearman(phis[i], phis[j]))
            for i in range(len(phis)) for j in range(i + 1, len(phis))]
    res["F16_seed_rho"] = {"pairs": pair, "median": float(np.median(pair)),
                           "min": float(np.min(pair))}
    print(f"  [SHAP] F=16 K=128 换种子: rho 中位 {res['F16_seed_rho']['median']:.3f} "
          f"最小 {res['F16_seed_rho']['min']:.3f}", flush=True)
    return res


# ---------------------------------------------------------------- IG 部分
def ig_standard(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc, feat_idx,
                device, steps, batch_size=BATCH):
    """标准 IG：均值基线 + 直线路径 + m 步中点梯度。

    返回 (IG_f, completeness_ratio, residual_frac, n)。
      completeness_ratio = ΣIG_f / (B(b+R_F) - B(b))，应接近 1；
      residual_frac      = median_t ||(h-b) - R_F|| / ||h-b||。
    """
    idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
    Wsc = Wsc.index_select(1, idx_t)          # (D, F) 只保留被归因的特征
    F = len(feat_idx)
    term = np.zeros(F, dtype=np.float64)
    resid = []
    bR_vals, b0_vals, b0_base = [], [], []
    n = 0
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tok, chunk, device)
        cap = {}
        handle = layer_mod.register_forward_hook(_m47._leaf_hook(cap, "h"))
        try:
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        finally:
            handle.remove()
        h = cap["h"].detach().float()
        logits = out.logits[:, -1, :].float()
        B0 = _m47._behavior(logits, chunk)
        b0_vals.extend(B0.detach().cpu().tolist())
        with torch.no_grad():
            sparse = sae.encode((h - mean_t) / std_t)[0].index_select(-1, idx_t)
            R = sparse @ Wsc.t()                                  # (B,T,D)
            delta = h - mean_t.view(1, 1, -1)
            num = torch.linalg.norm(delta - R, dim=-1)
            den = torch.linalg.norm(delta, dim=-1) + 1e-12
            resid.append((num / den).reshape(-1).cpu().double().numpy())
        del out, logits, B0

        # B(b + R_F)：路径终点，用于 completeness 检验
        cap_r = {}

        def hk_r(module, args, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            dt = hs.dtype
            newh = (mean_t.view(1, 1, -1) + R).to(dt)
            cap_r["h"] = newh
            return (newh,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else newh

        handle = layer_mod.register_forward_hook(hk_r)
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        finally:
            handle.remove()
        bR_vals.extend(_m47._behavior(out.logits[:, -1, :].float(), chunk).cpu().tolist())
        del out, cap_r

        cap_b = {}

        def hk_b(module, args, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            dt = hs.dtype
            newh = torch.nn.functional.pad(
                mean_t.view(1, 1, -1).expand(hs.shape[0], hs.shape[1], -1), (0, 0, 0, 0))
            cap_b["h"] = newh
            return (newh.to(dt),) + tuple(output[1:]) if isinstance(output, (tuple, list)) else newh.to(dt)

        handle = layer_mod.register_forward_hook(hk_b)
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        finally:
            handle.remove()
        b0_base.extend(_m47._behavior(out.logits[:, -1, :].float(), chunk).cpu().tolist())
        del out, cap_b

        gsum = torch.zeros_like(h)
        for k in range(1, steps + 1):
            alpha = k / float(steps)
            cap2 = {}

            def hk(module, args, output, alpha=alpha):
                hs = output[0] if isinstance(output, (tuple, list)) else output
                dt = hs.dtype
                # 路径 b -> b + R_F：起点是均值基线（标准 IG），终点是本层特征的
                # 重建位移。沿此路径 Σ_f IG_f = B(b+R_F) - B(b) 精确成立。
                cur = (mean_t.view(1, 1, -1) + alpha * R)
                leaf = cur.requires_grad_(True)
                cap2["leaf"] = leaf
                return (leaf.to(dt),) + tuple(output[1:]) if isinstance(output, (tuple, list)) else leaf.to(dt)

            handle = layer_mod.register_forward_hook(hk)
            try:
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            finally:
                handle.remove()
            B = _m47._behavior(out.logits[:, -1, :].float(), chunk)
            g = torch.autograd.grad(B.sum(), cap2["leaf"])[0].float()
            g = g * attn.unsqueeze(-1).float()
            gsum += g
            del out, B, g, cap2
        gbar = gsum / steps
        t = torch.einsum("btd,df,btf->bf", gbar, Wsc, sparse)
        term += t.sum(0).cpu().double().numpy()
        n += len(chunk)
        del h, gsum, gbar, sparse, R, t

    ig_f = term / max(n, 1)
    # 子空间内的 completeness：Σ_f IG_f 应等于 B(b+R_F) - B(b)
    span = float(np.mean(bR_vals) - np.mean(b0_base))
    comp = float(ig_f.sum() / span) if abs(span) > 1e-9 else float("nan")
    # 投影损失：同一批 IG 只能解释"真实输入变化 B(h)-B(b)"中的多大比例
    span_in = float(np.mean(b0_vals) - np.mean(b0_base))
    comp_in = (float(abs(ig_f.sum()) / abs(span_in)) if abs(span_in) > 1e-9
               else float("nan"))
    return ig_f, comp, comp_in, float(np.median(np.concatenate(resid))), n


def do_ig(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc, feat_idx, rows,
          b0_ref, device, steps_list=(8, 32, 128)):
    eff = np.array([abs(r["B_ablate"] - b0_ref) for r in rows], dtype=np.float64)
    out = {}
    for st in steps_list:
        t0 = time.time()
        ig_f, comp, comp_in, resid, n = ig_standard(model, tok, items, layer_mod, sae,
                                                    mean_t, std_t, Wsc, feat_idx, device, st)
        out[str(st)] = {"rho_vs_effect": float(spearman(np.abs(ig_f), eff)),
                        "completeness_ratio": comp,
                        "completeness_vs_input_change": comp_in,
                        "residual_frac": resid,
                        "sum_ig": float(ig_f.sum()),
                        "seconds": time.time() - t0}
        print(f"  [IG] steps={st}: rho(|IG|,|ΔB|)={out[str(st)]['rho_vs_effect']:.3f} "
              f"completeness={comp:.3f} 残差={resid:.3f} ({out[str(st)]['seconds']:.0f}s)",
              flush=True)
    return out


# ---------------------------------------------------------------- 主流程
def run_combo(model_name, layer, task, args):
    out_file = OUT / model_name / f"layer{layer}" / f"{task}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and not args.force:
        print(f"  [跳过] 已有 {out_file}")
        return json.load(open(out_file, encoding="utf-8"))

    src = SRC / model_name / f"layer{layer}" / f"{task}.json"
    if not src.exists():
        print(f"  [跳过] 缺 36 输出: {src}")
        return None
    rec36 = json.load(open(src, encoding="utf-8"))
    if rec36.get("at_chance"):
        print("  [跳过] 该组合在 chance 水平")
        return None
    block_idx = rec36.get("block_idx")
    _, rows = _m47.load_targets(str(src))
    if len(rows) < 40:
        print("  [跳过] 可用特征不足")
        return None
    feat_idx = [r["feature_idx"] for r in rows]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(model_name)
    layers_mod = find_decoder_layers(model)
    layer_mod = layers_mod[block_idx]
    loaded = load_per_layer_sae(ROOT, model_name, layer, device)
    if loaded is None:
        print(f"  [跳过] 缺 SAE: {model_name} layer{layer}")
        return None
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)
    W = sae.decoder.weight.detach().float()
    Wsc = W * std_t.view(-1, 1)

    res = {"model": model_name, "layer": layer, "task": task,
           "n_features_pool": len(feat_idx), "energy": {}}

    items, _ = _m36.TASK_BUILDERS[task](tok, args.n_prompts, seed=42)
    print(f"[{model_name} L{layer} {task}] prompts={len(items)} pool={len(feat_idx)}",
          flush=True)

    if not args.skip_shap:
        # F=8/F=16 的项都只取 M2 排序的前若干，故 rows 顺序按 M2 稳定排序
        sub_items = items[:args.shap_prompts]
        res["shap"] = do_shap(model, tok, sub_items, layer_mod, sae, mean_t, std_t, Wsc,
                              feat_idx, rows, rec36.get("b0") or 0.0, device, args.seed)
        res["shap_n_prompts"] = len(sub_items)
    if not args.skip_ig:
        res["ig"] = do_ig(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc,
                          feat_idx, rows, rec36.get("b0") or 0.0, device)

    _save(out_file, res)
    del model, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", type=str, default=None,
                    help="model:layer:task,model:layer:task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--shap_prompts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_shap", action="store_true")
    ap.add_argument("--skip_ig", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    combos = DEFAULT_COMBOS
    if args.combos:
        combos = []
        for tok in args.combos.split(","):
            m, l, t = tok.split(":")
            combos.append((m, int(l), t))
    if args.list:
        for c in combos:
            print(c)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    for m, l, t in combos:
        try:
            run_combo(m, l, t, args)
        except Exception:
            print(f"[错误] {m} L{l} {t}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
