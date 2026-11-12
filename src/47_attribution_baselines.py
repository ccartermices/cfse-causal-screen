# -*- coding: utf-8 -*-
"""Step 4: 昂贵归因基线 vs 零干预一阶预测器

问题
----
42 号实验已经证明: 用「一次前向 + 一次反向」算出的解析一阶预测器 pred_full
(pooled spearman ≈ +0.73) 远优于只看激活幅度的标准指标 M2 (≈ +0.23)。
审稿人会问: 这个便宜的解析近似是否只是「碰巧」? 与**昂贵的忠实归因方法**
(Integrated Gradients / KernelSHAP / 注意力归因) 相比, 它是同样好、更好、还是更差?

本脚本在**完全相同的 prompt 集与干预语义**下, 对每个 (model, mid-layer, task) 计算:

  (a) 一阶解析预测器 (重用 42 的公式, 本脚本内重算, 与 36 目标同 prompt 数)
        pred_full   Σ_t s_f(t) · (W_dec[:,f]⊙std)·g(t)
        pred_last   仅末位 token
        pred_align  纯 decoder-方向对齐 |(W_dec[:,f]⊙std)·ḡ|
  (b) Integrated Gradients (IG): 沿残差流 h 从数据均值 h' 到真实 h 的直线路径,
        m 步右 Riemann 和; 归因到特征 f 的方向 W_dec[:,f]⊙std:
        IG_f = -Σ_t Σ_d Wsc[d,f]·(h_td-h'_td)·ḡ_td  (取负号 -> 预测 ablation 的 ΔB)
  (c) KernelSHAP: 对 top-F (默认 16, 按 M2 选) 特征做多人合作博弈,
        v(S) = 对 S 中特征做**多特征联合 ablation** 时的平均 B
        (与 36 单特征 ablation 完全同一语义: Δh = -Σ_{f∈S} s_f·W_dec[:,f]⊙std);
        分层采样 K 个联盟 + 精确效率约束 Σφ = v(全)-v(∅)
  (d) 注意力归因: 末层各头平均、末位 token 对各位置的注意力 a_t,
        score_f = -Σ_t a_t·s_f(t)·(Wsc[:,f]·g_t)
  (e) 随机分数地板 (对照, 期望 ρ≈0)
  (f) 标准指标 M2 (36 落盘) 作为参照

真值: 36 号实验落盘的 C_ablate = |B_ablate - b0| (200 prompts, seed 42, 主链)。
本脚本**不重做任何单特征干预**, 只重算预测器; 并额外校验:
  - b0 与 36 的一致性 (同 prompt 集, 非 fp8 模型应 <1e-6)
  - SHAP 空联盟 == b0; 单特征联盟 == 36 的 B_ablate (前 2 个特征)
  - 效率约束 |Σφ-(v_full-v_empty)| ~ 1e-12

比较口径:
  - 全特征集 (≈200): pred_full / pred_last / pred_align / ig / attn / M2 / random
  - SHAP 子集 (top-16): 上述全部 + shap, 逐组合配对 Δρ

资源: CPU 严格节流 (OMP/MKL=1, torch threads=1), 串行, 每模型只加载一次基座,
      独显 cuda:0。IG 与 attention 需 autograd, fp8 模型先反量化为 bf16 (与 42 一致)。
输出: output/phaseStep2/causal_attribution{tag}/{model}/layer{L}/{task}.json
      output/phaseStep2/causal_attribution{tag}/attribution_summary.json/.md
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from importlib import import_module

torch.set_num_threads(1)

from model_utils import load_config, resolve_model_cfg, model_needs_fp8

_m05 = import_module("05_train_text_sae")
LogSaver = _m05.LogSaver

_m27 = import_module("27_h6_semantic_injection")
find_decoder_layers = _m27.find_decoder_layers
load_model_for_generation = _m27.load_model_for_generation

_m36 = import_module("36_causal_validity")
TASK_BUILDERS = _m36.TASK_BUILDERS
_encode_with_prefix = _m36._encode_with_prefix
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae
_save = _m36._save

_m34 = import_module("34_step1_metric_validity")
spearman = _m34.spearman
pearson = _m34.pearson

_m42 = import_module("42_causal_margin_probe")
fp8_dequant_selfcheck = _m42.fp8_dequant_selfcheck

METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
           "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]

OUT_TAG = ""      # --out_tag: 同时决定读 36 与写 47 的目录后缀
PRED_KEYS = ["pred_full", "pred_last", "pred_align", "ig", "attn",
             "M2_mean_act_task", "random"]
PRED_KEYS_SHAP = ["shap"] + PRED_KEYS


# ----------------------------------------------------------------------------
# 组合发现: 36 主链中有效的中层合成组合 (ioi / gt)
# ----------------------------------------------------------------------------
def discover_combos(base_dir, config, models, tasks):
    root = base_dir / "output" / "phaseStep2" / f"causal_validity{OUT_TAG}"
    combos = []
    for m in models:
        mdir = root / m
        if not mdir.is_dir():
            continue
        try:
            _, model_cfg = resolve_model_cfg(config, m)
            mid = int(model_cfg["target_layers"][1])
        except Exception:
            mid = None
        for ldir in sorted(mdir.glob("layer*")):
            try:
                layer = int(ldir.name.replace("layer", ""))
            except ValueError:
                continue
            if mid is not None and layer != mid:
                continue
            for task in tasks:
                f = ldir / f"{task}.json"
                if not f.exists():
                    continue
                try:
                    rec = json.load(open(f, encoding="utf-8"))
                except Exception:
                    continue
                if rec.get("error") or rec.get("at_chance"):
                    continue
                feats = [r for r in (rec.get("features") or [])
                         if r.get("C_ablate") is not None]
                if len(feats) < 20:
                    continue
                combos.append({"model": m, "layer": layer, "task": task,
                               "block_idx": rec.get("block_idx"),
                               "b0": rec.get("b0"), "b0_acc": rec.get("b0_acc"),
                               "src": str(f), "n_features": len(feats)})
    return combos


def load_targets(src_path):
    rec = json.load(open(src_path, encoding="utf-8"))
    rows = []
    for r in rec.get("features") or []:
        if r.get("C_ablate") is None:
            continue
        rows.append({
            "feature_idx": int(r["feature_idx"]),
            "C_ablate": float(r["C_ablate"]),
            "B_ablate": float(r["B_ablate"]) if r.get("B_ablate") is not None else None,
            "M2_mean_act_task": (float(r["M2_mean_act_task"])
                                 if r.get("M2_mean_act_task") is not None else None),
            "M1_act_freq_task": r.get("M1_act_freq_task"),
            "M3_max_act_task": r.get("M3_max_act_task"),
            "M4_recon_mse_gain": r.get("M4_recon_mse_gain"),
            "M5_ctx_coherence": r.get("M5_ctx_coherence"),
            "M6_corpus_freq": r.get("M6_corpus_freq"),
        })
    return rec, rows


# ----------------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------------
def _behavior(logits, chunk):
    cid = torch.tensor([it["correct_id"] for it in chunk], device=logits.device)
    did = torch.tensor([it["distractor_id"] for it in chunk], device=logits.device)
    return (logits.gather(1, cid[:, None]) - logits.gather(1, did[:, None])).squeeze(1)


def _leaf_hook(cap, key="h"):
    """把 block 输出转成可求导的叶子 (float32), 再按原 dtype 送回模型。

    这样即使所有参数 requires_grad=False, 也能对 h 求 ∂B/∂h, 并保证数值不变。
    """
    def hk(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        dt = hs.dtype
        h32 = hs.detach().float().requires_grad_(True)
        cap[key] = h32
        out = h32.to(dt)
        return (out,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else out
    return hk


def _sae_sparse(sae, h, mean_t, std_t):
    with torch.no_grad():
        sparse, _ = sae.encode((h.float() - mean_t) / std_t)
    return sparse


# ----------------------------------------------------------------------------
# (a) 一阶解析预测器
# ----------------------------------------------------------------------------
def first_order_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t,
                      Wsc, feat_idx, device, batch_size=32):
    D, F = Wsc.shape
    idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
    tf_sum = np.zeros(F, dtype=np.float64)
    tl_sum = np.zeros(F, dtype=np.float64)
    g_sum = np.zeros(D, dtype=np.float64)
    b0s = []
    n_tok = 0
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
        cap = {}
        handle = layer_mod.register_forward_hook(_leaf_hook(cap, "h"))
        try:
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        finally:
            handle.remove()
        h = cap["h"]
        logits = out.logits[:, -1, :].float()
        B = _behavior(logits, chunk)
        b0s.extend(B.detach().cpu().tolist())
        g = torch.autograd.grad(B.sum(), h)[0].float() * attn.unsqueeze(-1).float()
        sparse = _sae_sparse(sae, h.detach(), mean_t, std_t).index_select(-1, idx_t)
        tf = torch.einsum("btd,df,btf->bf", g, Wsc, sparse)
        tl = torch.einsum("bd,df,bf->bf", g[:, -1, :], Wsc, sparse[:, -1, :])
        tf_sum += tf.sum(0).cpu().double().numpy()
        tl_sum += tl.sum(0).cpu().double().numpy()
        g_sum += g.sum(dim=(0, 1)).cpu().double().numpy()
        n_tok += int(attn.sum().item())
        del h, logits, B, g, sparse, tf, tl, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    n = max(len(b0s), 1)
    pred_full = -(tf_sum / n)
    pred_last = -(tl_sum / n)
    g_bar = g_sum / max(n_tok, 1)
    Wsc_np = Wsc.detach().cpu().double().numpy()
    return {"pred_full": pred_full, "pred_last": pred_last,
            "pred_align": np.abs(Wsc_np.T @ g_bar),
            "pred_align_signed": -(Wsc_np.T @ g_bar),
            "b0": float(np.mean(b0s)), "b0_list": b0s}


# ----------------------------------------------------------------------------
# (b) Integrated Gradients (SAE 重构基线, 直线路径)
#   路径: h(α) = h - (1-α)·R, R = Σ_f s_f·W_dec[:,f]⊙std  (= 36 的全特征 ablation 位移)
#   归因: IG_f = Σ_t s_f(t)·(Wsc[:,f]·ḡ_t), ḡ 为路径上 m 步平均梯度
#   恒等式: Σ_f IG_f = B(h) - B(h-R)  (全特征 ablation 的总效应)
# ----------------------------------------------------------------------------
def ig_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t, Wsc,
             feat_idx, device, batch_size=16, steps=8):
    F = Wsc.shape[1]
    idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
    term = np.zeros(F, dtype=np.float64)
    b0s = []
    sum_check = 0.0
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
        cap = {}
        handle = layer_mod.register_forward_hook(_leaf_hook(cap, "h"))
        try:
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        finally:
            handle.remove()
        h = cap["h"]
        logits = out.logits[:, -1, :].float()
        B0 = _behavior(logits, chunk)
        b0s.extend(B0.detach().cpu().tolist())
        with torch.no_grad():
            sparse = sae.encode((h.detach() - mean_t) / std_t)[0].index_select(-1, idx_t)
            R = (sparse @ Wsc.t()).detach()                       # (B,T,D)
        del out, logits, B0, cap
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for k in range(steps):
            alpha = (k + 1.0) / steps
            cap2 = {}

            def hk(module, args, output):
                hs = output[0] if isinstance(output, (tuple, list)) else output
                dt = hs.dtype
                hcur = hs.detach().float()
                leaf = (hcur - (1.0 - alpha) * R).requires_grad_(True)
                cap2["leaf"] = leaf
                out2 = leaf.to(dt)
                return (out2,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else out2

            handle = layer_mod.register_forward_hook(hk)
            try:
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            finally:
                handle.remove()
            logits = out.logits[:, -1, :].float()
            B = _behavior(logits, chunk)
            g = torch.autograd.grad(B.sum(), cap2["leaf"])[0].float()
            g = g * attn.unsqueeze(-1).float()
            with torch.no_grad():
                t = torch.einsum("btd,df,btf->bf", g, Wsc, sparse)
                sum_check += float(t.sum().item()) / steps
            term += t.sum(0).cpu().double().numpy()
            del out, logits, B, g, t, cap2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del h, R, sparse
    n = max(len(b0s), 1)
    ig_f = term / (steps * n)
    ig_signed = -ig_f
    return {"ig_signed": ig_signed, "ig": np.abs(ig_signed),
            "ig_sum_signed": float(ig_signed.sum()),
            "ig_sum_check": float(sum_check / n)}


# ----------------------------------------------------------------------------
# (c) KernelSHAP: 多特征联合 ablation 的价值函数
# ----------------------------------------------------------------------------
def shap_weight(s, F):
    if s <= 0 or s >= F:
        return 1e6
    from math import comb
    return (F - 1.0) / (comb(F, s) * s * (F - s))


def build_coalitions(F, K, seed):
    rng = np.random.RandomState(seed)
    rows, seen = [], set()

    def add(row):
        t = tuple(int(x) for x in row)
        if t in seen:
            return False
        seen.add(t)
        rows.append(row)
        return True

    add(np.zeros(F))
    sizes = list(range(1, F))
    budget = max(0, K - 2)
    base, extra = budget // len(sizes), budget % len(sizes)
    for i, s in enumerate(sizes):
        need = base + (1 if i < extra else 0)
        got, tries = 0, 0
        while got < need and tries < max(200, need * 200):
            tries += 1
            row = np.zeros(F)
            row[rng.choice(F, size=s, replace=False)] = 1.0
            if add(row):
                got += 1
    add(np.ones(F))
    Z = np.stack(rows).astype(np.float64)
    w = np.array([shap_weight(int(z.sum()), F) for z in Z], dtype=np.float64)
    return Z, w


def kernel_shap_exact(Z, y, w, total):
    """带精确效率约束的加权最小二乘: min Σ w_k (φ·z_k - y_k)^2, s.t. Σφ=total.

    消去最后一个变量后解无约束 LS, 数值稳定且严格满足约束。
    """
    F = Z.shape[1]
    sw = np.sqrt(w)
    zf = Z[:, F - 1]
    y2 = (y - zf * total) * sw
    Z2 = (Z[:, :F - 1] - zf[:, None]) * sw[:, None]
    phi_rest, *_ = np.linalg.lstsq(Z2, y2, rcond=None)
    phi = np.zeros(F)
    phi[:F - 1] = phi_rest
    phi[F - 1] = total - float(phi_rest.sum())
    return phi


def make_multi_ablate_hook(sae, mean_t, std_t, idx_t, mask):
    """多特征联合 ablation: Δh = -Σ_{f∈mask} s_f·(W_dec[:,f]⊙std).

    逐特征累加, 且沿用 36 的 element-wise 乘法顺序 (coef·W_f·std), 而不是
    coef @ Wsc.t() 一次 matmul 或预乘 Wsc=W⊙std: BLAS 的分块累加与乘法结合序
    与 36 相差 ~3e-8, 经 bf16 残差流与 bf16 logits 放大后会表现为 1 个 logit
    ULP (≈0.06) 的 B 偏差. 此实现使单特征 mask 与 36 逐位一致, 保证 SHAP
    单特征联盟可精确复现落盘的 B_ablate.
    """
    W_dec_s = sae.decoder.weight.index_select(1, idx_t).detach().float()
    std_v = std_t.view(1, 1, -1)

    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        dt = hs.dtype
        h = hs.float()
        with torch.no_grad():
            sparse, _ = sae.encode((h - mean_t) / std_t)
            coef = sparse.index_select(-1, idx_t) * mask          # (B,T,Fs)
            delta = torch.zeros_like(h)
            for i in range(W_dec_s.shape[1]):
                if float(mask[i]) == 0.0:
                    continue
                delta = delta - coef[..., i:i + 1] * W_dec_s[:, i].view(1, 1, -1) * std_v
        hn = (h + delta).to(dt)
        return (hn,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else hn
    return hook


def shap_value(model, tokenizer, items, layer_mod, hook_fn, device, batch_size=32):
    vals = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
        handle = layer_mod.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn,
                               use_cache=False).logits[:, -1, :].float()
        finally:
            handle.remove()
        vals.extend(_behavior(logits, chunk).cpu().tolist())
        del logits
    return float(np.mean(vals)), vals


def shap_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t, Wsc,
               shap_idx, n_coalitions, seed, device, batch_size=32):
    F = len(shap_idx)
    idx_t = torch.tensor(shap_idx, dtype=torch.long, device=device)
    Z, w = build_coalitions(F, n_coalitions, seed)
    v = np.zeros(Z.shape[0])
    for i in range(Z.shape[0]):
        mask = torch.tensor(Z[i], dtype=torch.float32, device=device)
        hk = make_multi_ablate_hook(sae, mean_t, std_t, idx_t, mask)
        v[i], _ = shap_value(model, tokenizer, items, layer_mod, hk, device, batch_size)
    v_empty, v_full = float(v[0]), float(v[-1])
    y = v - v_empty
    phi = kernel_shap_exact(Z, y, w, v_full - v_empty)
    return {"phi": phi, "v": v, "Z": Z, "n_coalitions": int(Z.shape[0]),
            "v_empty": v_empty, "v_full": v_full,
            "sum_phi": float(phi.sum()),
            "efficiency_gap": float(abs(phi.sum() - (v_full - v_empty)))}


# ----------------------------------------------------------------------------
# (d) 注意力归因: 末层各头平均、末位 token 对各位置的注意力
# ----------------------------------------------------------------------------
def attention_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t, Wsc,
                    feat_idx, device, batch_size=16):
    F = Wsc.shape[1]
    idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
    signed = np.zeros(F, dtype=np.float64)
    n = 0
    prev_impl = getattr(model.config, "_attn_implementation", None)
    if prev_impl == "sdpa":
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            pass
    try:
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
            cap = {}
            handle = layer_mod.register_forward_hook(_leaf_hook(cap, "h"))
            try:
                out = model(input_ids=input_ids, attention_mask=attn,
                            output_attentions=True, use_cache=False)
            finally:
                handle.remove()
            if not getattr(out, "attentions", None):
                return {"available": False, "attn": None, "attn_signed": None,
                        "reason": "model returned empty attentions "
                                  f"(attn_impl={getattr(model.config, '_attn_implementation', None)})"}
            a = out.attentions[-1].float().mean(dim=1)[:, -1, :]   # (B,T)
            h = cap["h"]
            logits = out.logits[:, -1, :].float()
            B = _behavior(logits, chunk)
            g = torch.autograd.grad(B.sum(), h)[0].float() * attn.unsqueeze(-1).float()
            sparse = _sae_sparse(sae, h.detach(), mean_t, std_t).index_select(-1, idx_t)
            with torch.no_grad():
                contrib = torch.einsum("btd,df->btf", g, Wsc) * sparse
                vals = torch.einsum("bt,btf->bf", a, contrib)
            signed += vals.sum(0).cpu().double().numpy()
            n += int(vals.shape[0])
            del out, h, logits, B, g, sparse, contrib, vals, a, cap
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:                                        # noqa: BLE001
        return {"available": False, "attn": None, "attn_signed": None,
                "reason": f"{type(e).__name__}: {e}"}
    finally:
        if prev_impl is not None:
            try:
                model.config._attn_implementation = prev_impl
            except Exception:
                pass
    signed = -(signed / max(n, 1))
    return {"available": True, "attn_signed": signed, "attn": np.abs(signed),
            "reason": None}


# ----------------------------------------------------------------------------
# 单组合
# ----------------------------------------------------------------------------
def run_one(args, base_dir, m, combo, tag):
    layer, task = combo["layer"], combo["task"]
    out_dir = (base_dir / "output" / "phaseStep2" / f"causal_attribution{OUT_TAG}"
               / m / f"layer{layer}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"
    resume_file = out_dir / f"resume_47_{task}_layer{layer}.json"
    if out_file.exists() and resume_file.exists() and not args.force:
        try:
            if json.load(open(resume_file, encoding="utf-8")).get("status") == "completed":
                print(f"  [跳过] {m} L{layer} {task} 已完成")
                return json.load(open(out_file, encoding="utf-8"))
        except Exception:
            pass

    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rec36, rows = load_targets(combo["src"])
    feat_idx = [r["feature_idx"] for r in rows]
    result = {"model": m, "layer": layer, "task": task,
              "timestamp": datetime.now().isoformat(),
              "block_idx": combo["block_idx"], "n_features": len(feat_idx),
              "b0_36": combo["b0"], "b0_acc_36": combo["b0_acc"],
              "n_prompts": args.n_prompts, "seed": args.seed,
              "shap_top": args.shap_top, "shap_coalitions": args.shap_coalitions,
              "ig_steps": args.ig_steps}

    loaded = load_per_layer_sae(base_dir, m, layer, device)
    if loaded is None:
        result["error"] = "缺少 per-layer SAE"
        _save(out_file, result)
        return result
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)
    W = sae.decoder.weight.detach().float()                       # (D, hidden)
    Wsc = W * std_t.view(-1, 1)                                   # (D, F_all)

    model, tokenizer, layer_mod = tag["model"], tag["tokenizer"], tag["layer_mod"][layer]

    items, task_meta = TASK_BUILDERS[task](tokenizer, args.n_prompts, seed=args.seed)
    if not items:
        result["error"] = f"任务构造失败: {task_meta}"
        _save(out_file, result)
        return result
    result["task_meta_n"] = len(items)

    sel_W = Wsc.index_select(1, torch.tensor(feat_idx, dtype=torch.long, device=device))
    fo = first_order_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t,
                           sel_W, feat_idx, device, args.batch_size)
    result["b0_here"] = fo["b0"]
    result["b0_abs_diff_vs_36"] = abs(fo["b0"] - (combo["b0"] if combo["b0"] is not None else 0.0))
    print(f"  [一阶] b0={fo['b0']:.4f} (36={combo['b0']:.4f}, "
          f"|Δ|={result['b0_abs_diff_vs_36']:.2e})")

    attn_res = attention_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t,
                               sel_W, feat_idx, device, min(args.batch_size, 16))
    print(f"  [注意力] available={attn_res['available']} "
          + (f"({attn_res['reason']})" if not attn_res["available"] else ""))

    ig_res = ig_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t,
                      sel_W, feat_idx, device, args.batch_size, args.ig_steps)
    result["ig_sum_signed"] = ig_res["ig_sum_signed"]
    result["ig_sum_check"] = ig_res["ig_sum_check"]
    print(f"  [IG] 完成; ΣIG_signed={ig_res['ig_sum_signed']:+.4f} "
          f"(路径积分 {ig_res['ig_sum_check']:+.4f})")

    # ---- SHAP 子集: 按 M2 取 top-F ----
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i]["M2_mean_act_task"]
                                  if rows[i]["M2_mean_act_task"] is not None else -1.0),
                   reverse=True)
    shap_pos = order[:args.shap_top]
    shap_idx = [feat_idx[i] for i in shap_pos]
    shap_res = shap_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t, Wsc,
                          shap_idx, args.shap_coalitions,
                          args.seed + 4700 + layer, device, args.batch_size)
    phi = shap_res["phi"]
    print(f"  [SHAP] F={len(shap_idx)} K={shap_res['n_coalitions']} "
          f"v∅={shap_res['v_empty']:.4f} v全={shap_res['v_full']:.4f} "
          f"效率gap={shap_res['efficiency_gap']:.2e}")

    # ---- 校验: 空联盟 == b0; 单特征联盟 == 36 B_ablate (前 2 个) ----
    check = {"shap_empty_vs_b0": abs(shap_res["v_empty"] - fo["b0"]),
             "singleton_vs_36": []}
    idx_t = torch.tensor(shap_idx, dtype=torch.long, device=device)
    Wsc_s = Wsc.index_select(1, idx_t)
    if not args.skip_checks:
        for j in range(min(2, len(shap_idx))):
            if rows[shap_pos[j]]["B_ablate"] is None:
                continue
            mask = torch.zeros(len(shap_idx), device=device)
            mask[j] = 1.0
            hk = make_multi_ablate_hook(sae, mean_t, std_t, idx_t, mask)
            bj, _ = shap_value(model, tokenizer, items, layer_mod, hk, device,
                               args.batch_size)
            check["singleton_vs_36"].append({
                "feature_idx": int(shap_idx[j]), "B_here": float(bj),
                "B_36": float(rows[shap_pos[j]]["B_ablate"]),
                "abs_diff": float(abs(bj - rows[shap_pos[j]]["B_ablate"]))})
    result["checks"] = check
    result["shap"] = {"feature_idx": [int(x) for x in shap_idx],
                      "phi": [float(x) for x in phi],
                      "n_coalitions": shap_res["n_coalitions"],
                      "v_empty": shap_res["v_empty"], "v_full": shap_res["v_full"],
                      "sum_phi": shap_res["sum_phi"],
                      "efficiency_gap": shap_res["efficiency_gap"]}
    result["attention_available"] = attn_res["available"]
    if not attn_res["available"]:
        result["attention_reason"] = attn_res["reason"]

    # ---- 合并逐特征表 ----
    b0 = combo["b0"] if combo["b0"] is not None else fo["b0"]
    rng = np.random.RandomState(args.seed + 47)
    rand_scores = rng.normal(size=len(rows))
    shap_map = {int(f): float(p) for f, p in zip(shap_idx, phi)}
    rows_out = []
    for i, r in enumerate(rows):
        dB = (r["B_ablate"] - b0) if r["B_ablate"] is not None else None
        row = {
            "feature_idx": r["feature_idx"], "C_ablate": r["C_ablate"],
            "dB_measured": dB,
            "pred_full_v": float(fo["pred_full"][i]),
            "pred_full": float(abs(fo["pred_full"][i])),
            "pred_last_v": float(fo["pred_last"][i]),
            "pred_last": float(abs(fo["pred_last"][i])),
            "pred_align": float(fo["pred_align"][i]),
            "ig_v": float(ig_res["ig_signed"][i]),
            "ig": float(ig_res["ig"][i]),
            "attn_v": (float(attn_res["attn_signed"][i])
                       if attn_res["available"] else None),
            "attn": (float(attn_res["attn"][i]) if attn_res["available"] else None),
            "shap": shap_map.get(r["feature_idx"]),
            "shap_abs": (abs(shap_map[r["feature_idx"]])
                         if r["feature_idx"] in shap_map else None),
            "random": float(abs(rand_scores[i])),
            "in_shap_subset": bool(r["feature_idx"] in shap_map),
        }
        for mm in METRICS:
            row[mm] = r.get(mm)
        rows_out.append(row)

    # ---- 相关性: 全特征集 / SHAP 子集 ----
    def _corr(sel_rows):
        out = {}
        for k in PRED_KEYS:
            xs, ys = [], []
            for row in sel_rows:
                v = row.get(k)
                if v is None or row["C_ablate"] is None:
                    continue
                xs.append(v); ys.append(row["C_ablate"])
            out[k] = {"spearman": spearman(xs, ys), "n": len(xs)}
        if any(row["shap_abs"] is not None for row in sel_rows):
            xs = [row["shap_abs"] for row in sel_rows if row["shap_abs"] is not None]
            ys = [row["C_ablate"] for row in sel_rows if row["shap_abs"] is not None]
            out["shap"] = {"spearman": spearman(xs, ys), "n": len(xs)}
        return out

    corr_full = _corr(rows_out)
    corr_shap = _corr([row for row in rows_out if row["in_shap_subset"]])
    result["correlations_full"] = corr_full
    result["correlations_shap"] = corr_shap
    result["features"] = rows_out
    result["runtime_sec"] = round(time.time() - t_start, 1)
    _save(out_file, result)
    _save(resume_file, {"status": "completed", "completed_at": datetime.now().isoformat()})

    print("  [相关/全] " + "; ".join(
        f"{k}={v['spearman']:.3f}" for k, v in corr_full.items() if v.get("spearman") is not None))
    print("  [相关/SHAP子集] " + "; ".join(
        f"{k}={v['spearman']:.3f}" for k, v in corr_shap.items() if v.get("spearman") is not None))
    return result


# ----------------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------------
def _pooled(records, key, only_shap=False):
    kcol = "shap_abs" if key == "shap" else key
    xs_all, ys_all, per_combo = [], [], []
    for r in records:
        rows = [row for row in (r.get("features") or [])
                if row.get("C_ablate") is not None and row.get(kcol) is not None
                and (row.get("in_shap_subset") or not only_shap)]
        if len(rows) < 5:
            continue
        xs = np.array([row[kcol] for row in rows], dtype=np.float64)
        ys = np.array([row["C_ablate"] for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            continue
        if xs.std() <= 1e-12 or ys.std() <= 1e-12:
            continue
        xs_all.append((xs - xs.mean()) / xs.std())
        ys_all.append((ys - ys.mean()) / ys.std())
        per_combo.append(spearman(xs.tolist(), ys.tolist()))
    if not xs_all:
        return None
    X, Y = np.concatenate(xs_all), np.concatenate(ys_all)
    return {"pooled_spearman": spearman(X.tolist(), Y.tolist()),
            "pooled_pearson": pearson(X.tolist(), Y.tolist()),
            "mean_combo_spearman": float(np.mean(per_combo)),
            "std_combo_spearman": float(np.std(per_combo)),
            "n_combos": len(per_combo), "n_features": int(len(X)),
            "per_combo": [float(v) for v in per_combo]}


def _paired(records, a, b, field):
    diffs, xs, ys = [], [], []
    for r in records:
        ca = ((r.get(field) or {}).get(a) or {}).get("spearman")
        cb = ((r.get(field) or {}).get(b) or {}).get("spearman")
        if ca is None or cb is None:
            continue
        xs.append(float(ca)); ys.append(float(cb)); diffs.append(float(ca) - float(cb))
    if len(diffs) < 3:
        return None
    diffs = np.array(diffs)
    from scipy import stats
    p = None
    if np.any(np.abs(diffs) > 0):
        try:
            p = float(stats.wilcoxon(diffs, zero_method="zsplit").pvalue)
        except Exception:
            p = None
    rng = np.random.RandomState(42)
    boot = [float(np.mean(rng.choice(diffs, size=len(diffs), replace=True)))
            for _ in range(5000)]
    return {"a": a, "b": b, "n": int(len(diffs)),
            "mean_a": float(np.mean(xs)), "mean_b": float(np.mean(ys)),
            "mean_delta": float(np.mean(diffs)),
            "delta_ci95": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
            "win": int((diffs > 0).sum()), "loss": int((diffs < 0).sum()),
            "wilcoxon_p": p}


def summarize(records, out_dir, meta):
    summary = {"meta": meta, "pooled_full": {}, "pooled_shap": {},
               "paired_shap": [], "paired_full": [], "per_combo": []}
    for k in PRED_KEYS:
        s = _pooled(records, k, only_shap=False)
        if s:
            summary["pooled_full"][k] = s
    for k in PRED_KEYS_SHAP:
        s = _pooled(records, k, only_shap=True)
        if s:
            summary["pooled_shap"][k] = s

    pairs_shap = [("shap", "pred_full"), ("shap", "M2_mean_act_task"),
                  ("shap", "ig"), ("shap", "attn"), ("shap", "random"),
                  ("pred_full", "M2_mean_act_task"), ("ig", "pred_full"),
                  ("ig", "M2_mean_act_task"), ("pred_full", "random")]
    pairs_full = [("pred_full", "M2_mean_act_task"), ("pred_full", "ig"),
                  ("pred_full", "attn"), ("pred_full", "pred_last"),
                  ("ig", "M2_mean_act_task"), ("attn", "M2_mean_act_task"),
                  ("pred_full", "random")]
    for a, b in pairs_shap:
        p = _paired(records, a, b, "correlations_shap")
        if p:
            summary["paired_shap"].append(p)
    for a, b in pairs_full:
        p = _paired(records, a, b, "correlations_full")
        if p:
            summary["paired_full"].append(p)

    for r in records:
        summary["per_combo"].append({
            "combo": f"{r['model']}|L{r['layer']}|{r['task']}",
            "n_features": r.get("n_features"),
            "b0": r.get("b0_36"), "b0_here": r.get("b0_here"),
            "b0_abs_diff": r.get("b0_abs_diff_vs_36"),
            "shap_efficiency_gap": (r.get("shap") or {}).get("efficiency_gap"),
            "corr_full": {k: (v or {}).get("spearman")
                          for k, v in (r.get("correlations_full") or {}).items()},
            "corr_shap": {k: (v or {}).get("spearman")
                          for k, v in (r.get("correlations_shap") or {}).items()},
            "runtime_sec": r.get("runtime_sec"),
        })
    _save(out_dir / "attribution_summary.json", summary)

    L = ["# Step 4: 昂贵归因基线 vs 零干预一阶预测器", "",
         f"- 生成时间: {meta['timestamp']}",
         f"- 组合数: {len(records)} (mid 层, seed={meta['seed']}, "
         f"n_prompts={meta['n_prompts']})",
         f"- KernelSHAP: top-{meta['shap_top']} 按 M2 选择, "
         f"{meta['shap_coalitions']} 联盟; IG 步数 {meta['ig_steps']}",
         "- 真值: 36 落盘 C_ablate (200 prompts, 单特征 ablation)", "",
         "## 1. Pooled 相关 (全特征集, 逐组合 z-score)", "",
         "| 预测器 | pooled spearman | 组合均值±std | n 特征 |",
         "|---|---|---|---|"]
    for k, s in summary["pooled_full"].items():
        L.append(f"| {k} | {s['pooled_spearman']:.3f} | "
                 f"{s['mean_combo_spearman']:.3f}±{s['std_combo_spearman']:.3f} | "
                 f"{s['n_features']} |")
    L += ["", "## 2. Pooled 相关 (SHAP 子集 top-16)", "",
          "| 预测器 | pooled spearman | 组合均值±std | n |", "|---|---|---|---|"]
    for k, s in summary["pooled_shap"].items():
        L.append(f"| {k} | {s['pooled_spearman']:.3f} | "
                 f"{s['mean_combo_spearman']:.3f}±{s['std_combo_spearman']:.3f} | "
                 f"{s['n_features']} |")

    def _ptab(pairs, title):
        L.extend(["", f"## {title}", "",
                  "| 对比 | 均值 A | 均值 B | Δ | Δ95%CI | 胜/负 | Wilcoxon p |",
                  "|---|---|---|---|---|---|---|"])
        for p in pairs:
            ci = p["delta_ci95"]
            pv = "—" if p["wilcoxon_p"] is None else f"{p['wilcoxon_p']:.4f}"
            L.append(f"| {p['a']} − {p['b']} | {p['mean_a']:.3f} | {p['mean_b']:.3f} | "
                     f"{p['mean_delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                     f"{p['win']}/{p['loss']} | {pv} |")
    _ptab(summary["paired_shap"], "3. 逐组合配对 (SHAP 子集, n=16 特征/组合)")
    _ptab(summary["paired_full"], "4. 逐组合配对 (全特征集)")

    L += ["", "## 5. 逐组合", "",
          "| 组合 | n | b0 | b0Δ36 | SHAP | pred_full | IG | attn | M2 | 随机 | eff.gap |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["per_combo"]:
        cs = r["corr_shap"]
        cf = r["corr_full"]

        def f(v):
            return "—" if v is None else f"{v:.3f}"
        L.append(f"| {r['combo']} | {r['n_features']} | {f(r['b0'])} | "
                 f"{(r['b0_abs_diff'] if r['b0_abs_diff'] is not None else float('nan')):.2e} | "
                 f"{f(cs.get('shap'))} | {f(cf.get('pred_full'))} | {f(cf.get('ig'))} | "
                 f"{f(cf.get('attn'))} | {f(cf.get('M2_mean_act_task'))} | "
                 f"{f(cf.get('random'))} | "
                 f"{(r['shap_efficiency_gap'] if r['shap_efficiency_gap'] is not None else float('nan')):.1e} |")

    L += ["", "## 6. 判定", ""]
    pf = (summary["pooled_full"].get("pred_full") or {}).get("pooled_spearman")
    ig = (summary["pooled_full"].get("ig") or {}).get("pooled_spearman")
    at = (summary["pooled_full"].get("attn") or {}).get("pooled_spearman")
    m2 = (summary["pooled_full"].get("M2_mean_act_task") or {}).get("pooled_spearman")
    sh = (summary["pooled_shap"].get("shap") or {}).get("pooled_spearman")
    pf_s = (summary["pooled_shap"].get("pred_full") or {}).get("pooled_spearman")
    L.append(f"- 全特征集: pred_full={_fmt(pf)}, IG={_fmt(ig)}, attn={_fmt(at)}, M2={_fmt(m2)}")
    L.append(f"- SHAP 子集: SHAP={_fmt(sh)}, pred_full={_fmt(pf_s)}")
    for p in summary["paired_shap"]:
        if p["a"] == "shap" and p["b"] == "pred_full":
            if p["mean_delta"] > 0 and (p["wilcoxon_p"] or 1) < 0.05:
                L.append("- **结论**: KernelSHAP 显著优于便宜的一阶预测器 —— "
                         "应把 SHAP 作为主归因口径, pred_full 作为快速近似。")
            elif p["mean_delta"] < 0 and (p["wilcoxon_p"] or 1) < 0.05:
                L.append("- **结论**: 便宜的一阶预测器显著优于 KernelSHAP —— "
                         "解析近似不仅更快, 而且排序更准 (SHAP 方差是主因)。")
            else:
                L.append("- **结论**: 二者无显著差异 —— 一阶解析近似已达到昂贵归因的排序水平。")
    if pf is not None and m2 is not None:
        L.append(f"- 一阶预测器相对 M2 的增益: {pf - m2:+.3f}")
    (out_dir / "attribution_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return summary


def _fmt(v):
    return "—" if v is None else f"{v:+.3f}"


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Step 4: 昂贵归因基线 vs 一阶预测器")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--tasks", type=str, default="ioi,gt")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shap_top", type=int, default=16)
    ap.add_argument("--shap_coalitions", type=int, default=128)
    ap.add_argument("--ig_steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tag", type=str, default="")
    ap.add_argument("--limit", type=int, default=0, help="每模型最多组合数 (调试)")
    ap.add_argument("--skip_checks", action="store_true",
                    help="跳过单特征联盟与 36 B_ablate 的一致性检查")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global OUT_TAG
    OUT_TAG = args.out_tag

    config = load_config(args.config)
    base_dir = Path(args.config).parent
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [args.model] if args.model else list(config["models"].keys())

    out_root = base_dir / "output" / "phaseStep2" / f"causal_attribution{OUT_TAG}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(out_root / "logs" / f"47_attribution_{ts}.log"))
    sys.stdout = log_saver

    combos = discover_combos(base_dir, config, models, tasks)
    print(f"[发现] {len(combos)} 个有效组合: " + ", ".join(
        f"{c['model']}|L{c['layer']}|{c['task']}" for c in combos))
    if not combos:
        print("[退出] 没有可用组合")
        log_saver.close()
        sys.stdout = log_saver.terminal
        return

    records = []
    try:
        by_model = {}
        for c in combos:
            by_model.setdefault(c["model"], []).append(c)
        for m, cs in by_model.items():
            if args.limit:
                cs = cs[:args.limit]
            _, model_cfg = resolve_model_cfg(config, m)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_path = base_dir / model_cfg["local_path"]
            if not model_path.exists():
                for p in (base_dir / "models").glob("*/*"):
                    if p.is_dir() and (p / "config.json").exists():
                        model_path = p
                        break
            print(f"\n===== [{m}] 加载基座模型 ({len(cs)} 个组合) =====")
            if model_needs_fp8(model_cfg, model_path):
                try:
                    from huggingface_hub import constants as _hf_const
                    _hf_const.HF_HUB_OFFLINE = True
                except Exception as _e:
                    print(f"  [fp8] 设置离线模式失败: {_e}")
                try:
                    os.environ.setdefault(
                        "TRITON_CACHE_DIR",
                        os.path.join(os.path.expanduser("~"), ".cache", "triton"))
                except Exception:
                    pass
                try:
                    import huggingface_hub._snapshot_download as _sd
                    _sd.read_tree_cache = lambda *a, **k: None
                except Exception as _e:
                    print(f"  [fp8] patch read_tree_cache 失败: {_e}")
            model, tokenizer = load_model_for_generation(base_dir, model_cfg, model_path)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            if model_needs_fp8(model_cfg, model_path):
                fp8_dequant_selfcheck(model, tokenizer, device)
            try:
                model.config.use_cache = False
            except Exception:
                pass
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
            for p in model.parameters():
                p.requires_grad_(False)
            layers_mod = find_decoder_layers(model)
            probe = "When Mary and John went to the store, John gave a drink to"
            lmap = calibrate_layer_map(model, tokenizer, layers_mod, device, probe)
            layer_mod = {}
            for c in cs:
                bi = c["block_idx"]
                if bi is None:
                    bi = (lmap.get(str(c["layer"])) or {}).get("block_idx")
                if bi is None or bi >= len(layers_mod):
                    print(f"  [跳过] {m} L{c['layer']}: 层对齐失败")
                    continue
                layer_mod[c["layer"]] = layers_mod[bi]
            tag = {"model": model, "tokenizer": tokenizer, "layer_mod": layer_mod}
            for c in cs:
                if c["layer"] not in layer_mod:
                    continue
                key = f"{m}|L{c['layer']}|{c['task']}"
                print(f"\n[{key}] n_feat={c['n_features']} b0={c['b0']:.3f}")
                try:
                    res = run_one(args, base_dir, c["model"], c, tag)
                except Exception as e:                            # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"[失败] {key}: {e}")
                    continue
                if res.get("error"):
                    print(f"  [错误] {res['error']}")
                    continue
                records.append(res)
                summarize(records, out_root, {
                    "timestamp": ts, "n_prompts": args.n_prompts, "seed": args.seed,
                    "shap_top": args.shap_top, "shap_coalitions": args.shap_coalitions,
                    "ig_steps": args.ig_steps, "models": models, "tasks": tasks})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        log_saver.close()
        sys.stdout = log_saver.terminal

    print(f"[完成] 有效组合={len(records)}")
    print(f"[输出] {out_root / 'attribution_summary.md'}")


if __name__ == "__main__":
    main()
