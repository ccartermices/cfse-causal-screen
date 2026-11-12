# -*- coding: utf-8 -*-
"""E20 — §5 的 48 组网格换 held-out 协议（选择-评估分离）

为什么必须重跑
--------------
§5 的主表 (pred_full ρ=+0.725, 55/60) 用的是**同一批 prompt** 既算分数又算真值。
分数本身是激活的函数，于是它部分预测了自己的评估集 —— §7.1 已量化这一泄漏
(qwen3.5-2b/IOI: 0.810 → 0.903)。§7 只在 18 个组合上做了分离，
§5 的 48 组网格仍是旧口径，这是全文最大的软肋。

为什么不能离线重建
-------------------
`causal_validity/{model}/layer{L}/{task}.json` 里的 C_ablate 是**按 prompt 平均后的标量**
(|mean_p ΔB|)，逐 prompt 的效应从未落盘，因此无法离线拆半重建。
必须重跑消融。

本脚本做什么
------------
对 48 组网格 (10 模型 × {ioi, gt} × 3 层)，在**一次前向**里同时取到：

  真值（两条口径，同一次前向）
    f32  : 钩 lm_head 输入，在 float32 下重算 correct/distractor 两个 logit
    bf16 : 模型原生 logits
    E[feature, prompt] = ΔB(prompt) 逐 prompt 保存

  分数（逐 prompt）
    pred_full[p, feature]  一阶预测器的**逐 prompt** 值（§5 的 gradient_probe 只给聚合量，
                            这里必须逐 prompt 存，否则拆半无从谈起）
    M2 / M3[p, feature]     逐 prompt 的激活均值 / 最大值

落盘
----
  output/phaseStep2/cfse_grid_heldout/{model}/layer{L}/{task}.npz
      E32, Ebf16 : (n_feat, n_prompt) 逐 prompt 真值
      pred_p     : (n_prompt, n_feat) 逐 prompt 预测（有符号）
      M2_p, M3_p : (n_prompt, n_feat) 逐 prompt 激活统计
      cand, b0_32, b0_16, meta
  output/phaseStep2/cfse_grid_heldout/{model}/layer{L}/{task}.json
      元信息 + κ / rel（bootstrap 口径）

拆半协议与 §7 完全一致
----------------------
  选择半集重算分数（M2/M3 取选择半集均值；pred 取选择半集逐 prompt 平均）
  评估半集的真值 = |mean_p E|
  rel = Spearman(选择半集真值, 评估半集真值)
  ratio = ρ / sqrt(rel)

用法
----
  python 77_grid_heldout.py --model qwen3.5-2b --layer 16 --task ioi
  python 77_grid_heldout.py --all            # 全 48 组
  python 77_grid_heldout.py --list           # 只列网格
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
_m48 = import_module("48_cfse_offline")
_m53 = import_module("53_reliability_curve")
_m57 = import_module("57_resolution_fix")
_m63 = import_module("63_e3_kappa_scan")

find_decoder_layers = _m27.find_decoder_layers
load_model_for_generation = _m27.load_model_for_generation

spearman = _m48.spearman

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_grid_heldout"
# config.json 的 local_path 是相对**仓库根**的（./models/...），与 63 一致
REPO = ROOT

# §5 的 48 组主网格（ioi + gt，10 模型 × 3 层），来自 causal_validity/summary.json
GRID: list[tuple[str, int, str]] = []
for _m in ("minicpm5-1b", "ministral-3-3b", "phi-3.5-mini", "pythia-1.4b",
           "pythia-2.8b", "pythia-6.9b", "qwen2.5-7b-instruct", "qwen3.5-0.8b",
           "qwen3.5-2b", "tinyllama-1.1b"):
    for _l, _t in ((8, "gt"), (8, "ioi"), (16, "gt"), (16, "ioi"), (22, "gt"), (22, "ioi"),
                   (10, "gt"), (10, "ioi"), (21, "gt"), (21, "ioi"), (25, "gt"), (25, "ioi"),
                   (30, "gt"), (30, "ioi"), (26, "gt"), (26, "ioi"), (7, "gt"), (7, "ioi"),
                   (18, "gt"), (18, "ioi"), (9, "gt"), (9, "ioi")):
        GRID.append((_m, _l, _t))
GRID = sorted(set(GRID))


# ----------------------------------------------------------------------------
# 网格：只保留 §5 里实际有效的 48 组（at_chance 的 phi ioi 三层要排除）
# ----------------------------------------------------------------------------
def build_grid() -> list[tuple[str, int, str]]:
    """从 causal_validity/summary.json 读取真实的 60/72 组，去掉 at_chance 与 error。"""
    p = BASE / "causal_validity" / "summary.json"
    if not p.exists():
        return sorted(set(GRID))
    d = json.load(open(p, encoding="utf-8"))["runs"]
    out = []
    for k, v in d.items():
        if v.get("error") or v.get("at_chance"):
            continue
        m, l, t = k.split("|")
        out.append((m, int(l.lstrip("L")), t))
    return sorted(set(out))


def loaded_features(model, layer, task):
    p = BASE / "causal_validity" / model / f"layer{layer}" / f"{task}.json"
    if not p.exists():
        return []
    d = json.load(open(p, encoding="utf-8"))
    feats = sorted(d.get("features", []), key=lambda r: -abs(r.get("M2_mean_act_task", 0.0)))
    return feats


# ----------------------------------------------------------------------------
# 逐 prompt 的一阶预测器
#
# 42.gradient_probe 只返回跨 prompt 聚合量（tf_sum / n_prompt），
# 而 held-out 需要**逐 prompt**的预测值才能拆半。这里重写一份：
#   pred_full[p, f] = -(1/|tokens(p)|) * Σ_t s_f(t)·(W_dec[:,f]⊙std)·g(t)
# 注意 42 的实现是 sum over 所有 prompt / n_prompt（平均），
# 逐 prompt 版则是每个 prompt 内部对 token 求平均 —— 两者在全体平均后一致。
# ----------------------------------------------------------------------------
def per_prompt_pred(model, tokenizer, items, layer_mod, sae, mean_t, std_t,
                    Wsc, feat_idx, device, batch_size=16):
    """返回 (pred_full (n_prompt, n_feat) 有符号, b0 (n_prompt,))。

    pred_full[p, f] = -(1/T_p) Σ_{t in prompt p} s_f(t)·(W_dec[:,f]⊙std)·g_p(t)
    T_p = prompt p 的 token 数（含 padding 位，g 已被 mask 置零，故等价于有效 token 数）
    """
    F = Wsc.shape[1]
    idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
    n = len(items)
    out_pred = np.zeros((n, F), dtype=np.float64)
    out_b0 = np.zeros(n, dtype=np.float64)
    for s in range(0, n, batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _m36._encode_with_prefix(tokenizer, chunk, device)
        cap = {}

        def hk(module, args, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            cap["h"] = hs

        handle = layer_mod.register_forward_hook(hk)
        try:
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :].float()
        finally:
            handle.remove()
        h = cap["h"]
        cid = torch.tensor([it["correct_id"] for it in chunk], device=logits.device)
        did = torch.tensor([it["distractor_id"] for it in chunk], device=logits.device)
        B = (logits.gather(1, cid[:, None]) - logits.gather(1, did[:, None])).squeeze(1)
        out_b0[s:s + len(chunk)] = B.detach().cpu().double().numpy()

        g = torch.autograd.grad(B.sum(), h)[0].float()
        mask = attn.unsqueeze(-1).float()
        g = g * mask
        with torch.no_grad():
            normed = (h.detach().float() - mean_t) / std_t
            sparse, _ = sae.encode(normed)
            s_sel = sparse.index_select(-1, idx_t)                  # (B,T,F)
            tf = torch.einsum("btd,df,btf->bf", g, Wsc, s_sel)      # (B,F)
        ntok = mask.sum(dim=(1, 2)).clamp(min=1.0)                 # (B,)
        out_pred[s:s + len(chunk)] = (-(tf / ntok.view(-1, 1))).detach().cpu().double().numpy()
        del g, logits, B, h, cap, sparse, s_sel, tf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out_pred, out_b0


# ----------------------------------------------------------------------------
# 真值：一次前向同时取 bf16 与 f32
# ----------------------------------------------------------------------------
def per_prompt_act(model, tokenizer, items, layer_mod, sae, mean_t, std_t,
                   cand, device, batch_size=16):
    """返回 (M2_p (n_prompt, n_feat), M3_p (n_prompt, n_feat))：逐 prompt 的激活均值/最大值。

    左 padding，所有 prompt 的 token 数不同 —— 逐 prompt 统计时必须各自取有效 token，
    不能把整个 batch 展平后统一平均（那样得到的是 batch 均值而非 prompt 均值）。
    """
    idx_t = torch.tensor(cand, dtype=torch.long, device=device)
    n = len(items)
    M2_p = np.zeros((n, len(cand)), dtype=np.float64)
    M3_p = np.zeros((n, len(cand)), dtype=np.float64)
    for s in range(0, n, batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _m36._encode_with_prefix(tokenizer, chunk, device)
        cap = {}

        def hk(module, args, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            cap["h"] = hs.detach().float()

        handle = layer_mod.register_forward_hook(hk)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            handle.remove()
        h = cap["h"]
        with torch.no_grad():
            normed = (h - mean_t) / std_t
            sparse, _ = sae.encode(normed)
            s_sel = sparse.index_select(-1, idx_t)               # (B,T,F)
        mask = attn.unsqueeze(-1).float()                        # (B,T,1)
        s_sel = s_sel * mask
        denom = mask.sum(dim=1).clamp(min=1.0)                   # (B,1)
        M2_p[s:s + len(chunk)] = (s_sel.sum(dim=1) / denom).cpu().double().numpy()
        M3_p[s:s + len(chunk)] = s_sel.max(dim=1).values.cpu().double().numpy()
        del h, cap, sparse, s_sel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return M2_p, M3_p


def baseline_b(model, tokenizer, items, layer_mod, device, batch_size=16):
    """无干预的 per-prompt 行为，f32 与 bf16 两条口径。"""
    vals32, b32 = _m57.batch_behavior_f32(model, tokenizer, items, layer_mod, None,
                                          device, batch_size)
    vals16, b16 = _m36.batch_behavior(model, tokenizer, items, layer_mod, None,
                                      device, batch_size)
    return np.asarray(vals32, dtype=np.float64), np.asarray(vals16, dtype=np.float64)


# ----------------------------------------------------------------------------
def kappa_of(E):
    """κ = sd_between / sd_within（与 n 无关）。"""
    sdm = E.mean(axis=1)
    sdv = E.std(axis=1, ddof=1)
    sb = float(sdm.std(ddof=1))
    sw = float(sdv.mean())
    return sb / max(sw, 1e-12)


def rel_of(c_sel, c_ev):
    return float(spearman(c_sel, c_ev))


def run_one(model, tok, m, layer, task, args, device):
    out_dir = OUT / m / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_p = out_dir / f"{task}.npz"
    json_p = out_dir / f"{task}.json"
    if npz_p.exists() and json_p.exists() and not args.force:
        print(f"  [跳过] {m} L{layer} {task}", flush=True)
        return json.load(open(json_p, encoding="utf-8"))

    layers_mod = find_decoder_layers(model)
    lmap = _m36.calibrate_layer_map(model, tok, layers_mod, device,
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
    meta = built[1] if isinstance(built, tuple) and len(built) > 1 else {}
    if not items:
        print(f"  [任务不可构造] {m} {task}", flush=True)
        return None

    feats = loaded_features(m, layer, task)
    if not feats:
        print(f"  [缺候选特征] {m} L{layer} {task}", flush=True)
        return None
    cand = [int(x["feature_idx"]) for x in feats][:args.n_features]

    print(f"  [{m} L{layer} {task}] prompts={len(items)} features={len(cand)}",
          flush=True)

    # ---- 1) 无干预基线（逐 prompt，两条口径）----
    b0_32, b0_16 = baseline_b(model, tok, items, layer_mod, device, args.batch_size)
    acc = float(np.mean(b0_32 > 0))
    print(f"    b0 f32={b0_32.mean():.4f} bf16={b0_16.mean():.4f} acc={acc:.3f}",
          flush=True)

    # ---- 2) 真值矩阵（逐 prompt）----
    t0 = time.time()
    E32 = np.zeros((len(cand), len(items)), dtype=np.float64)
    E16 = np.zeros((len(cand), len(items)), dtype=np.float64)
    for i, f in enumerate(cand):
        hk = _m36.make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        v32, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, hk,
                                         device, args.batch_size)
        v16, _ = _m36.batch_behavior(model, tok, items, layer_mod, hk,
                                     device, args.batch_size)
        E32[i] = np.asarray(v32, dtype=np.float64) - b0_32
        E16[i] = np.asarray(v16, dtype=np.float64) - b0_16
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(cand)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"    真值矩阵 {time.time()-t0:.0f}s", flush=True)

    # ---- 3) 逐 prompt 分数 ----
    t0 = time.time()
    W = sae.decoder.weight.detach().float()
    Wsel = W.index_select(1, torch.tensor(cand, dtype=torch.long, device=device))
    Wsc = Wsel * std_t.view(-1, 1)
    pred_p, b0_pred = per_prompt_pred(model, tok, items, layer_mod, sae,
                                       mean_t, std_t, Wsc, cand, device,
                                       args.batch_size)
    print(f"    逐 prompt pred {time.time()-t0:.0f}s", flush=True)

    # ---- 4) 逐 prompt 激活统计 ----
    # 注意：36.capture_sparse 是**按 batch 展平** token 的，无法还原到单个 prompt，
    # 所以这里自己写一份逐 prompt 版本（batch=1 太慢，改用逐 prompt 索引重建）。
    M2_p, M3_p = per_prompt_act(model, tok, items, layer_mod, sae, mean_t, std_t,
                                 cand, device, args.batch_size)
    print(f"    逐 prompt M2/M3 {M2_p.shape}", flush=True)

    # ---- 5) 落盘 ----
    np.savez_compressed(
        npz_p,
        E32=E32.astype(np.float32),
        E16=E16.astype(np.float32),
        pred_p=pred_p.astype(np.float32),
        M2_p=M2_p.astype(np.float32),
        M3_p=M3_p.astype(np.float32),
        cand=np.asarray(cand, dtype=np.int32),
        b0_32=b0_32.astype(np.float32),
        b0_16=b0_16.astype(np.float32),
        b0_pred=b0_pred.astype(np.float32),
    )

    k32 = kappa_of(E32)
    k16 = kappa_of(E16)
    vc = _m53.variance_components(E32)
    rec = {
        "model": m, "layer": layer, "task": task,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_prompts": len(items), "n_features": len(cand),
        "block_idx": block_idx, "task_meta": meta,
        "b0_acc": acc,
        "b0_f32": float(b0_32.mean()), "b0_bf16": float(b0_16.mean()),
        "kappa_f32": k32, "kappa_bf16": k16,
        "icc_f32": vc["icc"],
        "median_abs_E_f32": float(np.median(np.abs(E32))),
        "median_abs_E_bf16": float(np.median(np.abs(E16))),
        "frac_zero_f32": float((np.abs(E32) < 1e-12).mean()),
        "frac_zero_bf16": float((np.abs(E16) < 1e-12).mean()),
        "npz": str(npz_p.relative_to(ROOT)),
    }
    json_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"    κ f32={k32:.3f} bf16={k16:.3f} rel?", flush=True)
    return rec


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


def main():
    ap = argparse.ArgumentParser(description="E20：48 组网格换 held-out 协议")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--spec", nargs="*", default=None, help="model:layer:task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_features", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    grid = build_grid()
    if args.list:
        print(f"# 网格 {len(grid)} 组")
        for m, l, t in grid:
            print(f"{m}:{l}:{t}")
        return

    if args.spec:
        specs = []
        for s in args.spec:
            m, l, t = s.split(":")
            specs.append((m, int(l), t))
    elif args.model:
        specs = [(m, l, t) for m, l, t in grid
                 if m == args.model
                 and (args.layer is None or l == args.layer)
                 and (args.task is None or t == args.task)]
    elif args.all:
        specs = grid
    else:
        specs = grid

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    by_model = {}
    for m, l, t in specs:
        by_model.setdefault(m, []).append((l, t))

    for m, jobs in by_model.items():
        print(f"\n===== {m}（{len(jobs)} 个 spec）=====", flush=True)
        try:
            model, tok = load_model(m)
        except Exception as e:
            print(f"[加载失败] {m}: {e}", flush=True)
            traceback.print_exc()
            continue
        for layer, task in jobs:
            try:
                run_one(model, tok, m, layer, task, args, device)
            except Exception as e:
                traceback.print_exc()
                print(f"[失败] {m} L{layer} {task}: {e}", flush=True)
                continue
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[写出] {OUT}")


if __name__ == "__main__":
    main()
