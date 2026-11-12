# -*- coding: utf-8 -*-
"""E5 — 用 CE / KL 而不是 margin 重测效应：指标选择的敏感性

回应审稿意见 C01（R1-M1 / R3-M2 共同提出）。

问题
----
全文的「效应」dB 定义为正确 token 与干扰 token 的 logit 差（margin）。
但 margin 是一个**二点**读数：它只看两个候选 token，对分布其他部分不敏感。
审稿人因此质疑：如果改用交叉熵或全分布 KL，M1--M6 与效应的关系是否还成立？

设计
----
对同一批组合、同一批特征、同一批 prompt，在三种读数下各算一次效应：

  (1) margin   dB = logit[c] - logit[d]            （现稿口径）
  (2) CE       dCE = CE_ablated - CE_base，CE = -log p(c)  （正确 token 的负对数似然）
  (3) KL       dKL = KL(p_base || p_ablated)（nats，全词表，取末位 token）

三者都在 float32 下用 lm_head 权重重算 logits（与 57 的 f32 口径一致），
因此差别只来自「读数」，不来自数值精度。

报告
----
  - 逐特征聚合后，三种效应两两的 Spearman 相关（同一组合内）
  - 一阶预测器 pred 对三种效应的 rho（若 rho_KL 与 rho_margin 接近，
    说明结论不依赖 margin 这一读数）
  - 跨组合的中位数

用法
----
  python 9A_ce_kl.py --list
  python 9A_ce_kl.py --all
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
_m77 = import_module("77_grid_heldout")

find_decoder_layers = _m27.find_decoder_layers
load_model_for_generation = _m27.load_model_for_generation
spearman = _m77.spearman

ROOT = Path(__file__).resolve().parent
REPO = ROOT
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_e5_cekl"

# 覆盖 3 个家族 x 2 个任务
DEFAULT_COMBOS = [
    ("qwen3.5-2b", 16, "ioi"),
    ("qwen3.5-2b", 16, "gt"),
    ("tinyllama-1.1b", 21, "ioi"),
    ("pythia-1.4b", 16, "ioi"),
]

N_FEATURES = 120
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


def _readout(model, tok, items, layer_mod, hook_fn, device, batch_size=BATCH):
    """一次前向，返回三种读数的逐 prompt 向量：

      margin (B,), ce (B,), logp_c (B,)
    其中 CE 用正确 token 的负对数似然；logits 用 lm_head 权重在 float32 下重算。
    """
    mlp = model.get_output_embeddings()
    W = mlp.weight.detach()
    marg, ces, lps = [], [], []
    n = len(items)
    for s in range(0, n, batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _m36._encode_with_prefix(tok, chunk, device)
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
        logits = h @ W.t().float()                       # (B, V) float32
        logp = torch.log_softmax(logits, dim=-1)
        lp_c = logp.gather(1, cid[:, None]).squeeze(1)
        lp_d = logp.gather(1, did[:, None]).squeeze(1)
        marg.extend((lp_c - lp_d).cpu().tolist())        # 与 margin 同序（softmax 单调）
        ces.extend((-lp_c).cpu().tolist())
        lps.extend(lp_c.cpu().tolist())
    return (np.asarray(marg, dtype=np.float64),
            np.asarray(ces, dtype=np.float64),
            np.asarray(lps, dtype=np.float64))


def _kl_readout(model, tok, items, layer_mod, hook_fn, base_logp, device,
                batch_size=BATCH):
    """返回逐 prompt 的 KL(p_base || p_ablated)（nats）。

    base_logp 为基线 log_softmax，逐 batch 切片对齐，避免重复前向。
    """
    mlp = model.get_output_embeddings()
    W = mlp.weight.detach()
    out = np.zeros(len(items), dtype=np.float64)
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _m36._encode_with_prefix(tok, chunk, device)
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
        logp = torch.log_softmax(h @ W.t().float(), dim=-1)
        lp0 = base_logp[s:s + len(chunk)].to(logp.device)
        kl = (lp0.exp() * (lp0 - logp)).sum(dim=-1)
        out[s:s + len(chunk)] = kl.cpu().double().numpy()
    return out


def base_logp_full(model, tok, items, layer_mod, device, batch_size=BATCH):
    """基线全词表 log_softmax（float32），供 KL 复用。"""
    mlp = model.get_output_embeddings()
    W = mlp.weight.detach()
    chunks = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _m36._encode_with_prefix(tok, chunk, device)
        cap = {}

        def pre(mod, args):
            cap["h"] = args[0]

        h_head = mlp.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            h_head.remove()
        h = cap["h"]
        h = (h[0] if isinstance(h, (tuple, list)) else h)[:, -1, :].float()
        chunks.append(torch.log_softmax(h @ W.t().float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def run_combo(m, layer, task, args, device, model=None, tok=None):
    out_dir = OUT / m / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_p = out_dir / f"{task}.json"
    if json_p.exists() and not args.force:
        print(f"  [跳过] {m} L{layer} {task}", flush=True)
        return json.load(open(json_p, encoding="utf-8"))

    if model is None or tok is None:
        model, tok = load_model(m)
    layers_mod = find_decoder_layers(model)
    lmap = _m36.calibrate_layer_map(
        model, tok, layers_mod, device,
        "When Mary and John went to the store, John gave a drink to")
    layer_mod = layers_mod[lmap[str(layer)]["block_idx"]]

    loaded = _m36.load_per_layer_sae(ROOT, m, layer, device)
    if loaded is None:
        print(f"  [缺 SAE] {m} L{layer}", flush=True)
        return None
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    built = _m36.TASK_BUILDERS[task](tok, args.n_prompts, seed=42)
    items = built[0] if isinstance(built, tuple) else built
    feats = _m77.loaded_features(m, layer, task)
    if not items or not feats:
        print(f"  [缺任务或特征] {m} {task}", flush=True)
        return None
    cand = [int(x["feature_idx"]) for x in feats][:N_FEATURES]

    print(f"  [{m} L{layer} {task}] prompts={len(items)} features={len(cand)}",
          flush=True)

    t0 = time.time()
    m0, ce0, lp_c0 = _readout(model, tok, items, layer_mod, None, device)
    lp0 = base_logp_full(model, tok, items, layer_mod, device)
    print(f"    基线完成 ({time.time()-t0:.0f}s)", flush=True)

    # 一阶预测器
    W = sae.decoder.weight.detach().float()
    Wsel = W.index_select(1, torch.tensor(cand, dtype=torch.long, device=device))
    Wsc = Wsel * std_t.view(-1, 1)
    pred_p, _ = _m77.per_prompt_pred(model, tok, items, layer_mod, sae,
                                     mean_t, std_t, Wsc, cand, device, BATCH)
    pred = np.abs(pred_p.mean(axis=0))

    dm = np.zeros((len(cand), len(items)))
    dc = np.zeros((len(cand), len(items)))
    dk = np.zeros((len(cand), len(items)))
    for i, f in enumerate(cand):
        hk = _m36.make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        mi, ci, _ = _readout(model, tok, items, layer_mod, hk, device)
        dm[i] = mi - m0
        dc[i] = ci - ce0
        dk[i] = _kl_readout(model, tok, items, layer_mod, hk, lp0, device)
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(cand)} ({time.time()-t0:.0f}s)", flush=True)

    # 逐特征聚合
    a_m = np.abs(dm).mean(axis=1)
    a_c = np.abs(dc).mean(axis=1)
    a_k = dk.mean(axis=1)

    def sp(a, b):
        r = spearman(a, b)
        return float(r) if np.isfinite(r) else float("nan")

    rec = {
        "schema": "cfse_e5_cekl/v1",
        "model": m, "layer": layer, "task": task,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_prompts": len(items), "n_features": len(cand),
        "median": {
            "abs_dmargin": float(np.median(np.abs(dm))),
            "abs_dce": float(np.median(np.abs(dc))),
            "dkl": float(np.median(dk)),
            "p90_dkl": float(np.percentile(dk, 90)),
        },
        "effect_corr": {
            "abs_dmargin_vs_abs_dce": sp(a_m, a_c),
            "abs_dmargin_vs_dkl": sp(a_m, a_k),
            "abs_dce_vs_dkl": sp(a_c, a_k),
        },
        "pred_rho": {
            "margin": sp(pred, a_m),
            "ce": sp(pred, a_c),
            "kl": sp(pred, a_k),
        },
        "rel": {
            "margin": _rel(dm), "ce": _rel(dc), "kl": _rel(dk),
        },
        "b0_margin_mean": float(m0.mean()),
        "b0_ce_mean": float(ce0.mean()),
        "elapsed_s": time.time() - t0,
    }
    json_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    e = rec["effect_corr"]
    p = rec["pred_rho"]
    print(f"    相关 margin~CE {e['abs_dmargin_vs_abs_dce']:.3f}  "
          f"margin~KL {e['abs_dmargin_vs_dkl']:.3f} | "
          f"rho_pred: margin {p['margin']:.3f} CE {p['ce']:.3f} KL {p['kl']:.3f}",
          flush=True)
    return rec


def _rel(E, n_splits=10, seed0=4242):
    """折叠口径 split-half 可靠性，与 §7.2 一致。"""
    n_feat, n_prompt = E.shape
    h = n_prompt // 2
    vals = []
    for s in range(n_splits):
        rng = np.random.RandomState(seed0 + 1000 * s)
        perm = rng.permutation(n_prompt)
        sel, ev = perm[:h], perm[h:]
        r = spearman(np.abs(E[:, sel].mean(axis=1)), np.abs(E[:, ev].mean(axis=1)))
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="E5：CE/KL 读数下的效应")
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

    combos = [(m, int(l), t) for m, l, t in
              ([args.combo.split(":")] if args.combo else DEFAULT_COMBOS)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for m, l, t in combos:
        print(f"[E5] === {m} L{l} {t} ===", flush=True)
        model = None
        try:
            model, tok = load_model(m)
            r = run_combo(m, l, t, args, device, model, tok)
            if r:
                results.append(r)
        except Exception:
            traceback.print_exc()
        finally:
            model = None
            torch.cuda.empty_cache()

    if results:
        print("\n" + "=" * 96)
        print(f"{'combo':34s} {'m~CE':>6s} {'m~KL':>6s} {'rho_m':>6s} "
              f"{'rho_CE':>7s} {'rho_KL':>7s} {'relKL':>6s}")
        print("-" * 96)
        for r in results:
            e, p = r["effect_corr"], r["pred_rho"]
            print(f"{r['model']}|{r['layer']}|{r['task']:18s} "
                  f"{e['abs_dmargin_vs_abs_dce']:6.3f} {e['abs_dmargin_vs_dkl']:6.3f} "
                  f"{p['margin']:6.3f} {p['ce']:7.3f} {p['kl']:7.3f} "
                  f"{r['rel']['kl']:6.3f}")
        print("=" * 96)
        med = {k: float(np.nanmedian([r["pred_rho"][k] for r in results]))
               for k in ("margin", "ce", "kl")}
        print(f"中位 rho_pred: margin {med['margin']:.3f} CE {med['ce']:.3f} "
              f"KL {med['kl']:.3f}")
        (OUT / "e5_summary.json").write_text(json.dumps(
            {"results": results, "median_rho_pred": med,
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
