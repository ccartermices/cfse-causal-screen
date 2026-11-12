# -*- coding: utf-8 -*-
"""E9 — 并集特征池：指标比较是否被 M2 的选择系统性削权

回应审稿意见 C02（R1-M2）。

问题
----
现稿的特征池是「按 M2（任务 token 上的平均激活）取前 200」。所有其他指标都在这个
M2 预先选好的池里被评估，因此 M2 在池内的秩相关是它的**上界**，而其他指标的高效应
尾部可能被截断。审稿人要求：并集池重算表 1 与 η²。

设计
----
对每个组合：
  1. 候选宇宙 U = 该层 SAE 中按 M2 排序的前 1000 个特征（不改变"非退化特征"的
     前提，但比 200 宽 5 倍）；
  2. 在 U 上一次性算出 M2、pred_full、IG(8 步)、attn 四种打分（都是向量化算子，
     成本与特征数几乎无关）；
  3. 并集池 P = (M2 前 60) ∪ (pred_full 前 60) ∪ (IG 前 60) ∪ (attn 前 60)
              ∪ (U 中排名 200--1000 里随机 60 个)；
  4. 只对 P 中**尚未测过**的特征补测消融效应（每个特征一次批量前向）；
  5. 在 P 上重算各指标与 |ΔB| 的 Spearman，并与原 200 池的读数配对比较。

同时给出 M2 分层读数（把 200 池按 M2 四分位切开），用来直接看"选择效应"：
如果 M2 的优势只出现在它自己选出的第一层，选择偏差就成立。

用法
----
  python 9I_c02_union_pool.py --list
  python 9I_c02_union_pool.py --all
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
OUT = BASE / "cfse_e9_unionpool"
SRC = BASE / "causal_validity"

# 8 个组合, 均同时具备 36 单特征干预与 47 归因两份结果
DEFAULT_COMBOS = [
    ("qwen3.5-2b", 16, "ioi"),
    ("qwen3.5-2b", 16, "gt"),
    ("tinyllama-1.1b", 14, "ioi"),
    ("tinyllama-1.1b", 14, "gt"),
    ("minicpm5-1b", 16, "ioi"),
    ("ministral-3-3b", 17, "ioi"),
    ("pythia-1.4b", 16, "ioi"),
    ("qwen2.5-7b-instruct", 18, "ioi"),
]

UNIVERSE = 700
TOP_K = 50
RAND_K = 50
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
            _HF_OFF = True
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


def activation_stats(model, tok, items, layer_mod, sae, mean_t, std_t, device,
                     batch_size=BATCH):
    """返回 (mean_act, max_act, act_freq)，均为长度 F_all 的数组。"""
    F_all = sae.decoder.weight.shape[1]
    s_sum = np.zeros(F_all, dtype=np.float64)
    s_max = np.zeros(F_all, dtype=np.float64)
    s_cnt = np.zeros(F_all, dtype=np.float64)
    n_tok = 0
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tok, chunk, device)
        cap = {}
        handle = layer_mod.register_forward_hook(_m47._leaf_hook(cap, "h"))
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        finally:
            handle.remove()
        h = cap["h"].detach().float()
        with torch.no_grad():
            sp = sae.encode((h - mean_t) / std_t)[0]              # (B,T,F)
        m = attn.unsqueeze(-1).float()
        s_sum += (sp * m).sum(dim=(0, 1)).cpu().double().numpy()
        s_max = np.maximum(s_max, sp.max(dim=0).values.max(dim=0).values.cpu()
                           .double().numpy())
        s_cnt += ((sp > 0).float() * m).sum(dim=(0, 1)).cpu().double().numpy()
        n_tok += int(attn.sum().item())
        del h, sp, cap
    mean_act = s_sum / max(n_tok, 1)
    freq = s_cnt / max(n_tok, 1)
    return mean_act, s_max, freq


def ablate_effects(model, tok, items, layer_mod, sae, mean_t, std_t, feats, device,
                   batch_size=BATCH):
    out = {}
    for f in feats:
        hk = _m36.make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        _, bm = _m36.batch_behavior(model, tok, items, layer_mod, hk, device, batch_size)
        out[int(f)] = float(bm)
    return out


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
        print("  [跳过] at chance")
        return None
    b0 = rec36.get("b0")
    block_idx = rec36.get("block_idx")
    _, rows = _m47.load_targets(str(src))
    know = {r["feature_idx"]: r["B_ablate"] for r in rows if r.get("B_ablate") is not None}
    if len(know) < 40:
        print("  [跳过] 已知效应不足")
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(model_name)
    layer_mod = find_decoder_layers(model)[block_idx]
    loaded = load_per_layer_sae(ROOT, model_name, layer, device)
    if loaded is None:
        print(f"  [跳过] 缺 SAE {model_name} L{layer}")
        return None
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)
    Wsc = sae.decoder.weight.detach().float() * std_t.view(-1, 1)

    items, _ = _m36.TASK_BUILDERS[task](tok, args.n_prompts, seed=42)
    print(f"[{model_name} L{layer} {task}] prompts={len(items)}", flush=True)

    t0 = time.time()
    mean_act, max_act, freq = activation_stats(model, tok, items, layer_mod, sae,
                                               mean_t, std_t, device)
    order_all = np.argsort(-mean_act)
    univ = order_all[:args.universe].tolist()
    print(f"  [激活] 宇宙 {len(univ)} 个特征 ({time.time()-t0:.0f}s)", flush=True)

    Wsel = Wsc.index_select(1, torch.tensor(univ, dtype=torch.long, device=device))
    t0 = time.time()
    fo = _m47.first_order_probe(model, tok, items, layer_mod, sae, mean_t, std_t,
                                Wsel, univ, device, args.batch_size)
    print(f"  [一阶] 完成 ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    at = _m47.attention_probe(model, tok, items, layer_mod, sae, mean_t, std_t,
                              Wsel, univ, device, args.batch_size)
    print(f"  [注意力] available={at['available']} ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    ig = _m47.ig_probe(model, tok, items, layer_mod, sae, mean_t, std_t, Wsel,
                       univ, device, args.batch_size, args.ig_steps)
    print(f"  [IG] 完成 ({time.time()-t0:.0f}s)", flush=True)

    # pred_full 是带符号的预测 ΔB（消融时整体为负），与 |ΔB| 比较必须取绝对值；
    # 47 落盘的 "pred_full" 也是 abs。ig/attn 返回的已经是绝对值。
    score = {"M2": np.array([mean_act[i] for i in univ]),
             "pred_full": np.abs(fo["pred_full"]),
             "ig": ig["ig"]}
    if at["available"]:
        score["attn"] = at["attn"]

    # ---- 并集池 ----
    rng = np.random.RandomState(args.seed)
    pool = set()
    for k, v in score.items():
        for j in np.argsort(-np.abs(v))[:args.top_k]:
            pool.add(univ[int(j)])
    tail = [u for u in univ[200:] if u not in pool]
    pool |= set(rng.choice(tail, size=min(args.rand_k, len(tail)), replace=False).tolist())
    new = [f for f in pool if f not in know]
    print(f"  [并集池] |P|={len(pool)}, 其中已测 {len(pool)-len(new)}, 需补测 {len(new)}",
          flush=True)

    t0 = time.time()
    add = ablate_effects(model, tok, items, layer_mod, sae, mean_t, std_t, new, device,
                         args.batch_size)
    print(f"  [补测] {len(new)} 个特征 ({time.time()-t0:.0f}s)", flush=True)

    B = dict(know)
    B.update(add)
    feats = sorted(pool)
    eff = np.array([abs(B[f] - b0) for f in feats], dtype=np.float64)
    sc = {k: np.array([v[univ.index(f)] for f in feats], dtype=np.float64)
          for k, v in score.items()}
    m2 = np.array([mean_act[f] for f in feats], dtype=np.float64)
    m1 = np.array([freq[f] for f in feats], dtype=np.float64)
    m3 = np.array([max_act[f] for f in feats], dtype=np.float64)

    res = {
        "model": model_name, "layer": layer, "task": task,
        "b0": b0, "n_prompts": len(items), "universe": args.universe,
        "pool_size": len(feats), "n_new": len(new),
        "feature_idx": [int(f) for f in feats],
        "M1_act_freq_task": m1.tolist(),
        "M2_mean_act_task": m2.tolist(),
        "M3_max_act_task": m3.tolist(),
        "B_ablate": [float(B[f]) for f in feats],
        "abs_effect": eff.tolist(),
        "pred_full": sc["pred_full"].tolist(),
        "ig": sc["ig"].tolist(),
        "attn": (sc.get("attn").tolist() if "attn" in sc else None),
        "rho": {k: float(spearman(v, eff)) for k, v in sc.items()},
    }
    res["rho"]["M1"] = float(spearman(m1, eff))
    res["rho"]["M3"] = float(spearman(m3, eff))

    # M2 四分位分层
    q = np.argsort(-m2)
    quart = []
    for i in range(4):
        idxq = q[i * len(q) // 4:(i + 1) * len(q) // 4]
        quart.append({k: float(spearman(v[idxq], eff[idxq])) for k, v in sc.items()})
    res["rho_by_M2_quartile"] = quart

    print("  [读数] " + " ".join(f"{k}={v:+.3f}" for k, v in res["rho"].items()), flush=True)
    _save(out_file, res)
    del model, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--universe", type=int, default=UNIVERSE)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--rand_k", type=int, default=RAND_K)
    ap.add_argument("--ig_steps", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=BATCH)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    combos = DEFAULT_COMBOS
    if args.combos:
        combos = []
        for t in args.combos.split(","):
            m, l, tk = t.split(":")
            combos.append((m, int(l), tk))
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
