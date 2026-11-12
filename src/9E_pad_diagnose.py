# -*- coding: utf-8 -*-
"""E1-diagnose — 左填充是否在移动位置编码？

E1 显示：同一 dtype 下换 batch（8/16/32），效应矩阵的逐元素相关只有 0.64--0.84，
pythia-1.4b 甚至降到 0.15。这不像「浮点抖动」，更像系统性的口径差异。

怀疑：`_encode_with_prefix` 用左填充补齐 batch，但前向时**没有传 position_ids**。
HF 的默认行为是 position_ids = arange(L)，于是被补齐的 prompt 的真实 token
拿到的位置编号整体后移 k 位（k = 该 batch 里最长序列与它的长度差）。
对用 RoPE 的模型，这等于把整句旋转了一个偏移量——不是数值噪声，是换了输入。

本脚本做三件事：
  1. 对同一批 prompt，分别以 batch=16 与 batch=1 前向，比较末位 logits 与 margin；
  2. 在传 position_ids=(cumsum(mask)-1).clamp(min=0) 后重复，看差异是否消失；
  3. 报告两种口径下「换 batch」引起的 margin 变化量，与中位效应对比。
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from model_utils import (load_config, resolve_model_cfg, model_needs_fp8,  # noqa: E402
                         dequantize_fp8_model)

_m27 = import_module("27_h6_semantic_injection")
_m36 = import_module("36_causal_validity")

load_model_for_generation = _m27.load_model_for_generation
ROOT = Path(__file__).resolve().parent
REPO = ROOT
OUT = ROOT / "output" / "phaseStep2" / "cfse_e1_pad"


def load_model(name):
    cfg = load_config(ROOT / "config.json")
    _, mc = resolve_model_cfg(cfg, name)
    model_path = REPO / mc["local_path"]
    if model_needs_fp8(mc, model_path):
        try:
            from huggingface_hub import constants as _hf
            _hf.HF_HUB_OFFLINE = True
        except Exception:
            pass
    model, tok = load_model_for_generation(REPO, mc, model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model_needs_fp8(mc, model_path):
        dequantize_fp8_model(model)
    model.eval()
    return model, tok


def margins(model, tok, items, layer_mod, device, batch_size, pin):
    """返回逐 prompt 的 margin（float32，用 lm_head 重算）。

    pin=True 时传 position_ids=(cumsum(mask)-1).clamp(min=0)，
    使真实 token 的位置编号与不填充时一致。
    """
    mlp = model.get_output_embeddings()
    W = mlp.weight.detach()
    out = np.zeros(len(items))
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        ids, attn = _m36._encode_with_prefix(tok, chunk, device)
        kw = {}
        if pin:
            pos = (attn.cumsum(dim=-1) - 1).clamp(min=0)
            pos = pos.masked_fill(attn == 0, 0)
            kw["position_ids"] = pos
        cap = {}

        def pre(mod, args):
            cap["h"] = args[0]

        h_head = mlp.register_forward_pre_hook(pre)
        try:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=attn, **kw)
        finally:
            h_head.remove()
        h = cap["h"]
        h = (h[0] if isinstance(h, (tuple, list)) else h)[:, -1, :].float()
        cid = torch.tensor([it["correct_id"] for it in chunk], device=h.device)
        did = torch.tensor([it["distractor_id"] for it in chunk], device=h.device)
        lg = h @ W.t().float()
        out[s:s + len(chunk)] = (lg.gather(1, cid[:, None])
                                 - lg.gather(1, did[:, None])).squeeze(1).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3.5-2b,pythia-1.4b,tinyllama-1.1b")
    ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--task", default="ioi")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    recs = []
    for name in args.models.split(","):
        name = name.strip()
        print(f"[E1-pad] {name}", flush=True)
        model, tok = load_model(name)
        layers_mod = _m27.find_decoder_layers(model)
        built = _m36.TASK_BUILDERS[args.task](tok, args.n_prompts, seed=42)
        items = built[0] if isinstance(built, tuple) else built
        layer_mod = layers_mod[0]          # 该诊断不依赖层，占位即可

        t0 = time.time()
        m16 = margins(model, tok, items, layer_mod, device, 16, pin=False)
        m1 = margins(model, tok, items, layer_mod, device, 1, pin=False)
        m16p = margins(model, tok, items, layer_mod, device, 16, pin=True)
        m1p = margins(model, tok, items, layer_mod, device, 1, pin=True)

        def cmp(a, b):
            d = a - b
            return {"median_abs_diff": float(np.median(np.abs(d))),
                    "max_abs_diff": float(np.abs(d).max()),
                    "corr": float(np.corrcoef(a, b)[0, 1]),
                    "n_exact": int((np.abs(d) < 1e-6).sum()), "n": len(d)}

        rec = {
            "model": name, "task": args.task, "n_prompts": len(items),
            "n_padded_in_b16": int(sum(1 for it in items
                                       if len(tok(it["prompt"])["input_ids"])
                                       < max(len(tok(x["prompt"])["input_ids"])
                                             for x in items))),
            "unpinned_b16_vs_b1": cmp(m16, m1),
            "pinned_b16_vs_b1": cmp(m16p, m1p),
            "pinned_vs_unpinned_b16": cmp(m16p, m16),
            "median_abs_margin": float(np.median(np.abs(m16))),
            "elapsed_s": time.time() - t0,
        }
        recs.append(rec)
        print(f"  未固定位置 b16 vs b1: 中位差 {rec['unpinned_b16_vs_b1']['median_abs_diff']:.4f} "
              f"r={rec['unpinned_b16_vs_b1']['corr']:.4f}", flush=True)
        print(f"  固定位置   b16 vs b1: 中位差 {rec['pinned_b16_vs_b1']['median_abs_diff']:.4f} "
              f"r={rec['pinned_b16_vs_b1']['corr']:.4f}", flush=True)
        print(f"  中位 |margin| = {rec['median_abs_margin']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    (OUT / "pad_diagnosis.json").write_text(
        json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
