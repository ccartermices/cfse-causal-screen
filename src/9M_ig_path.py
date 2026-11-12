# -*- coding: utf-8 -*-
"""E8b — IG 的路径基线对照：均值基线 vs 现稿的重建位移基线

审稿意见 C06 说现稿 IG 的路径起点是"全特征位移"（h - R）而不是标准基线。
把基线换成均值 b 之后，同一批特征的秩相关明显下降。为了确认这个下降来自
**基线**而不是实现，本脚本用同一套积分代码跑两条路径：

  mean  : h(α) = b + α·R        （标准，起点为均值激活）
  recon : h(α) = h - (1-α)·R    （现稿，起点为 h-R）

两条路径都用 c_f 在原点的稀疏激活，都做 m=32 步，都与同一批 |ΔB| 比较。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import importlib

_m36 = importlib.import_module("36_causal_validity")
_m47 = importlib.import_module("47_attribution_baselines")
_m77 = importlib.import_module("77_grid_heldout")
spearman = _m77.spearman

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module as _im  # noqa: E402

_m9h = _im("9H_c05_c06")

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
SRC = BASE / "cfse_e8_shap_ig"
COMBOS = [("qwen3.5-2b", 16, "ioi"), ("pythia-1.4b", 16, "ioi")]
STEPS = 32


def ig_path(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc, feat_idx,
            device, steps, path, batch_size=32):
    idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
    Wsc = Wsc.index_select(1, idx_t)
    F = len(feat_idx)
    term = np.zeros(F, dtype=np.float64)
    n = 0
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _m36._encode_with_prefix(tok, chunk, device)
        cap = {}
        hd = layer_mod.register_forward_hook(_m47._leaf_hook(cap, "h"))
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        finally:
            hd.remove()
        h = cap["h"].detach().float()
        with torch.no_grad():
            sparse = sae.encode((h - mean_t) / std_t)[0].index_select(-1, idx_t)
            R = sparse @ Wsc.t()
        gsum = torch.zeros_like(h)
        for k in range(1, steps + 1):
            a = k / float(steps)
            cap2 = {}

            def hk(module, args, output, a=a):
                hs = output[0] if isinstance(output, (tuple, list)) else output
                dt = hs.dtype
                if path == "mean":
                    cur = mean_t.view(1, 1, -1) + a * R
                else:
                    cur = h - (1.0 - a) * R
                leaf = cur.requires_grad_(True)
                cap2["leaf"] = leaf
                return ((leaf.to(dt),) + tuple(output[1:])
                        if isinstance(output, (tuple, list)) else leaf.to(dt))

            hd = layer_mod.register_forward_hook(hk)
            try:
                out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            finally:
                hd.remove()
            B = _m47._behavior(out.logits[:, -1, :].float(), chunk)
            g = torch.autograd.grad(B.sum(), cap2["leaf"])[0].float()
            gsum += g * attn.unsqueeze(-1).float()
            del out, B, g, cap2
        gbar = gsum / steps
        term += torch.einsum("btd,df,btf->bf", gbar, Wsc, sparse).sum(0) \
            .cpu().double().numpy()
        n += len(chunk)
        del h, gsum, gbar, sparse, R
    return term / max(n, 1)


def main():
    out = {}
    for name, layer, task in COMBOS:
        f = SRC / name / f"layer{layer}" / f"{task}.json"
        if not f.exists():
            print(f"[跳过] {f}")
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        src36 = BASE / "causal_validity" / name / f"layer{layer}" / f"{task}.json"
        rec36 = json.loads(src36.read_text(encoding="utf-8"))
        b0 = rec36.get("b0") or 0.0
        _, rows = _m47.load_targets(str(src36))
        feat_idx = [r["feature_idx"] for r in rows]
        eff = np.array([abs(r["B_ablate"] - b0) for r in rows])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, tok = _m9h.load_model(name)
        layer_mod = _m9h.find_decoder_layers(model)[rec36["block_idx"]]
        sae, mean_np, std_np = _m36.load_per_layer_sae(ROOT, name, layer, device)
        mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
        std_t = torch.tensor(std_np, dtype=torch.float32, device=device)
        Wsc = sae.decoder.weight.detach().float() * std_t.view(-1, 1)
        items, _ = _m36.TASK_BUILDERS[task](tok, 200, seed=42)

        r = {}
        for path in ("mean", "recon"):
            ig = ig_path(model, tok, items, layer_mod, sae, mean_t, std_t, Wsc,
                         feat_idx, device, STEPS, path)
            r[path] = float(spearman(np.abs(ig), eff))
            print(f"[{name} L{layer} {task}] path={path:5s} "
                  f"rho(|IG|,|dB|)={r[path]:+.3f}", flush=True)
        rec.setdefault("ig_path", {})[str(STEPS)] = r
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        out[f"{name}|L{layer}|{task}"] = r
        del model, sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (SRC / "e8b_ig_path.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if out:
        km = np.median([v["mean"] for v in out.values()])
        kr = np.median([v["recon"] for v in out.values()])
        print(f"\n中位：mean 基线 {km:+.3f}，recon 基线 {kr:+.3f}")


if __name__ == "__main__":
    main()
