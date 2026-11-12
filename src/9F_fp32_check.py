# -*- coding: utf-8 -*-
"""E1-fp32 — 把模型权重换成 float32，换 batch 还差多少？

E1 显示同一 dtype 下换 batch（8/16/32），效应矩阵的逐元素相关只有 0.64--0.84；
9E 又排除了「左填充移动位置编码」这个可能（固定 position_ids 反而略差）。
剩下的嫌疑是 bf16 权重本身：残差流 h 的量级在 10--30，bf16 在该量级的 ULP 是
0.0625--0.125，而中位效应只有 0.02--0.17——**效应与一个 ULP 同量级**。
那么换 batch 改变的是「哪些舍入恰好发生」，即改变的是矩阵本身，而不是噪声大小。

本脚本做直接检验：把同一个模型以 float32 载入，重算同样 12 个特征在
batch 8/16/32 下的效应矩阵。若相关跳到 0.99 以上，则 bf16 舍入就是原因，
且「fp32 权重」是可执行的补救；若不跳，则另有来源。
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

from model_utils import load_config, resolve_model_cfg  # noqa: E402

_m27 = import_module("27_h6_semantic_injection")
_m36 = import_module("36_causal_validity")
_m57 = import_module("57_resolution_fix")
_m77 = import_module("77_grid_heldout")

ROOT = Path(__file__).resolve().parent
REPO = ROOT
OUT = ROOT / "output" / "phaseStep2" / "cfse_e1_fp32"


def load_fp32(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = load_config(ROOT / "config.json")
    _, mc = resolve_model_cfg(cfg, name)
    p = str(REPO / mc["local_path"])
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=mc.get("trust_remote_code", True))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        p, torch_dtype=torch.float32, device_map="auto",
        trust_remote_code=mc.get("trust_remote_code", True))
    model.eval()
    return model, tok


def load_bf16(name):
    cfg = load_config(ROOT / "config.json")
    _, mc = resolve_model_cfg(cfg, name)
    model, tok = _m27.load_model_for_generation(REPO, mc, REPO / mc["local_path"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def effect_matrix(model, tok, items, layer_mod, sae, mean_t, std_t, cand,
                  device, batch_size):
    n_f, n_p = len(cand), len(items)
    E = np.zeros((n_f, n_p), dtype=np.float64)
    b0, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, None, device, batch_size)
    b0 = np.asarray(b0, dtype=np.float64)
    for i, f in enumerate(cand):
        hk = _m36.make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        v, _ = _m57.batch_behavior_f32(model, tok, items, layer_mod, hk, device, batch_size)
        E[i] = np.asarray(v, dtype=np.float64) - b0
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-2b")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--n_features", type=int, default=12)
    ap.add_argument("--compare_bf16", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    model, tok = load_fp32(args.model)
    layers_mod = _m27.find_decoder_layers(model)
    lmap = _m36.calibrate_layer_map(
        model, tok, layers_mod, device,
        "When Mary and John went to the store, John gave a drink to")
    layer_mod = layers_mod[lmap[str(args.layer)]["block_idx"]]

    loaded = _m36.load_per_layer_sae(ROOT, args.model, args.layer, device)
    if loaded is None:
        raise SystemExit("缺 SAE")
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    built = _m36.TASK_BUILDERS[args.task](tok, args.n_prompts, seed=42)
    items = built[0] if isinstance(built, tuple) else built
    feats = _m77.loaded_features(args.model, args.layer, args.task)
    cand = [int(x["feature_idx"]) for x in feats][:args.n_features]

    mats = {}
    t0 = time.time()
    for bs in (8, 16, 32):
        mats[f"fp32_b{bs}"] = effect_matrix(model, tok, items, layer_mod, sae, mean_t,
                                           std_t, cand, device, bs)
        print(f"  fp32_b{bs} 中位|dB|={np.median(np.abs(mats[f'fp32_b{bs}'])):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    keys = list(mats)
    res = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A, B = mats[a], mats[b]
            fa, fb = np.abs(A.mean(axis=1)), np.abs(B.mean(axis=1))
            res[f"{a}_vs_{b}"] = {
                "cell_pearson": float(np.corrcoef(A.ravel(), B.ravel())[0, 1]),
                "feat_spearman": float(_m77.spearman(fa, fb)),
                "median_abs_cell_diff": float(np.median(np.abs(A - B))),
                "median_abs_effect": float(np.median(np.abs(A))),
            }
    for k, v in res.items():
        print(f"  {k}: 单元 r={v['cell_pearson']:.4f} 特征秩 rho={v['feat_spearman']:.4f} "
              f"中位单元差={v['median_abs_cell_diff']:.6f}")

    rec = {"model": args.model, "layer": args.layer, "task": args.task,
           "dtype": "float32", "n_prompts": len(items), "n_features": len(cand),
           "pairs": res,
           "median_abs_effect": float(np.median(np.abs(mats["fp32_b16"]))),
           "min_cell_pearson": float(min(v["cell_pearson"] for v in res.values())),
           "min_feat_spearman": float(min(v["feat_spearman"] for v in res.values()))}
    (OUT / f"{args.model}_l{args.layer}_{args.task}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nfp32 权重：单元 r 最小 {rec['min_cell_pearson']:.4f}，"
          f"特征秩 rho 最小 {rec['min_feat_spearman']:.4f}")

    # ---- 与 bf16 权重的直接对照：稿件里的数字会移动多少？ ----
    if args.compare_bf16:
        del model
        torch.cuda.empty_cache()
        print("\n[对照] 载入 bf16 权重，同一 batch=16 重算", flush=True)
        bmodel, btok = load_bf16(args.model)
        layers_b = _m27.find_decoder_layers(bmodel)
        lmap_b = _m36.calibrate_layer_map(
            bmodel, btok, layers_b, device,
            "When Mary and John went to the store, John gave a drink to")
        layer_b = layers_b[lmap_b[str(args.layer)]["block_idx"]]
        E_b = effect_matrix(bmodel, btok, items, layer_b, sae, mean_t, std_t,
                            cand, device, 16)
        E_f = mats["fp32_b16"]
        cmp = {
            "cell_pearson": float(np.corrcoef(E_f.ravel(), E_b.ravel())[0, 1]),
            "feat_spearman": float(_m77.spearman(np.abs(E_f.mean(axis=1)),
                                                 np.abs(E_b.mean(axis=1)))),
            "feat_pearson": float(np.corrcoef(np.abs(E_f.mean(axis=1)),
                                              np.abs(E_b.mean(axis=1)))[0, 1]),
            "median_abs_cell_diff": float(np.median(np.abs(E_f - E_b))),
            "median_abs_effect_fp32": float(np.median(np.abs(E_f))),
            "median_abs_effect_bf16": float(np.median(np.abs(E_b))),
            "frac_bf16_on_ulp_grid": float(np.mean(
                np.abs(E_b / 0.0625 - np.round(E_b / 0.0625)) < 1e-6)),
        }
        rec["fp32_vs_bf16_b16"] = cmp
        rec["pairs"]["fp32_b16_vs_bf16_b16"] = cmp
        (OUT / f"{args.model}_l{args.layer}_{args.task}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        np.savez_compressed(OUT / f"{args.model}_l{args.layer}_{args.task}_mats.npz",
                            E_fp32_b16=E_f.astype(np.float32),
                            E_bf16_b16=E_b.astype(np.float32),
                            cand=np.asarray(cand, dtype=np.int32))
        print(f"  fp32 vs bf16（同 batch16）：单元 r={cmp['cell_pearson']:.4f} "
              f"特征秩 rho={cmp['feat_spearman']:.4f} "
              f"中位单元差={cmp['median_abs_cell_diff']:.6f}")
        print(f"  中位|dB|：fp32 {cmp['median_abs_effect_fp32']:.6f} vs "
              f"bf16 {cmp['median_abs_effect_bf16']:.6f}；"
              f"bf16 落在 0.0625 网格上的比例 {cmp['frac_bf16_on_ulp_grid']:.2%}")



if __name__ == "__main__":
    main()
