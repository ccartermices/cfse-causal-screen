# -*- coding: utf-8 -*-
"""E8 汇总：KernelSHAP 预算收敛 + 标准 IG
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
SRC = BASE / "cfse_e8_shap_ig"
OUT = SRC / "e8_summary.json"


def main():
    files = sorted(SRC.glob("*/layer*/*.json"))
    rows = []
    for f in files:
        if f.name.startswith("e8_"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        rows.append(d)
    if not rows:
        print("无结果")
        return

    print(f"组合数 {len(rows)}\n")
    # ---- C05 ----
    print("== KernelSHAP ==")
    print("组合                      F8 效率gap  K64   K128  K256 | F16 128-512 512-1024 | 种子ρ中位/最小")
    f8_conv = {}          # K -> [rho_vs_exact]；K 由实际生成值决定（63/108/182，非 2 的幂）
    seedmed, seedmin = [], []
    f16_pairs = {"128_vs_512": [], "512_vs_1024": [], "128_vs_1024": []}
    f16_eff = {"128": [], "512": [], "1024": []}
    for d in rows:
        sh = d.get("shap") or {}
        if not sh:
            continue
        conv = sh.get("F8_budget", [])
        cmap = {c["K"]: c["rho_vs_exact"] for c in conv}
        for k, v in cmap.items():
            f8_conv.setdefault(k, []).append(v)
        for k, v in (sh.get("F16_budget") or {}).items():
            if k in f16_pairs:
                f16_pairs[k].append(v)
        for k, v in (sh.get("F16_rho_vs_effect") or {}).items():
            if k in f16_eff:
                f16_eff[k].append(v)
        sd = sh.get("F16_seed_rho") or {}
        seedmed.append(sd.get("median", np.nan))
        seedmin.append(sd.get("min", np.nan))
        print(f"{d['model'][:14]:14s} L{d['layer']:<2d} {d['task']:4s} "
              f"{sh['exact_F8']['efficiency_gap']:.1e}  "
              + " ".join(f"{cmap.get(k, np.nan):.2f}" for k in (63, 108, 182))
              + f" | {sh['F16_budget'].get('128_vs_512', np.nan):.2f} "
              f"{sh['F16_budget'].get('512_vs_1024', np.nan):.2f} | "
              f"{sd.get('median', np.nan):.2f}/{sd.get('min', np.nan):.2f}")

    print("\nF=8 与精确 Shapley 的秩相关（中位）：")
    for k, v in f8_conv.items():
        if v:
            print(f"  K≈{k:3d}: {np.median(v):.3f}  (n={len(v)})")
    print("F=16 预算间秩相关（中位）：")
    for k, v in f16_pairs.items():
        if v:
            print(f"  {k}: {np.median(v):.3f}  min {np.min(v):.3f}  (n={len(v)})")
    print("F=16 |φ| 与 |ΔB| 的秩相关（中位）：")
    for k, v in f16_eff.items():
        if v:
            print(f"  K={k}: {np.median(v):+.3f}  (n={len(v)})")
    print(f"F=16 同预算换种子：中位 {np.median(seedmed):.3f}，最小 {np.min(seedmin):.3f}")

    # ---- C06 ----
    print("\n== 标准 IG ==")
    steps = ["8", "32", "128"]
    tab = {s: {"rho": [], "comp": [], "cin": [], "resid": []} for s in steps}
    for d in rows:
        ig = d.get("ig") or {}
        if not ig:
            continue
        print(f"{d['model'][:14]:14s} L{d['layer']:<2d} {d['task']:4s} " +
              " ".join(f"ρ{s}={ig[s]['rho_vs_effect']:+.3f}" for s in steps if s in ig))
        for s in steps:
            if s in ig:
                tab[s]["rho"].append(ig[s]["rho_vs_effect"])
                tab[s]["comp"].append(ig[s]["completeness_ratio"])
                tab[s]["cin"].append(ig[s]["completeness_vs_input_change"])
                tab[s]["resid"].append(ig[s]["residual_frac"])

    print("\n步数  ρ(|IG|,|ΔB|)  子空间完备性  相对输入变化  投影残差")
    for s in steps:
        if not tab[s]["rho"]:
            continue
        print(f"{s:>4s}  {np.median(tab[s]['rho']):+.3f}       "
              f"{np.median(tab[s]['comp']):.3f}        "
              f"{np.median(tab[s]['cin']):.3f}         "
              f"{np.median(tab[s]['resid']):.3f}")

    summary = {
        "n_combos": len(rows),
        "shap": {"f8_budget": {str(k): float(np.median(v)) for k, v in f8_conv.items() if v},
                 "f16_pairs": {k: float(np.median(v)) for k, v in f16_pairs.items() if v},
                 "f16_rho_vs_effect": {k: float(np.median(v)) for k, v in f16_eff.items() if v},
                 "seed_median": float(np.median(seedmed)),
                 "seed_min": float(np.min(seedmin))},
        "ig": {s: {"rho": float(np.median(tab[s]["rho"])),
                   "completeness": float(np.median(tab[s]["comp"])),
                   "completeness_vs_input": float(np.median(tab[s]["cin"])),
                   "residual": float(np.median(tab[s]["resid"]))}
               for s in steps if tab[s]["rho"]},
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
