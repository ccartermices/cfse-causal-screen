# -*- coding: utf-8 -*-
"""E9 汇总：并集池上的指标比较与 M2 分层

与现稿（M2 前 200 池）配对比较，并给出：
  - 每个指标在并集池上的 ρ（跨 8 组合的均值±标准差）
  - 指标因子的 η²（单因素方差解释比例）
  - 按 M2 四分位分层的 ρ，用来判定"选择效应"是否存在
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
SRC = BASE / "cfse_e9_unionpool"
OUT = SRC / "e9_summary.json"

ORDER = ["pred_full", "ig", "attn", "M2", "M1", "M3"]


def eta_sq(groups):
    """单因素 η²：组间平方和 / 总平方和。"""
    allv = np.concatenate(groups)
    grand = allv.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups if len(g))
    ss_t = ((allv - grand) ** 2).sum()
    return float(ss_b / ss_t) if ss_t > 0 else float("nan")


def main():
    rows = []
    for f in sorted(SRC.glob("*/layer*/*.json")):
        if f.name.startswith("e9_"):
            continue
        rows.append(json.load(open(f, encoding="utf-8")))
    if not rows:
        print("无结果")
        return

    print(f"组合数 {len(rows)}\n")
    print("组合                      池大小 补测   pred     ig     attn     M2      M1      M3")
    for d in rows:
        r = d["rho"]
        print(f"{d['model'][:14]:14s} L{d['layer']:<2d} {d['task']:4s} "
              f"{d['pool_size']:5d} {d['n_new']:4d}  " +
              "  ".join(f"{r.get(k, float('nan')):+.3f}" for k in ORDER))

    print("\n指标       并集池 ρ 均值±std     最小     最大")
    summary = {}
    for k in ORDER:
        v = np.array([d["rho"][k] for d in rows if k in d["rho"]], dtype=np.float64)
        if len(v) == 0:
            continue
        summary[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                      "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}
        print(f"{k:10s} {v.mean():+.3f}±{v.std(ddof=1):.3f}   {v.min():+.3f}  {v.max():+.3f}")

    groups = [np.array([d["rho"][k] for d in rows if k in d["rho"]]) for k in summary]
    e2 = eta_sq(groups)
    print(f"\n指标因子 η²（并集池 ρ）= {e2:.3f}")

    # M2 四分位分层
    print("\nM2 四分位（1=激活最高）：各指标 ρ 的跨组合中位")
    qtab = []
    for qi in range(4):
        vals = {k: [] for k in ORDER}
        for d in rows:
            qs = d.get("rho_by_M2_quartile") or []
            if len(qs) > qi:
                for k in ORDER:
                    if k in qs[qi]:
                        vals[k].append(qs[qi][k])
        row = {k: float(np.median(v)) if v else float("nan") for k, v in vals.items()}
        qtab.append(row)
        print(f"  Q{qi+1}: " + "  ".join(f"{k}={row[k]:+.3f}" for k in ORDER if k in row))

    res = {"n_combos": len(rows), "metrics": summary, "eta2_metric": e2,
           "quartile": qtab,
           "pool_sizes": [d["pool_size"] for d in rows],
           "n_new": [d["n_new"] for d in rows]}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
