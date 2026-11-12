# -*- coding: utf-8 -*-
"""Step C: 零成本方差分解 —— 哪个轴真正决定「选择器是否有用 / 因果效应有多大」

动因
----
§17–§23 已经把所有组合的实测因果效应落盘 (36 的单特征 C_ablate, 38/40 的 top-N 联合
C_norm/z/flip)。悬而未决的问题: 这些量的变异主要由 **模型 / 层 / 任务 / 指标 / 规模 N**
中的哪一轴决定? 论文里 §5 已有「决定因素是层」的定性说法 (C6), 需要定量化。

本脚本**不加载任何模型、不做任何干预**, 只对已落盘结果做方差分解:
  Part 1 (选择器层面, 源=40 part1_selector): y = log1p(C_norm),
      因子 = {metric, model, layer_slot, task, N} (全交叉)。
  Part 2 (单特征效应层面, 源=36 features): y = log1p(C_ablate),
      因子 = {model, layer_slot, task}。
分解方式: OLS + drop-one (Type II) 边际平方和, eta^2 = SS_f / SS_total。

输出: output/phaseStep2/variance_decomp/variance_decomp.json + .md
      回填 output/phaseF/phaseF_report.md §24

资源: 纯 CPU/IO, 无模型加载; 仍遵守环境节流 (与 36–42 一致)。
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import numpy as np
from pathlib import Path
from datetime import datetime


BASE = Path(__file__).parent
OUT = BASE / "output" / "phaseStep2" / "variance_decomp"
REPORT = BASE / "output" / "phaseF" / "phaseF_report.md"


def slot_of(model, layer, config):
    """把绝对层号映射为 early/mid/late 槽位 (跨模型可比)。"""
    cfg = config["models"].get(model) or {}
    tl = list(cfg.get("target_layers") or [])
    return ["early", "mid", "late"][tl.index(layer)] if layer in tl else f"L{layer}"


def collect_40(config):
    """选择器层面的长表: (model, layer_slot, task, metric, N, cnorm)。"""
    root = BASE / "output" / "phaseStep2" / "selector_margin"
    rows = []
    for p in sorted(root.glob("*/layer*/*.json")):
        if p.name.startswith(("resume_", "summary")):
            continue
        d = json.load(open(p, encoding="utf-8"))
        if not d.get("part1_selector"):
            continue
        slot = slot_of(d["model"], d["layer"], config)
        for x in d["part1_selector"]:
            cn = x["ablate"].get("C_norm")
            if cn is None:
                continue
            rows.append({"model": d["model"], "layer_slot": slot,
                         "task": d["task"], "metric": x["metric"],
                         "N": int(x["N"]), "cnorm": float(cn)})
    return rows


def collect_36(config):
    """单特征层面的长表: (model, layer_slot, task, C_ablate)。"""
    root = BASE / "output" / "phaseStep2" / "causal_validity"
    rows = []
    for p in sorted(root.glob("*/layer*/*.json")):
        if p.name.startswith(("resume_", "summary")):
            continue
        d = json.load(open(p, encoding="utf-8"))
        if d.get("error") or d.get("at_chance"):
            continue
        slot = slot_of(d["model"], d["layer"], config)
        for f in d.get("features") or []:
            if f.get("C_ablate") is None:
                continue
            rows.append({"model": d["model"], "layer_slot": slot,
                         "task": d["task"], "c_ablate": float(f["C_ablate"])})
    return rows


def eta_sq(rows, y_key, factors):
    """OLS + drop-one (Type II) 方差分解, 返回 {factor: eta2} 与 residual eta2。"""
    ys = np.array([r[y_key] for r in rows], float)
    n = len(ys)
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    levels = {f: sorted({r[f] for r in rows}) for f in factors}

    def design(cols):
        mats = [np.ones(n)]
        for f in cols:
            for lv in levels[f][1:]:
                mats.append(np.array([1.0 if r[f] == lv else 0.0 for r in rows]))
        return np.column_stack(mats)

    def ss_res(cols):
        X = design(cols)
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        return float(((ys - X @ beta) ** 2).sum())

    full = ss_res(factors)
    out = {}
    for f in factors:
        rest = [g for g in factors if g != f]
        out[f] = (ss_res(rest) - full) / ss_tot if ss_tot > 0 else 0.0
    return out, (full / ss_tot if ss_tot > 0 else 0.0), levels, int(ss_tot > 0)


def _table(title, eta, resid, levels):
    L = [f"### {title}", "",
         "| 因子 | 水平数 | eta^2 (方差占比) |", "|---|---|---|"]
    for f, v in eta.items():
        L.append(f"| {f} | {len(levels[f])} | {v:.3f} |")
    L.append(f"| (残差) | - | {resid:.3f} |")
    L.append("")
    return L


def main():
    config = json.load(open(BASE / "config.json", encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()

    r40 = collect_40(config)
    r36 = collect_36(config)
    for r in r40:
        r["y"] = float(np.log1p(r["cnorm"]))
    for r in r36:
        r["y"] = float(np.log1p(r["c_ablate"]))

    L = ["# Step C: 零成本方差分解", "",
         f"- 生成时间: {ts}", "- 数据: 已落盘结果, 无模型加载 / 无干预",
         f"- Part 1 行数 {len(r40)} (源 40: 组合 × 指标 × N)",
         f"- Part 2 行数 {len(r36)} (源 36: 组合 × 特征)", "",
         "响应变量取 log1p 以压尾部; 因子均为交叉分类变量 (层用 early/mid/late 槽位, 跨模型可比)。",
         ""]

    json_out = {"timestamp": ts}

    if r40:
        eta, resid, levels, ok = eta_sq(
            r40, "y", ["metric", "model", "layer_slot", "task", "N"])
        json_out["part1_selector_cnorm"] = {"eta2": eta, "residual": resid,
                                            "n": len(r40),
                                            "levels": {k: v for k, v in levels.items()}}
        L += ["## Part 1. 选择器质量 (log1p C_norm) 的方差来源", ""]
        L += _table("因子占比 (全部组合)", eta, resid, levels)

    if r36:
        eta2, resid2, levels2, ok2 = eta_sq(r36, "y", ["model", "layer_slot", "task"])
        json_out["part2_feature_effect"] = {"eta2": eta2, "residual": resid2,
                                            "n": len(r36),
                                            "levels": {k: v for k, v in levels2.items()}}
        L += ["## Part 2. 单特征因果效应 (log1p C_ablate) 的方差来源", ""]
        L += _table("因子占比 (全部特征)", eta2, resid2, levels2)
        L += ["> 残差 = 同一 (模型, 层, 任务) 组合内不同特征之间的差异, 即「同类组合内选哪个特征」",
              "> 的可变部分; 它越小, 说明效应量越由组合 (模型/层/任务) 决定而非特征个体决定。", ""]

    (OUT / "variance_decomp.json").write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8")
    md = "\n".join(L)
    (OUT / "variance_decomp.md").write_text(md, encoding="utf-8")
    print(f"[完成] {OUT / 'variance_decomp.md'}")

    if REPORT.exists():
        import importlib
        import sys as _sys
        _sys.path.insert(0, str(BASE))
        m37 = importlib.import_module("37_summarize_causal_validity")
        m37._upsert_section(REPORT, "## 24. Step C: 零成本方差分解", md)
        print(f"[回填] {REPORT} §24")


if __name__ == "__main__":
    main()
