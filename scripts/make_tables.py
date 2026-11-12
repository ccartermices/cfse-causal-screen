# -*- coding: utf-8 -*-
"""Rebuild every table in the paper from the shipped result files.

Only numpy is required. Nothing under `results/` is re-computed here: this script
formats the numbers that the pipeline wrote, so a reader can check the paper
against the released artefacts without a GPU.

Usage
-----
    python scripts/make_tables.py            # writes tables/*.csv and tables/*.md
    python scripts/make_tables.py --latex    # also prints LaTeX bodies
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "tables"

METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
           "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]
NICE = {"M1_act_freq_task": "M1 activation frequency",
        "M2_mean_act_task": "M2 mean activation",
        "M3_max_act_task": "M3 max activation",
        "M4_recon_mse_gain": "M4 reconstruction gain",
        "M5_ctx_coherence": "M5 context coherence",
        "M6_corpus_freq": "M6 corpus frequency"}


def load(name):
    p = RES / name
    if not p.exists():
        print(f"  [skip] {name} not found")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def write(name, rows, header):
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"{name}.csv"
    md = OUT / f"{name}.md"
    with open(csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(md, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    print(f"  -> tables/{name}.csv, tables/{name}.md ({len(rows)} rows)")


# --------------------------------------------------------------- Table 1
def table_gap():
    d = load("cfse_summary.json")
    if not d:
        return
    audit = load("pooled_audit.json") or {}
    pooled = {k: v.get("pooled") for k, v in (audit.get("same_prompt") or {}).items()}
    rows = []
    for m in METRICS:
        v = d["feature"].get(m) or {}
        rho = v.get("rho_abs") or {}
        p = pooled.get(m)
        rows.append([NICE[m],
                     f"{rho.get('mean', float('nan')):+.3f}",
                     f"{rho.get('sd', float('nan')):.3f}",
                     (f"{p:+.3f}" if p is not None else "-"),
                     rho.get("n", 0)])
    write("tab1_standard_metrics", rows,
          ["metric", "mean rho (per combination)", "sd", "pooled rho", "n"])


# --------------------------------------------------------------- gate
def table_gate():
    d = load("e7_stats.json")
    if not d:
        return
    cal = d.get("C09_gate_calibration") or {}
    gp = cal.get("grid_points") or []
    if not gp:
        return
    n = len(gp)
    agree = sum(1 for g in gp if g.get("verdict") == "admit"
                and g.get("rel_measured", 0) >= 0.25)
    rows = [["grid points", n],
            ["admit", sum(1 for g in gp if g.get("verdict") == "admit")],
            ["degrade", sum(1 for g in gp if g.get("verdict") == "degrade")],
            ["reject", sum(1 for g in gp if g.get("verdict") == "reject")],
            ["admit with measured rel>=0.25", agree]]
    write("tab_gate", rows, ["quantity", "value"])


# --------------------------------------------------------------- repro
def table_repro():
    s = load("e1_summary.json")
    a = load("e1_aggregate.json")
    if not s:
        return
    agg = {}
    if a:
        agg = {(x["model"], x["layer"], x["task"]): x for x in a["results"]}
    rows = []
    for r in s["results"]:
        x = agg.get((r["model"], r["layer"], r["task"]), {})
        sd = x.get("same_dtype_batch_ge8", {})
        rows.append([r["model"], r["layer"], r["task"],
                     f"{r['median_abs_effect_f32_b16']:.4f}",
                     f"{sd.get('median_cell_pearson', float('nan')):.3f}",
                     f"{sd.get('median_feat_spearman', float('nan')):.3f}",
                     f"{sd.get('median_cell_ratio', float('nan')):.2f}"])
    write("tab_repro", rows,
          ["model", "layer", "task", "median |dB|", "cell r", "feature rho",
           "noise/effect"])


# --------------------------------------------------------------- dose
def table_dose():
    d = load("e4_summary.json")
    if not d:
        return
    rows = []
    alphas = None
    for r in d["results"]:
        curve = {c["alpha"]: c for c in r["curve"]}
        if alphas is None:
            alphas = sorted(curve)
        rows.append([r["model"], r["layer"], r["task"]]
                    + [f"{curve[a]['rho_all']:.3f}" if a in curve else "-"
                       for a in alphas])
    write("tab_dose", rows,
          ["model", "layer", "task"] + [f"rho(a={a})" for a in (alphas or [])])


# --------------------------------------------------------------- CE / KL
def table_cekl():
    d = load("e5_summary.json")
    if not d:
        return
    rows = []
    for r in d["results"]:
        ec = r.get("effect_corr") or {}
        pr = r.get("pred_rho") or {}
        rows.append([r["model"], r["layer"], r["task"],
                     f"{ec.get('abs_dmargin_vs_abs_dce', float('nan')):.3f}",
                     f"{ec.get('abs_dmargin_vs_dkl', float('nan')):.3f}",
                     f"{ec.get('abs_dce_vs_dkl', float('nan')):.3f}",
                     f"{pr.get('margin', float('nan')):.3f}",
                     f"{pr.get('ce', float('nan')):.3f}",
                     f"{pr.get('kl', float('nan')):.3f}"])
    write("tab_cekl", rows,
          ["model", "layer", "task", "rho(margin,CE)", "rho(margin,KL)",
           "rho(CE,KL)", "pred-margin", "pred-CE", "pred-KL"])


# --------------------------------------------------------------- SAE-free
def table_saefree():
    d = load("e3_summary.json")
    if not d:
        return
    rows = []
    for r in d["results"]:
        rho = r.get("rho") or {}
        rows.append([r["model"], r["layer"], r["task"]]
                    + [f"{rho.get(k, float('nan')):.3f}" for k in
                       ("sae", "coord", "random", "pca")]
                    + [f"{r.get('delta_sae_minus_coord', float('nan')):+.3f}"])
    write("tab_saefree", rows,
          ["model", "layer", "task", "SAE", "coordinate", "random", "PCA",
           "SAE - coordinate"])


# --------------------------------------------------------------- SHAP / IG
def table_shapig():
    d = load("e8_summary.json")
    if not d:
        return
    rows = []
    sh = d.get("shap") or {}
    for k, v in (sh.get("f8_budget") or {}).items():
        rows.append(["F=8, K=" + k, f"{v:.3f}"])
    for k, v in (sh.get("f16_pairs") or {}).items():
        rows.append(["F=16, " + k.replace("_vs_", " vs "), f"{v:.3f}"])
    for k, v in (sh.get("f16_rho_vs_effect") or {}).items():
        rows.append(["F=16, K=" + k + " vs |dB|", f"{v:+.3f}"])
    rows.append(["F=16, K=128 seed median", f"{sh.get('seed_median', float('nan')):.3f}"])
    for s, v in (d.get("ig") or {}).items():
        rows.append([f"IG steps={s}, rho", f"{v['rho']:+.3f}"])
        rows.append([f"IG steps={s}, completeness", f"{v['completeness']:.3f}"])
        rows.append([f"IG steps={s}, projection residual", f"{v['residual']:.3f}"])
    write("tab_shap_ig", rows, ["quantity", "value"])


# --------------------------------------------------------------- union pool
def table_union():
    d = load("e9_summary.json")
    if not d:
        return
    rows = []
    for k, v in (d.get("metrics") or {}).items():
        rows.append([k, f"{v['mean']:+.3f}", f"{v['std']:.3f}",
                     f"{v['min']:+.3f}", f"{v['max']:+.3f}", v["n"]])
    write("tab_union_pool", rows, ["metric", "mean rho", "sd", "min", "max", "n"])
    print(f"  metric-factor eta^2 = {d.get('eta2_metric', float('nan')):.3f}")


# --------------------------------------------------------------- SAE seeds
def table_saeseeds():
    d = load("e10_summary.json")
    if not d:
        return
    rows = []
    for k, v in (d.get("summary") or {}).items():
        rows.append([k, f"{v['rho_mean']:+.3f}", f"{v['rho_std']:.3f}",
                     f"{v['rho_min']:+.3f}", f"{v['rho_max']:+.3f}",
                     f"{v['jaccard_mean']:.2f}"])
    write("tab_sae_seeds", rows,
          ["metric", "rho mean", "sd", "min", "max", "top-50 Jaccard"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.parse_args()
    print(f"results: {RES}")
    table_gap()
    table_gate()
    table_repro()
    table_dose()
    table_cekl()
    table_saefree()
    table_shapig()
    table_union()
    table_saeseeds()


if __name__ == "__main__":
    main()
