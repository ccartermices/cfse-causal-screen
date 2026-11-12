# -*- coding: utf-8 -*-
"""Trace every number quoted in the paper to the file it came from.

Each entry is (label, file, dotted path). Run this after the pipeline and compare
the output with the manuscript; a mismatch means the manuscript is stale, never the
other way round.

    python scripts/make_numbers.py            # writes docs/NUMBERS.md
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "NUMBERS.md"

# (label, file, path, format)
ENTRIES = [
    # ---- main audit ------------------------------------------------------
    ("pooled Spearman, M2 (60 combinations)", "pooled_audit.json",
     "same_prompt.M2_mean_act_task.pooled", "{:+.3f}"),
    ("pooled Spearman, predictor", "pooled_audit.json",
     "same_prompt.pred_full.pooled", "{:+.3f}"),
    ("per-combination mean rho, M1", "cfse_summary.json",
     "feature.M1_act_freq_task.rho_abs.mean", "{:+.3f}"),
    ("per-combination mean rho, M4", "cfse_summary.json",
     "feature.M4_recon_mse_gain.rho_abs.mean", "{:+.3f}"),
    ("feature-combination entries", "cfse_summary.json", "n_features_total", "{:d}"),
    ("combinations", "cfse_summary.json", "n_combos", "{:d}"),
    ("first-order predictor, held-out per-combination", "attrib_analysis.json",
     "full.pred_full.heldout_per_combo", "{:+.3f}"),
    ("measured effect, median |dB| (fp32)", "e1_summary.json",
     "results.0.median_abs_effect_f32_b16", "{:.4f}"),

    # ---- gate ------------------------------------------------------------
    ("gate: in-sample agreement", "e7_stats.json",
     "C09_gate_calibration.in_sample_agree", None),
    ("gate: leave-one-combination-out agreement", "e7_stats.json",
     "C09_gate_calibration.loco_agree", None),
    ("gate: 48-grid agreement", "e7_stats.json",
     "C09_gate_calibration.grid_agree", None),
    ("gate: false admissions on the grid", "e7_stats.json",
     "C09_gate_calibration.grid_false_admit", "{:d}"),

    # ---- batch / precision audit ----------------------------------------
    ("cell-level r, same precision different batch", "e1_aggregate.json",
     "results.0.same_dtype_batch_ge8.median_cell_pearson", "{:.3f}"),
    ("feature-level rho, same precision different batch", "e1_aggregate.json",
     "results.0.same_dtype_batch_ge8.median_feat_spearman", "{:.3f}"),
    ("noise / effect", "e1_aggregate.json",
     "results.0.same_dtype_batch_ge8.median_cell_ratio", "{:.2f}"),
    ("float32 cell r", "e1_fp32_summary.json", "min_cell_pearson", "{:.6f}"),
    ("float32 median cell deviation", "e1_fp32_summary.json",
     "max_median_abs_cell_diff", "{:.0e}"),

    # ---- dose response ---------------------------------------------------
    ("dose: rho at alpha=0.25", "e8_dose_overview.json", "rho_quarter", "{:.3f}"),
    ("dose: rho at alpha=1", "e8_dose_overview.json", "rho_one", "{:.3f}"),
    ("dose: rho at alpha=5", "e8_dose_overview.json", "rho_five", "{:.3f}"),
    ("dose: effect ratio at alpha=5", "e8_dose_overview.json", "effect_ratio", "{:.1f}"),

    # ---- CE / KL ---------------------------------------------------------
    ("rho(|dB|, |dCE|)", "e5_summary.json", "results.0.effect_corr.abs_dmargin_vs_abs_dce",
     "{:.3f}"),
    ("rho(|dB|, dKL)", "e5_summary.json", "results.0.effect_corr.abs_dmargin_vs_dkl",
     "{:.3f}"),
    ("predictor on margin", "e5_summary.json", "results.0.pred_rho.margin", "{:.3f}"),
    ("predictor on KL", "e5_summary.json", "results.0.pred_rho.kl", "{:.3f}"),

    # ---- SAE-free --------------------------------------------------------
    ("SAE-free: SAE basis", "e3_summary.json", "results.0.rho.sae_pred", "{:.3f}"),
    ("SAE-free: coordinate basis", "e3_summary.json", "results.0.rho.coord", "{:.3f}"),
    ("SAE-free: random basis", "e3_summary.json", "results.0.rho.random_dir", "{:.3f}"),

    # ---- review follow-ups ----------------------------------------------
    ("SHAP: K~63 vs exact (F=8)", "e8_summary.json", "shap.f8_budget.63", "{:.3f}"),
    ("SHAP: K~108 vs exact (F=8)", "e8_summary.json", "shap.f8_budget.108", "{:.3f}"),
    ("SHAP: K~182 vs exact (F=8)", "e8_summary.json", "shap.f8_budget.182", "{:.3f}"),
    ("SHAP: K=128 vs K=512 (F=16)", "e8_summary.json", "shap.f16_pairs.128_vs_512", "{:.3f}"),
    ("SHAP: seed-to-seed, K=128", "e8_summary.json", "shap.seed_median", "{:.3f}"),
    ("IG: completeness at 128 steps", "e8_summary.json", "ig.128.completeness", "{:.3f}"),
    ("IG: projection residual", "e8_summary.json", "ig.128.residual", "{:.3f}"),
    ("IG: standard-mean-baseline rho", "e8_ig_path_median.json", "median_mean_baseline", "{:+.3f}"),
    ("IG: local-baseline rho", "e8_ig_path_median.json", "median_local_baseline", "{:+.3f}"),
    ("union pool: predictor rho", "e9_summary.json", "metrics.pred_full.mean", "{:+.3f}"),
    ("union pool: M2 rho", "e9_summary.json", "metrics.M2.mean", "{:+.3f}"),
    ("union pool: metric eta^2", "e9_summary.json", "eta2_metric", "{:.3f}"),
    ("SAE seeds: rho(M2) spread", "e10_summary.json",
     "summary.M2_mean_act_task.rho_std", "{:.3f}"),
]


def dig(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def fmt(v, f):
    if v is None:
        return "—"
    if f is None:
        if isinstance(v, list):
            return f"[{len(v)} entries]"
        return str(v)
    try:
        return f.format(v)
    except Exception:
        return str(v)


def main():
    lines = ["# Number trace", "",
             "Generated by `scripts/make_numbers.py`. Every row is a direct read of a file "
             "in `results/`; the rightmost column is the JSON path, so a reader can verify "
             "any value with one `jq` call.", "",
             "| quantity | value | file | path |", "|---|---|---|---|"]
    missing = 0
    for label, fname, path, f in ENTRIES:
        p = RES / fname
        if not p.exists():
            lines.append(f"| {label} | _missing_ | {fname} | {path} |")
            missing += 1
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            lines.append(f"| {label} | _unreadable_ | {fname} | {path} |")
            missing += 1
            continue
        lines.append(f"| {label} | {fmt(dig(data, path), f)} | `{fname}` | `{path}` |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {OUT.relative_to(ROOT)}  ({len(ENTRIES) - missing}/{len(ENTRIES)} resolved)")
    if missing:
        print(f"   {missing} entries unresolved; run the pipeline stage that produces them")


if __name__ == "__main__":
    main()
