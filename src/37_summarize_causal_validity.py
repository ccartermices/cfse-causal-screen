# -*- coding: utf-8 -*-
"""Step 2-3: 因果效度汇总 (表 + 图 + 报告 §17 回填)

读取 output/phaseStep2/causal_validity/{model}/layer{L}/{task}.json,
生成:
  - output/phaseStep2/causal_validity/summary.json (脚本 36 已落盘, 此处补全聚合)
  - output/phaseStep2/causal_validity/causal_validity_summary.md (人读)
  - 图: cv1_metric_corr.png / cv2_scatter.png / cv3_ablate_vs_steer.png / cv4_baselines.png
  - 回填 output/phaseF/phaseF_report.md 的 §17 (幂等: 已有则替换)

资源: 纯 CPU 汇总, 无模型加载。严格节流。
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from importlib import import_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spearman = import_module("34_step1_metric_validity").spearman

METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
           "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]


def collect(root):
    runs = []
    for p in sorted(root.glob("*/layer*/*.json")):
        if p.name.startswith("resume_"):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "task" not in d or "model" not in d:
            continue
        d["_path"] = str(p)
        runs.append(d)
    return runs


def pooled_corr(runs, metric, target):
    """跨 run 池化 (逐 run 内 z-score 标准化后合并)."""
    xs, ys = [], []
    for r in runs:
        feats = r.get("features") or []
        a, b = [], []
        for f in feats:
            if f.get(metric) is None or f.get(target) is None:
                continue
            a.append(f[metric]); b.append(f[target])
        if len(a) < 5:
            continue
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.std() == 0 or b.std() == 0:
            continue
        xs.extend(((a - a.mean()) / a.std()).tolist())
        ys.extend(((b - b.mean()) / b.std()).tolist())
    return spearman(xs, ys), len(xs)


def main():
    ap = argparse.ArgumentParser(description="Step2-3: 因果效度汇总")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--report", type=str, default="output/phaseF/phaseF_report.md")
    args = ap.parse_args()

    base_dir = Path(args.config).parent
    root = base_dir / "output" / "phaseStep2" / "causal_validity"
    out_dir = root
    runs = collect(root)
    if not runs:
        print("[错误] 未找到任何 causal_validity 结果")
        return
    print(f"[汇总] 找到 {len(runs)} 个 (model,layer,task) 结果")

    ok = [r for r in runs if r.get("features") and not r.get("at_chance") and not r.get("error")]
    print(f"[汇总] 有效 (非 at_chance / 无错) 组合: {len(ok)}")

    # ---- 逐 run 概览表 ----
    rows = []
    for r in runs:
        rows.append({
            "model": r["model"], "layer": r.get("layer"), "task": r.get("task"),
            "error": r.get("error"), "at_chance": r.get("at_chance"),
            "b0_acc": r.get("b0_acc"), "align_max_abs_diff": r.get("align_max_abs_diff"),
            "hook_identity_ok": r.get("hook_identity_ok"),
            "batch_max_dev": (r.get("batching_check") or {}).get("max_abs_dev"),
            "n_features": len(r.get("features") or []),
            "base_random_feature": ((r.get("baselines") or {}).get("random_feature_ablate") or {}).get("mean"),
            "base_random_dir": ((r.get("baselines") or {}).get("random_dir_ablate") or {}).get("mean"),
            "base_all_feature": ((r.get("baselines") or {}).get("all_feature_ablate") or {}).get("C"),
            "base_mean_ablate": ((r.get("baselines") or {}).get("mean_ablate") or {}).get("C"),
        })

    # ---- 每指标 spearman(·, C_ablate/C_steer) 的跨 run 均值 ----
    per_metric = {}
    for metric in METRICS:
        for target in ("C_ablate", "C_steer"):
            vals = []
            for r in ok:
                v = (r.get("correlations") or {}).get(f"{metric}_vs_{target}")
                if v and v.get("spearman") is not None:
                    vals.append(v["spearman"])
            per_metric[f"{metric}_vs_{target}"] = {
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if vals else None,
                "n_runs": len(vals), "values": vals,
            }
    pooled = {}
    for metric in METRICS:
        for target in ("C_ablate", "C_steer"):
            s, n = pooled_corr(ok, metric, target)
            pooled[f"{metric}_vs_{target}"] = {"spearman": s, "n_points": n}

    # ---- 效果量分布: top 特征 vs 基线 ----
    dist = {}
    for r in ok:
        key = f"{r['model']}|L{r.get('layer')}|{r.get('task')}"
        ca = [f["C_ablate"] for f in r["features"] if f.get("C_ablate") is not None]
        cs = [f["C_steer"] for f in r["features"] if f.get("C_steer") is not None]
        b = r.get("baselines") or {}
        dist[key] = {
            "C_ablate_mean": float(np.mean(ca)), "C_ablate_max": float(np.max(ca)),
            "C_steer_mean": float(np.mean(cs)), "C_steer_max": float(np.max(cs)),
            "n_ablate_gt_baseline": int(sum(1 for v in ca
                                            if v > 3 * (((b.get("random_feature_ablate") or {}).get("std") or 1e-9)))),
            "random_feature_mean": (b.get("random_feature_ablate") or {}).get("mean"),
            "random_dir_mean": (b.get("random_dir_ablate") or {}).get("mean"),
            "all_feature_C": (b.get("all_feature_ablate") or {}).get("C"),
            "mean_ablate_C": (b.get("mean_ablate") or {}).get("C"),
        }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_runs_total": len(runs), "n_runs_valid": len(ok),
        "runs": rows, "per_metric_corr": per_metric, "pooled_corr": pooled,
        "effect_distribution": dist,
    }
    with open(out_dir / "causal_validity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- 图 ----
    _fig_metric_corr(per_metric, out_dir / "cv1_metric_corr.png")
    _fig_scatter(ok, out_dir / "cv2_scatter.png")
    _fig_ablate_vs_steer(ok, out_dir / "cv3_ablate_vs_steer.png")
    _fig_baselines(dist, out_dir / "cv4_baselines.png")

    # ---- 人读 md ----
    md = _render_md(summary)
    with open(out_dir / "causal_validity_summary.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[完成] {out_dir / 'causal_validity_summary.md'}")

    # ---- 回填报告 §17 ----
    rp = base_dir / args.report
    if rp.exists():
        _upsert_section(rp, "## 17. Phase Step2: SAE 指标的因果效度 (Causal Validity)",
                        md)
        print(f"[回填] {rp} §17")
    else:
        print(f"[跳过] 报告不存在: {rp}")


def _fig_metric_corr(per_metric, path):
    labels = [m for m in METRICS]
    ab = [per_metric.get(f"{m}_vs_C_ablate", {}).get("mean") for m in labels]
    st = [per_metric.get(f"{m}_vs_C_steer", {}).get("mean") for m in labels]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, [v if v is not None else 0 for v in ab], w, label="vs C_ablate")
    ax.bar(x + w / 2, [v if v is not None else 0 for v in st], w, label="vs C_steer")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("_", 1)[0] for m in labels])
    ax.set_ylabel("mean Spearman across runs")
    ax.set_title("Standard SAE metrics vs causal effect")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _fig_scatter(ok, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric in zip(axes, ("M2_mean_act_task", "M5_ctx_coherence")):
        for r in ok:
            xs = [f[metric] for f in r["features"] if f.get(metric) is not None
                  and f.get("C_ablate") is not None]
            ys = [f["C_ablate"] for f in r["features"] if f.get(metric) is not None
                  and f.get("C_ablate") is not None]
            if xs:
                ax.scatter(xs, ys, s=8, alpha=0.5,
                           label=f"{r['model'][:12]}|L{r.get('layer')}|{r.get('task')}")
        s = spearman([f[metric] for r in ok for f in r["features"]
                      if f.get(metric) is not None and f.get("C_ablate") is not None],
                     [f["C_ablate"] for r in ok for f in r["features"]
                      if f.get(metric) is not None and f.get("C_ablate") is not None])
        ax.set_xlabel(metric); ax.set_ylabel("C_ablate")
        ax.set_title(f"pooled spearman={s if s is None else round(s, 3)}")
        ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _fig_ablate_vs_steer(ok, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in ok:
        xs = [f["C_ablate"] for f in r["features"] if f.get("C_ablate") is not None
              and f.get("C_steer") is not None]
        ys = [f["C_steer"] for f in r["features"] if f.get("C_ablate") is not None
              and f.get("C_steer") is not None]
        if xs:
            ax.scatter(xs, ys, s=8, alpha=0.5,
                       label=f"{r['model'][:12]}|L{r.get('layer')}|{r.get('task')}")
    ax.set_xlabel("C_ablate"); ax.set_ylabel("C_steer")
    ax.set_title("Ablation vs steering effect")
    ax.grid(alpha=0.3)
    if len(ok) <= 8:
        ax.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _fig_baselines(dist, path):
    keys = list(dist.keys())[:12]
    labels = ["top-feat\nC_ablate(mean)", "top-feat\nC_ablate(max)",
              "random feat", "random dir", "mean-ablate", "all-feature"]
    vals = []
    for k in keys:
        d = dist[k]
        vals.append([d["C_ablate_mean"], d["C_ablate_max"],
                     d["random_feature_mean"] or 0, d["random_dir_mean"] or 0,
                     d["mean_ablate_C"] or 0, d["all_feature_C"] or 0])
    x = np.arange(len(keys)); w = 0.13
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(keys)), 4.5))
    for i, lb in enumerate(labels):
        ax.bar(x + (i - 2.5) * w, [v[i] for v in vals], w, label=lb)
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("|ΔB|")
    ax.set_title("Effect size: selected features vs baselines")
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def _render_md(summary):
    L = []
    L.append("")

    L.append(f"> 生成时间: {summary['timestamp']}")
    L.append(f"> 有效组合: {summary['n_runs_valid']} / {summary['n_runs_total']}")
    L.append("")
    L.append("### 17.1 逐组合概览")
    L.append("")
    L.append("| model | layer | task | B0 acc | 层对齐 max|diff| | hook恒等 | batch dev | n_feat | 随机特征C | 随机方向C | 全特征C | mean-ablate C | 状态 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary["runs"]:
        st = "OK"
        if r.get("error"):
            st = f"ERR: {str(r['error'])[:40]}"
        elif r.get("at_chance"):
            st = "at_chance"
        fmt = lambda v, n=4: ("—" if v is None else f"{v:.{n}f}")
        bd = r.get("batch_max_dev")
        bd_s = "—" if bd is None else f"{bd:.1e}"
        L.append(f"| {r['model']} | {r.get('layer')} | {r.get('task')} | "
                 f"{fmt(r.get('b0_acc'), 3)} | {fmt(r.get('align_max_abs_diff'), 1)} | "
                 f"{r.get('hook_identity_ok')} | {bd_s} | "
                 f"{r.get('n_features')} | {fmt(r.get('base_random_feature'))} | "
                 f"{fmt(r.get('base_random_dir'))} | {fmt(r.get('base_all_feature'), 3)} | "
                 f"{fmt(r.get('base_mean_ablate'), 3)} | {st} |")

    L.append("")
    L.append("### 17.2 标准指标 vs 因果效应量 (跨组合 spearman 均值)")
    L.append("")
    L.append("| metric | vs C_ablate (mean±std, n) | vs C_steer (mean±std, n) | pooled(ablate) | pooled n |")
    L.append("|---|---|---|---|---|")
    for m in METRICS:
        a = summary["per_metric_corr"].get(f"{m}_vs_C_ablate", {})
        b = summary["per_metric_corr"].get(f"{m}_vs_C_steer", {})
        p = summary["pooled_corr"].get(f"{m}_vs_C_ablate", {})
        f1 = "—" if a.get("mean") is None else f"{a['mean']:+.3f}±{a['std']:.3f} (n={a['n_runs']})"
        f2 = "—" if b.get("mean") is None else f"{b['mean']:+.3f}±{b['std']:.3f} (n={b['n_runs']})"
        f3 = "—" if p.get("spearman") is None else f"{p['spearman']:+.3f}"
        L.append(f"| {m} | {f1} | {f2} | {f3} | {p.get('n_points')} |")

    L.append("")
    L.append("### 17.3 效应量分布 vs 基线")
    L.append("")
    L.append("| 组合 | top C_ablate(mean) | top C_ablate(max) | 随机特征(mean) | 随机方向(mean) | 全特征 | mean-ablate |")
    L.append("|---|---|---|---|---|---|---|")
    for k, d in summary["effect_distribution"].items():
        g = lambda v: "—" if v is None else f"{v:.4f}"
        L.append(f"| {k} | {g(d['C_ablate_mean'])} | {g(d['C_ablate_max'])} | "
                 f"{g(d['random_feature_mean'])} | {g(d['random_dir_mean'])} | "
                 f"{g(d['all_feature_C'])} | {g(d['mean_ablate_C'])} |")
    L.append("")
    L.append("**图**: `cv1_metric_corr.png`(指标相关) / `cv2_scatter.png`(M–C 散点) / "
             "`cv3_ablate_vs_steer.png`(ablate vs steer) / `cv4_baselines.png`(效应量对比)")
    L.append("")
    L.append("> 口径说明: 本节的因果效应量由 **per-layer SAE** 的单特征 ablation/steering 得到, "
             "与 §16 的「整体注入分布级损伤 (kl_base_inj)」是不同口径; "
             "Phase E/phase3 的 auto-interp 分数基于跨层池化 SAE 的特征索引空间, 与 per-layer SAE 不可通约, "
             "故 M5 用本地无 LLM 的上下文 TF-IDF 一致性作为 auto-interp 代理。")
    L.append("")
    return "\n".join(L)


def _upsert_section(report_path, header, body):
    """幂等: 若已存在同 header 的 §17, 则替换其内容; 否则追加."""
    txt = report_path.read_text(encoding="utf-8")
    lines = txt.split("\n")
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(header.strip()):
            start = i
            break
    if start is not None:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if re.match(r"^## \d+\.", lines[j]):
                end = j
                break
        new = lines[:start] + [header] + body.split("\n") + lines[end:]
    else:
        new = lines + [header] + body.split("\n")
    report_path.write_text("\n".join(new), encoding="utf-8")


if __name__ == "__main__":
    main()
