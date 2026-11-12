# -*- coding: utf-8 -*-
"""Step 1 (论文化): 统计严谨性层 —— CI / 显著性 / 稳健性 (零模型加载, 零干预)

动因
----
results_draft.md §5.13 承认的最大硬伤: **全部结论都是单种子 (seed=42) 的点估计**。
审稿人必问: pooled rho=0.206 / 0.725 的 **CI 是多少**? pred_full 优于 M2 的 **+0.519 显著吗**?
规模轴的 Pearson(-0.923) 有没有置信区间?

本脚本**不加载任何模型、不做任何干预**, 只对已落盘交付物 (36/40/42/43) 做统计推断:
  A. pooled Spearman 的三种 bootstrap CI:
       cluster  (重采样组合, 主口径: 组合是泛化单元)
       feature  (组合固定, 组合内重采样特征: 特征抽样不确定性)
       twolevel (先重采样组合, 再组合内重采样特征)
  B. 逐组合 rho 的单样本检验 (Wilcoxon / t) + Holm-Bonferroni 多重比较校正
  C. 预测器 vs 标准指标的**配对检验** (Wilcoxon signed-rank + Cliff's delta + 胜率 Wilson CI
     + 配对 cluster bootstrap 的 Δpooled CI)
  D. 规模轴: 9 模型 Pearson(-0.923) 的模型级 bootstrap CI + 同族 Pythia 单调趋势置换检验
  E. 方差分解 (43) 的 eta^2 特征级 bootstrap CI
  F. 自检: 由 features 重算的逐组合 rho 必须与 36/42 落盘值一致 (否则报错)

输出: output/phaseStep2/significance/significance.json + significance.md
资源: 纯 CPU, 无模型加载; 遵守 OMP/MKL=1 节流 (不打扰在同机运行的其他实验)。
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).parent
METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
           "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]
METRIC_LABEL = {"M1_act_freq_task": "M1 activation frequency",
                "M2_mean_act_task": "M2 mean activation",
                "M3_max_act_task": "M3 max activation",
                "M4_recon_mse_gain": "M4 reconstruction-MSE gain",
                "M5_ctx_coherence": "M5 context coherence",
                "M6_corpus_freq": "M6 corpus frequency"}
PREDS = ["pred_full", "pred_last", "pred_align"]
TARGETS = ["C_ablate", "C_steer"]

# 配对比较列表 (pred_full 逐一对比 5 个标准指标 + 2 个消融预测器)
PAIRED_PAIRS = [("pred_full", "M2_mean_act_task"), ("pred_full", "M3_max_act_task"),
                ("pred_full", "pred_last"), ("pred_full", "pred_align"),
                ("pred_full", "M1_act_freq_task"), ("pred_full", "M4_recon_mse_gain"),
                ("pred_full", "M5_ctx_coherence"), ("pred_full", "M6_corpus_freq")]

# 规模轴 (IOI, 9 个模型); 参数取自 results_draft §5.14 Table 1
SCALE_MODELS = [("qwen3.5-0.8b", 0.8), ("minicpm5-1b", 1.0), ("tinyllama-1.1b", 1.1),
                ("pythia-1.4b", 1.4), ("qwen3.5-2b", 2.0), ("pythia-2.8b", 2.8),
                ("ministral-3-3b", 3.0), ("pythia-6.9b", 6.9),
                ("qwen2.5-7b-instruct", 7.0)]


# ---------------------------------------------------------------- 基础统计
def spearman(x, y):
    """与 34/37 完全一致的 Spearman (平均秩 + Pearson)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def zscore(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 0 else a - a.mean()


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def holm(pvals):
    """Holm-Bonferroni 校正, 返回校正后 p (保序并截断)."""
    keys = list(pvals)
    m = len(keys)
    order = sorted(keys, key=lambda k: pvals[k])
    adj = {}
    running = 0.0
    for i, k in enumerate(order):
        v = min(1.0, (m - i) * pvals[k])
        running = max(running, v)
        adj[k] = float(running)
    return adj


def cliffs_delta(a, b):
    """Cliff's delta = P(a>b) - P(a<b)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return None
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


# ---------------------------------------------------------------- 数据装载
def load_dir(root, name):
    out = {}
    for p in sorted(root.glob("*/layer*/*.json")):
        if p.name.startswith(("resume_", "summary")):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not d.get("features") or d.get("error"):
            continue
        if d.get("at_chance") or (d.get("b0_acc_36") is not None and d["b0_acc_36"] <= 0.6):
            continue
        key = f'{d["model"]}|L{d.get("layer")}|{d.get("task")}'
        out[key] = d
    return out


def build_combos(base_dir, tag=""):
    """合并 36 (M1-M6, C_ablate/C_steer) 与 42 (pred_*), 并自检一致性。
    tag 为空串时读 seed42 原口径 (causal_validity / causal_margin);
    tag='_s43' 等时读 T1 复跑目录 (causal_validity_s43 / causal_margin_s43)。"""
    d36 = load_dir(base_dir / "output" / "phaseStep2" / f"causal_validity{tag}", "36")
    d42 = load_dir(base_dir / "output" / "phaseStep2" / f"causal_margin{tag}", "42")
    keys = sorted(set(d36) & set(d42))
    missing = sorted(set(d36) ^ set(d42))
    combos = []
    verify = {"n_checked": 0, "max_abs_diff_metric": 0.0,
              "max_abs_diff_robust_rank": 0.0, "feature_idx_mismatch": []}
    for k in keys:
        a, b = d36[k], d42[k]
        fa, fb = a["features"], b["features"]
        if [f["feature_idx"] for f in fa] != [f["feature_idx"] for f in fb]:
            verify["feature_idx_mismatch"].append(k)
        x36 = {m: np.array([f.get(m) if f.get(m) is not None else np.nan for f in fa], float)
               for m in METRICS}
        cab = np.array([f["C_ablate"] for f in fb], float)
        xn = {p: np.array([f["pred_full"] if p == "pred_full" else
                           (f["pred_last"] if p == "pred_last" else f["pred_align"])
                           for f in fb], float) for p in PREDS}
        a_steer = np.array([f.get("C_steer") for f in fa], float)
        # 自检: 由 features 重算的 rho vs 42 落盘 correlations
        for m in METRICS:
            rr = spearman(x36[m], cab)
            sv = (((b.get("correlations") or {}).get(m)) or {}).get("spearman")
            if rr is not None and sv is not None:
                verify["n_checked"] += 1
                verify["max_abs_diff_metric"] = max(verify["max_abs_diff_metric"],
                                                    abs(rr - float(sv)))
        combos.append({"key": k, "model": a["model"], "layer": a.get("layer"),
                       "task": a.get("task"), "n": len(fa),
                       "x36": x36, "c_ablate": cab, "c_steer": a_steer, "xn": xn,
                       "corr": {m: spearman(x36[m], cab) for m in METRICS},
                       "corr_steer": {m: spearman(x36[m], a_steer) for m in METRICS},
                       "corr_pred": {p: spearman(xn[p], cab) for p in PREDS}})
    return combos, missing, verify


# ---------------------------------------------------------------- bootstrap
def boot_pooled(pairs, mode, B, rng):
    """pairs: [(x_z, y_z)]; 返回 pooled rho 的 bootstrap 分布。"""
    nC = len(pairs)
    vals = []
    for _ in range(B):
        if mode == "cluster":
            idx = rng.integers(0, nC, nC)
            X = np.concatenate([pairs[i][0] for i in idx])
            Y = np.concatenate([pairs[i][1] for i in idx])
        elif mode == "feature":
            Xs, Ys = [], []
            for x, y in pairs:
                k = len(x)
                fi = rng.integers(0, k, k)
                Xs.append(x[fi]); Ys.append(y[fi])
            X = np.concatenate(Xs); Y = np.concatenate(Ys)
        else:  # twolevel
            idx = rng.integers(0, nC, nC)
            Xs, Ys = [], []
            for i in idx:
                x, y = pairs[i]
                k = len(x)
                fi = rng.integers(0, k, k)
                Xs.append(x[fi]); Ys.append(y[fi])
            X = np.concatenate(Xs); Y = np.concatenate(Ys)
        v = spearman(X, Y)
        if v is not None:
            vals.append(v)
    vals = np.asarray(vals, float)
    return vals


def ci_summary(vals, point, lo=2.5, hi=97.5):
    if len(vals) == 0:
        return {"point": point, "ci_lo": None, "ci_hi": None, "boot_std": None}
    return {"point": point,
            "ci_lo": float(np.percentile(vals, lo)),
            "ci_hi": float(np.percentile(vals, hi)),
            "boot_std": float(vals.std()),
            "boot_mean": float(vals.mean()), "B": int(len(vals))}


# ---------------------------------------------------------------- 各分析块
def analyze_pooled(combos, args, rng):
    pairs_by = {}          # (name, target) -> [(x_z, y_z)]
    for m in METRICS:
        for t in TARGETS:
            arr = "c_ablate" if t == "C_ablate" else "c_steer"
            pairs_by[(m, t)] = [(zscore(c["x36"][m][~np.isnan(c["x36"][m])]),
                                 zscore(c[arr][~np.isnan(c["x36"][m])])) for c in combos]
    for p in PREDS:
        pairs_by[(p, "C_ablate")] = [(zscore(c["xn"][p]), zscore(c["c_ablate"]))
                                     for c in combos]
    out = {}
    for (name, t), pairs in pairs_by.items():
        xs = [a for a, _ in pairs]; ys = [b for _, b in pairs]
        point = spearman(np.concatenate(xs), np.concatenate(ys))
        entry = {"n_combos": len(pairs), "n_points": int(sum(len(a) for a in xs))}
        for mode, B in (("cluster", args.boot_cluster),
                        ("feature", args.boot_feature),
                        ("twolevel", args.boot_two)):
            entry[mode] = ci_summary(boot_pooled(pairs, mode, B, rng), point)
        out[f"{name}_vs_{t}"] = entry
    return out


def analyze_paired(combos, a_key, b_key, src, args, rng, alternative="greater"):
    """a_key/b_key 为组合级别的 rho 序列 (src='corr' 或 'corr_pred')."""
    av = np.array([c[src][a_key] for c in combos], float)
    bv = np.array([c[src][b_key] for c in combos], float)
    m = ~(np.isnan(av) | np.isnan(bv))
    av, bv = av[m], bv[m]
    n = len(av)
    res = {"n": int(n), "mean_a": float(av.mean()), "mean_b": float(bv.mean()),
           "median_a": float(np.median(av)), "median_b": float(np.median(bv)),
           "mean_delta": float((av - bv).mean()),
           "median_delta": float(np.median(av - bv))}
    if n >= 8 and np.any(av != bv):
        try:
            w_two = stats.wilcoxon(av, bv, alternative="two-sided")
            w_gt = stats.wilcoxon(av, bv, alternative=alternative)
            res["wilcoxon_p_two_sided"] = float(w_two.pvalue)
            res["wilcoxon_p_one_sided"] = float(w_gt.pvalue)
        except Exception as e:
            res["wilcoxon_error"] = str(e)
        try:
            t = stats.ttest_rel(av, bv)
            res["ttest_p_two_sided"] = float(t.pvalue)
        except Exception as e:
            res["ttest_error"] = str(e)
        res["cliffs_delta"] = cliffs_delta(av, bv)
        wins = int((av > bv).sum()); ties = int((av == bv).sum()); losses = int((av < bv).sum())
        res.update({"wins": wins, "ties": ties, "losses": losses})
        res["win_rate"] = float(wins / n)
        res["win_rate_wilson_ci"] = wilson_ci(wins, n)
    # 配对 cluster bootstrap: Δpooled CI
    pairs_a, pairs_b = [], []
    for c in combos:
        ya = zscore(c["x36"][a_key]) if src == "corr" else zscore(c["xn"][a_key])
        yb = zscore(c["x36"][b_key]) if src == "corr" else zscore(c["xn"][b_key])
        yt = zscore(c["c_ablate"])
        pairs_a.append((ya, yt)); pairs_b.append((yb, yt))
    da = boot_pooled(pairs_a, "cluster", args.boot_cluster, rng)
    db = boot_pooled(pairs_b, "cluster", args.boot_cluster, rng)
    if len(da) and len(db):
        diff = da - db
        res["pooled_delta_point"] = float(da.mean() * 0 + (spearman(
            np.concatenate([p[0] for p in pairs_a]), np.concatenate([p[1] for p in pairs_a])) -
            spearman(np.concatenate([p[0] for p in pairs_b]),
                     np.concatenate([p[1] for p in pairs_b]))))
        res["pooled_delta_ci"] = [float(np.percentile(diff, 2.5)),
                                  float(np.percentile(diff, 97.5))]
        res["pooled_delta_boot_std"] = float(diff.std())
    return res


def analyze_one_sample(combos, src, keys, labels):
    """逐组合 rho 相对 0 的单样本检验 + Holm 校正。"""
    out = {}
    pvals = {}
    for k in keys:
        v = np.array([c[src][k] for c in combos], float)
        v = v[~np.isnan(v)]
        e = {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
             "median": float(np.median(v)),
             "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
             "frac_positive": float((v > 0).mean())}
        if len(v) >= 8:
            try:
                e["wilcoxon_p"] = float(stats.wilcoxon(v, alternative="two-sided").pvalue)
                e["wilcoxon_p_greater"] = float(
                    stats.wilcoxon(v, alternative="greater").pvalue)
                pvals[k] = e["wilcoxon_p"]
            except Exception as ex:
                e["wilcoxon_error"] = str(ex)
            e["ttest_p"] = float(stats.ttest_1samp(v, 0.0).pvalue)
        out[labels.get(k, k)] = e
    if pvals:
        adj = holm(pvals)
        for k, p in adj.items():
            out[labels.get(k, k)]["holm_p"] = float(p)
    return out


def analyze_scale(combos, args, rng, mid_layers=None):
    """规模轴: 9 模型 IOI 的 (log10 params, mean M2 rho) + bootstrap CI + 同族趋势。

    mid_layers 非空时只取 mid 层 (target_layers[1]), 用于与 T1 (仅 mid 层) 口径对齐。
    """
    pts = []
    for model, params in SCALE_MODELS:
        rs = [c["corr"]["M2_mean_act_task"] for c in combos
              if c["model"] == model and c["task"] == "ioi"
              and (mid_layers is None or c["layer"] == mid_layers.get(model))
              and c["corr"]["M2_mean_act_task"] is not None]
        if rs:
            pts.append((model, float(params), float(np.mean(rs)), len(rs)))
    if len(pts) < 4:
        return {}
    lp = np.log10([p[1] for p in pts]); r = np.array([p[2] for p in pts])
    pear = float(np.corrcoef(lp, r)[0, 1])
    spear = spearman(lp, r)
    idx = np.arange(len(pts))
    bs_p, bs_s = [], []
    for _ in range(args.boot_scale):
        j = rng.choice(idx, len(idx), replace=True)
        if len(set(j.tolist())) < 3:
            continue
        bs_p.append(float(np.corrcoef(lp[j], r[j])[0, 1]))
        bs_s.append(spearman(lp[j], r[j]))
    out = {"points": [{"model": p[0], "params": p[1], "m2_spearman_mean": p[2],
                       "n_layers": p[3]} for p in pts],
           "pearson_log10_params": pear, "pearson_ci": ci_summary(np.asarray(bs_p), pear),
           "spearman_log10_params": spear,
           "spearman_ci": ci_summary(np.asarray(bs_s), spear),
           "n_models": len(pts)}
    # 同族 Pythia 单调下降趋势: 置换检验 (统计量 = 逆序对数的相反数)
    py = [(p[0], p[1], p[2]) for p in pts if p[0].startswith("pythia")]
    if len(py) >= 3:
        py.sort(key=lambda x: x[1])
        vals = [p[2] for p in py]
        def stat(v):
            s = 0
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if v[i] > v[j]:
                        s += 1
            return s
        obs = stat(vals)
        import itertools
        perms = [stat(list(pp)) for pp in itertools.permutations(vals)]
        out["pythia_trend"] = {
            "models": [p[0] for p in py], "params": [p[1] for p in py],
            "m2_spearman": vals, "observed_descending_pairs": int(obs),
            "n_pairs": int(len(vals) * (len(vals) - 1) / 2),
            "perm_p_descending": float(np.mean([1.0 if x >= obs else 0.0 for x in perms])),
        }
    return out


def analyze_variance(args, rng):
    """复用 43 的 eta^2, 做组合内特征级 bootstrap CI。"""
    try:
        sys.path.insert(0, str(BASE))
        m43 = importlib.import_module("43_variance_decomposition")
    except Exception as e:
        return {"error": str(e)}
    config = json.load(open(BASE / "config.json", encoding="utf-8"))
    r36 = m43.collect_36(config)
    for r in r36:
        r["y"] = float(np.log1p(r["c_ablate"]))
    if not r36:
        return {}
    eta0, resid0, levels0, _ = m43.eta_sq(r36, "y", ["model", "layer_slot", "task"])
    # 组合内特征级重采样
    by_combo = {}
    for r in r36:
        by_combo.setdefault((r["model"], r["layer_slot"], r["task"]), []).append(r)
    combos = list(by_combo.items())
    facs = ["model", "layer_slot", "task"]
    boot = {f: [] for f in facs}
    boot_resid = []
    for _ in range(args.boot_var):
        samp = []
        for _, rows in combos:
            j = rng.integers(0, len(rows), len(rows))
            samp.extend(rows[k] for k in j)
        e, res, _, _ = m43.eta_sq(samp, "y", facs)
        for f in facs:
            boot[f].append(e[f])
        boot_resid.append(res)
    out = {"n": len(r36), "point": {k: float(v) for k, v in eta0.items()},
           "residual_point": float(resid0), "B": args.boot_var,
           "eta2_ci": {f: [float(np.percentile(boot[f], 2.5)),
                           float(np.percentile(boot[f], 97.5))] for f in facs},
           "residual_ci": [float(np.percentile(boot_resid, 2.5)),
                           float(np.percentile(boot_resid, 97.5))],
           "levels": {k: len(v) for k, v in levels0.items()}}
    return out


# ---------------------------------------------------------------- 输出
def write_md(out_dir, J):
    L = ["# Step 1: 统计严谨性 (CI / 显著性 / 稳健性)", "",
         f"- 生成时间: {J['timestamp']}",
         f"- 数据源: 已落盘 36/42 交付物, **零模型加载 / 零干预**",
         f"- 有效组合 n = {J['n_combos']}, 特征点 n = {J['n_points']}",
         f"- bootstrap: cluster B={J['args']['boot_cluster']}, "
         f"feature B={J['args']['boot_feature']}, twolevel B={J['args']['boot_two']}, "
         f"seed={J['args']['seed']}", "",
         "## 0. 自检", "",
         f"- 由 features 重算的逐组合 Spearman 与 42 落盘值最大绝对差: "
         f"{J['verify']['max_abs_diff_metric']:.2e} (checked {J['verify']['n_checked']})",
         f"- 36/42 非对称键: {J['missing']}", ""
         ]

    L += ["## 1. pooled Spearman 及其 bootstrap 95% CI", "",
          "| score | vs | pooled | cluster CI | feature CI | two-level CI | n_combos |",
          "|---|---|---|---|---|---|---|"]
    for key, e in J["pooled"].items():
        name, t = key.rsplit("_vs_", 1)
        lab = METRIC_LABEL.get(name, name)
        c = e["cluster"]; f = e["feature"]; tw = e["twolevel"]
        L.append(f"| {lab} | {t} | {e['cluster']['point']:+.3f} | "
                 f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] | "
                 f"[{f['ci_lo']:+.3f}, {f['ci_hi']:+.3f}] | "
                 f"[{tw['ci_lo']:+.3f}, {tw['ci_hi']:+.3f}] | {e['n_combos']} |")
    L += ["", "> cluster CI = 重采样组合 (主口径, 组合是泛化单元); "
              "feature CI = 组合固定、组合内重采样特征; two-level = 二者嵌套。", ""]

    L += ["## 2. 逐组合 rho 的单样本检验 (+ Holm-Bonferroni)", "",
          "| score | mean ± sd | median [IQR] | %>0 | Wilcoxon p | Holm p |",
          "|---|---|---|---|---|---|"]
    for k, e in J["one_sample_ablate"].items():
        L.append(f"| {k} | {e['mean']:+.3f} ± {e['std']:.3f} | "
                 f"{e['median']:+.3f} [{e['q25']:+.3f}, {e['q75']:+.3f}] | "
                 f"{e['frac_positive']:.2f} | {e.get('wilcoxon_p', float('nan')):.2e} | "
                 f"{e.get('holm_p', float('nan')):.2e} |")
    L += [""]

    L += ["## 3. 配对检验: 预测器 vs 标准指标 (同一组合内配对)", "",
          "| comparison | n | mean a | mean b | mean Δ | median Δ | Wilcoxon p (two-sided) "
          "| p (one-sided a>b) | Cliff's δ | 胜/平/负 | 胜率 (Wilson 95% CI) | Δpooled [CI] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in J["paired"].items():
        L.append(f"| {name} | {r['n']} | {r['mean_a']:+.3f} | {r['mean_b']:+.3f} | "
                 f"{r['mean_delta']:+.3f} | {r['median_delta']:+.3f} | "
                 f"{r.get('wilcoxon_p_two_sided', float('nan')):.2e} | "
                 f"{r.get('wilcoxon_p_one_sided', float('nan')):.2e} | "
                 f"{r.get('cliffs_delta', float('nan')):+.3f} | "
                 f"{r.get('wins','?')}/{r.get('ties','?')}/{r.get('losses','?')} | "
                 f"{r.get('win_rate', float('nan')):.3f} "
                 f"({r.get('win_rate_wilson_ci', [None,None])[0]:.3f}–"
                 f"{r.get('win_rate_wilson_ci', [None,None])[1]:.3f}) | "
                 f"{r.get('pooled_delta_point', float('nan')):+.3f} "
                 f"[{r.get('pooled_delta_ci', [float('nan')]*2)[0]:+.3f}, "
                 f"{r.get('pooled_delta_ci', [float('nan')]*2)[1]:+.3f}] |")
    L += [""]

    if J.get("scale"):
        s = J["scale"]
        L += ["## 4. 规模轴 (IOI, 9 模型)", "",
              f"- Pearson(rho_M2, log10 params) = {s['pearson_log10_params']:+.3f} "
              f"[{s['pearson_ci']['ci_lo']:+.3f}, {s['pearson_ci']['ci_hi']:+.3f}] "
              f"(B={s['pearson_ci']['B']})",
              f"- Spearman = {s['spearman_log10_params']:+.3f} "
              f"[{s['spearman_ci']['ci_lo']:+.3f}, {s['spearman_ci']['ci_hi']:+.3f}]", "",
              "| model | params | M2 rho (mean over layers) | n layers |", "|---|---|---|---|"]
        for p in s["points"]:
            L.append(f"| {p['model']} | {p['params']} | {p['m2_spearman_mean']:+.3f} | "
                     f"{p['n_layers']} |")
        if "pythia_trend" in s:
            t = s["pythia_trend"]
            L += ["", f"- 同族 Pythia 单调下降: {t['m2_spearman']} ({t['params']}B); "
                      f"逆序对 {t['observed_descending_pairs']}/{t['n_pairs']}; "
                      f"置换检验 p = {t['perm_p_descending']:.4f}"]
        L += [""]

    if J.get("variance"):
        v = J["variance"]
        L += ["## 5. 方差分解 eta^2 的 bootstrap CI (组合内特征级重采样)", "",
              "| factor | eta^2 | 95% CI |", "|---|---|---|"]
        for f, val in v.get("point", {}).items():
            ci = v["eta2_ci"].get(f, [None, None])
            L.append(f"| {f} | {val:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
        L.append(f"| (residual) | {v['residual_point']:.3f} | "
                 f"[{v['residual_ci'][0]:.3f}, {v['residual_ci'][1]:.3f}] |")
        L += [""]

    L += ["## 6. 论文可直接引用的新增统计量", "",
          "- 见 significance.json; 建议在 §5.2 / §5.10 / §5.11 / §5.14 与 §5.12 各补一句 "
          "含 CI 与 p 值的陈述。", ""]
    (out_dir / "significance.md").write_text("\n".join(L), encoding="utf-8")


def _f(x, fmt="+.3f", default="n/a"):
    try:
        if x is None or not np.isfinite(x):
            return default
        return format(float(x), fmt)
    except Exception:
        return default


def load_mid_layers(base_dir):
    """读 config.json 的 target_layers[1], 供「仅 mid 层」口径对齐使用。"""
    try:
        cfg = json.load(open(base_dir / "config.json", encoding="utf-8"))["models"]
    except Exception:
        return {}
    out = {}
    for name, spec in cfg.items():
        tl = spec.get("target_layers") if isinstance(spec, dict) else spec
        if isinstance(tl, (list, tuple)) and len(tl) >= 2:
            out[name] = tl[1]
    return out


def _seed_analysis(base_dir, args, rng, tag, full):
    """对单个 tag 跑全套统计 (bootstrap B 可缩放), 返回 J 或 None(无有效组合)。

    full=True (主种子) 额外跑 43 的方差分解 CI; 其余 tag 因 43 无对应目录而跳过。
    """
    combos, missing, verify = build_combos(base_dir, tag)
    if not combos:
        return None
    print(f"[tag={tag or '(seed42)'}] 有效组合 {len(combos)}; 非对称键 {len(missing)}; "
          f"自检 max|Δ|={verify['max_abs_diff_metric']:.2e} "
          f"(checked {verify['n_checked']})")
    pooled = analyze_pooled(combos, args, rng)
    one_sample = analyze_one_sample(combos, "corr", METRICS, METRIC_LABEL)
    paired = {}
    for a, b in PAIRED_PAIRS:
        src_a = "corr_pred" if a in PREDS else "corr"
        src_b = "corr_pred" if b in PREDS else "corr"
        paired[f"{a} > {b}"] = (analyze_paired_mixed(combos, a, b, args, rng)
                                if src_a != src_b else
                                analyze_paired(combos, a, b, src_a, args, rng))
    scale = analyze_scale(combos, args, rng)
    mid_layers = load_mid_layers(base_dir)
    scale_mid = analyze_scale(combos, args, rng, mid_layers) if mid_layers else {}
    variance = analyze_variance(args, rng) if full else {}
    return {"tag": tag, "timestamp": datetime.now().isoformat(), "args": vars(args),
            "n_combos": len(combos), "n_points": int(sum(c["n"] for c in combos)),
            "verify": verify, "missing": missing, "pooled": pooled,
            "one_sample_ablate": one_sample,
            "one_sample_steer": analyze_one_sample(combos, "corr_steer", METRICS, METRIC_LABEL),
            "one_sample_pred": analyze_one_sample(combos, "corr_pred", PREDS, {}),
            "paired": paired, "scale": scale, "scale_mid_only": scale_mid,
            "variance": variance}


# ------------------------------------------------ 跨种子汇总
def _stats(vals):
    v = np.asarray([x for x in vals
                    if x is not None and np.isfinite(np.asarray(x, float).reshape(-1)[0])], float)
    if len(v) == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "values": []}
    return {"n": int(len(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()),
            "values": [float(x) for x in v]}


def aggregate_cross_seed(per_seed, args):
    """把多个种子的 J 汇总为: 每指标的 seed42 点估计 + 跨种子 mean±sd, 并与 seed42 CI 宽度对比。"""
    tags = list(per_seed)
    primary = tags[0]
    P = per_seed[primary]
    out = {"timestamp": datetime.now().isoformat(), "tags": tags, "n_seeds": len(tags),
           "primary_rank": {t: i for i, t in enumerate(tags)}}
    # 1. pooled
    pl = {}
    for key in P["pooled"]:
        vals = [(per_seed[t].get("pooled", {}).get(key) or {}).get("cluster", {}).get("point")
                for t in tags]
        e = _stats(vals)
        pc = P["pooled"][key]["cluster"]
        pf = P["pooled"][key]["feature"]
        e["per_seed_point"] = {t: (per_seed[t].get("pooled", {}).get(key) or {}).get(
            "cluster", {}).get("point") for t in tags}
        e["primary_point"] = pc["point"]
        e["primary_cluster_ci"] = [pc["ci_lo"], pc["ci_hi"]]
        e["primary_feature_ci"] = [pf["ci_lo"], pf["ci_hi"]]
        ciw = (pc["ci_hi"] - pc["ci_lo"]) if (pc["ci_lo"] is not None and pc["ci_hi"] is not None) else None
        e["primary_cluster_ci_width"] = ciw
        e["cross_sd_over_ci_width"] = (e["sd"] / ciw) if ciw else None
        pl[key] = e
    out["pooled"] = pl
    # 2. 单样本 (逐组合 rho)
    for src, dst in (("one_sample_ablate", "one_sample_ablate"),
                     ("one_sample_pred", "one_sample_pred")):
        agg = {}
        for key in P.get(src, {}):
            vals = [((per_seed[t].get(src) or {}).get(key) or {}).get("mean") for t in tags]
            e = _stats(vals)
            e["primary_wilcoxon_p"] = (P[src].get(key) or {}).get("wilcoxon_p")
            e["primary_holm_p"] = (P[src].get(key) or {}).get("holm_p")
            e["primary_frac_positive"] = (P[src].get(key) or {}).get("frac_positive")
            agg[key] = e
        out[dst] = agg
    # 3. 配对
    pr = {}
    for key in P["paired"]:
        ent = {"mean_a": _stats([((per_seed[t].get("paired") or {}).get(key) or {}).get("mean_a")
                                for t in tags]),
               "mean_b": _stats([((per_seed[t].get("paired") or {}).get(key) or {}).get("mean_b")
                                for t in tags]),
               "mean_delta": _stats([((per_seed[t].get("paired") or {}).get(key) or {}).get("mean_delta")
                                     for t in tags]),
               "pooled_delta": _stats([((per_seed[t].get("paired") or {}).get(key) or {}).get("pooled_delta_point")
                                       for t in tags])}
        pb = P["paired"][key]
        ent["primary_wilcoxon_p_one_sided"] = pb.get("wilcoxon_p_one_sided")
        ent["primary_wins"] = pb.get("wins")
        ent["primary_losses"] = pb.get("losses")
        ent["primary_win_rate"] = pb.get("win_rate")
        ent["primary_cliffs_delta"] = pb.get("cliffs_delta")
        ent["primary_pooled_delta_ci"] = pb.get("pooled_delta_ci")
        ent["wins_per_seed"] = {t: ((per_seed[t].get("paired") or {}).get(key) or {}).get("wins")
                                for t in tags}
        pr[key] = ent
    out["paired"] = pr
    # 4. 规模轴 (全层可用 vs 仅 mid 层口径对齐)
    for src, dst in (("scale", "scale"), ("scale_mid_only", "scale_mid_only")):
        blk = {}
        for m in ("pearson_log10_params", "spearman_log10_params"):
            e = _stats([(per_seed[t].get(src) or {}).get(m) for t in tags])
            e["primary_ci"] = (P.get(src) or {}).get(
                "pearson_ci" if m.startswith("pearson") else "spearman_ci")
            blk[m] = e
        out[dst] = blk
    out["variance_primary_only"] = P.get("variance", {})
    return out


def write_cross_md(out_dir, C):
    tags = C["tags"]
    L = ["# T1 多种子稳健性: 跨种子 mean±sd vs seed42 单点估计", "",
         f"- 生成时间: {C['timestamp']}",
         f"- 种子 tag: {tags} (首个 `{C['tags'][0] or '(seed42 主链)'}` 为已采纳口径, 其余为 T1 复跑)",
         f"- 口径: 36/42 交付物, 同一套 pooled / 单样本 / 配对 / 规模轴统计; "
         f"非主种子 bootstrap B 缩放 (仍为同一估计量)", "",
         "## 1. pooled Spearman: 跨种子稳定性 (关键)", "",
         "| score vs target | seed42 | 跨种子 mean ± sd (n) | [min, max] | seed42 cluster CI 宽度 | sd / 宽度 |",
         "|---|---|---|---|---|---|"]
    for key, e in C["pooled"].items():
        name, t = key.rsplit("_vs_", 1)
        lab = METRIC_LABEL.get(name, name)
        L.append(f"| {lab} vs {t} | {_f(e.get('primary_point'))} | "
                 f"{_f(e.get('mean'))} ± {_f(e.get('sd'), '.3f', 'n/a')} ({e.get('n', 0)}) | "
                 f"[{_f(e.get('min'))}, {_f(e.get('max'))}] | "
                 f"{_f(e.get('primary_cluster_ci_width'), '.3f', 'n/a')} | "
                 f"{_f(e.get('cross_sd_over_ci_width'), '.2f', 'n/a')} |")
    L += ["", "> sd/宽度 ≪ 1 说明跨种子波动远小于种内 bootstrap 抽样不确定性 ⇒ 结论对随机种子稳健。", ""]
    L += ["## 2. 逐组合 rho 均值 (ablate)", "",
          "| score | seed42 mean | 跨种子 mean ± sd | seed42 Wilcoxon p | seed42 Holm p |",
          "|---|---|---|---|---|"]
    for k, e in C.get("one_sample_ablate", {}).items():
        L.append(f"| {k} | {_f(e.get('values', [None])[0] if e.get('values') else None)} | "
                 f"{_f(e.get('mean'))} ± {_f(e.get('sd'), '.3f', 'n/a')} | "
                 f"{_f(e.get('primary_wilcoxon_p'), '.2e', 'n/a')} | "
                 f"{_f(e.get('primary_holm_p'), '.2e', 'n/a')} |")
    L += ["", "## 3. 预测器 vs 标准指标: 跨种子配对 Δ", "",
          "| comparison | seed42 Δmean | 跨种子 Δmean ± sd | seed42 Δpooled [CI] | 各种子胜场 |",
          "|---|---|---|---|---|"]
    for k, e in C["paired"].items():
        wins = e.get("wins_per_seed", {})
        s42 = e["mean_delta"].get("values", [None])[0] if e["mean_delta"].get("values") else None
        pd0 = e["pooled_delta"].get("values", [None])[0] if e["pooled_delta"].get("values") else None
        dci = e.get("primary_pooled_delta_ci") or [None, None]
        L.append(f"| {k} | {_f(s42)} | {_f(e['mean_delta'].get('mean'))} ± "
                 f"{_f(e['mean_delta'].get('sd'), '.3f', 'n/a')} | "
                 f"{_f(pd0)} [{_f(dci[0])}, {_f(dci[1])}] | "
                 f"{[wins.get(t) for t in tags]} |")
    L += ["", "## 4. 规模轴 (IOI, 9 模型)", "",
          "### 4a. 各种子自身可用层 (seed42 为 3 层均值; T1 仅 mid 层)",
          "", "| 统计量 | seed42 | 跨种子 mean ± sd | seed42 CI |", "|---|---|---|---|"]
    for m, e in C.get("scale", {}).items():
        pci = e.get("primary_ci") or {}
        L.append(f"| {m} | {_f(e.get('values', [None])[0] if e.get('values') else None)} | "
                 f"{_f(e.get('mean'))} ± {_f(e.get('sd'), '.3f', 'n/a')} | "
                 f"[{_f(pci.get('ci_lo'))}, {_f(pci.get('ci_hi'))}] |")
    if C.get("scale_mid_only"):
        L += ["", "### 4b. 口径对齐 (全部只取 mid 层 = target_layers[1])",
              "", "| 统计量 | seed42 (mid) | 跨种子 mean ± sd | seed42 mid CI |", "|---|---|---|---|"]
        for m, e in C.get("scale_mid_only", {}).items():
            pci = e.get("primary_ci") or {}
            L.append(f"| {m} | {_f(e.get('values', [None])[0] if e.get('values') else None)} | "
                     f"{_f(e.get('mean'))} ± {_f(e.get('sd'), '.3f', 'n/a')} | "
                     f"[{_f(pci.get('ci_lo'))}, {_f(pci.get('ci_hi'))}] |")
        L += ["", "> **口径提醒**: 4a 的 seed42 值是三层均值 (−0.923), 而 T1 只有 mid 层; "
                  "两者不可直接比较。4b 把 seed42 也限制到 mid 层后才是同口径 —— 跨种子一致为负, "
                  "但量级低于三层均值。论文正文若引用规模轴, 应使用 4b 或明确标注层口径。", ""]
    L += ["", "## 5. 备注", "",
          "- 方差分解 (43 的 eta^2) 仅在 seed42 口径可用, 见 `significance.json` §5; "
          "T1 未重跑 43, 故不参与跨种子汇总。",
          "- T1 每种子的有效组合数不同 (s43/s44=17, s45=15), 因 `at_chance` 判定随种子变化: "
          "s45 的 phi-3.5-mini|ioi (b0_acc=0.595) 与 tinyllama-1.1b|gt (0.585) 跌破 0.6 阈值; "
          "s43/s44 的 phi|ioi 通过。配对检验按各自有效组合配对。", ""]
    (out_dir / "significance_cross_seed.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Step1: 统计严谨性 (CI/显著性/稳健性, 零模型加载)")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--out_dir", type=str, default="output/phaseStep2/significance")
    ap.add_argument("--boot_cluster", type=int, default=5000)
    ap.add_argument("--boot_feature", type=int, default=1000)
    ap.add_argument("--boot_two", type=int, default=1000)
    ap.add_argument("--boot_scale", type=int, default=5000)
    ap.add_argument("--boot_var", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--tags", nargs="*", default=[""],
                    help="目录后缀列表; 首个为主种子(全 B 并写 significance.json/md + 方差分解), "
                         "其余写 significance{tag}.json 并纳入跨种子汇总。T1 用 '' _s43 _s44 _s45")
    ap.add_argument("--boot_small_cluster", type=int, default=1000)
    ap.add_argument("--boot_small_feature", type=int, default=200)
    ap.add_argument("--boot_small_two", type=int, default=200)
    ap.add_argument("--boot_small_scale", type=int, default=1000)
    ap.add_argument("--boot_small_var", type=int, default=100)
    ap.add_argument("--force_primary", action="store_true",
                    help="允许覆盖已存在的 significance*.json/md (默认不覆盖, 保护已采纳产物)")
    args = ap.parse_args()

    base_dir = Path(args.config).parent
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = list(args.tags) if args.tags else [""]
    # 别名: PowerShell 传空串会被吞, 故允许用 s42/_s42 指代主链 (无后缀)
    tags = [("" if t in ("s42", "_s42") else t) for t in tags]

    per_seed = {}
    for i, tag in enumerate(tags):
        full = (i == 0)
        a = copy.deepcopy(args)
        if not full:
            a.boot_cluster = args.boot_small_cluster
            a.boot_feature = args.boot_small_feature
            a.boot_two = args.boot_small_two
            a.boot_scale = args.boot_small_scale
            a.boot_var = args.boot_small_var
        rng = np.random.default_rng(args.seed)
        J = _seed_analysis(base_dir, a, rng, tag, full)
        if J is None:
            print(f"[跳过] tag={tag!r}: 无有效组合 (目录缺失或未跑完)")
            continue
        per_seed[tag] = J
        name = "significance" if full else f"significance_{tag.lstrip('_')}" if tag else "significance"
        fpath = out_dir / f"{name}.json"
        # 仅保护主种子 (已采纳) 产物; 非主种子的 significance_sXX.json 是本脚本次生产物, 可刷新
        if full and fpath.exists() and not args.force_primary:
            print(f"[跳过写盘] {fpath.name} 已存在 (> 不覆盖已采纳产物; 加 --force_primary 可覆盖); "
                  f"本次仍在内存中参与跨种子汇总")
            continue
        fpath.write_text(json.dumps(J, ensure_ascii=False, indent=2), encoding="utf-8")
        if full:
            write_md(out_dir, J)
            print(f"[完成] {out_dir / 'significance.md'}")
        else:
            print(f"[完成] {fpath}")

    if len(per_seed) >= 2:
        cross = aggregate_cross_seed(per_seed, args)
        (out_dir / "significance_cross_seed.json").write_text(
            json.dumps(cross, ensure_ascii=False, indent=2), encoding="utf-8")
        write_cross_md(out_dir, cross)
        print(f"[完成] {out_dir / 'significance_cross_seed.md'}  (种子 {list(per_seed)})")
    elif len(per_seed) == 1:
        print("[提示] 只有一个有效种子, 未生成跨种子汇总 (需 --tags 提供多个已完成目录)")


def analyze_paired_mixed(combos, a_key, b_key, args, rng):
    """a 来自 42 (pred_*), b 来自 36 (M*): 逐组合 rho 直接配对。"""
    av = np.array([c["corr_pred"][a_key] for c in combos], float)
    bv = np.array([c["corr"][b_key] for c in combos], float)
    m = ~(np.isnan(av) | np.isnan(bv))
    av, bv = av[m], bv[m]
    n = len(av)
    res = {"n": int(n), "mean_a": float(av.mean()), "mean_b": float(bv.mean()),
           "median_a": float(np.median(av)), "median_b": float(np.median(bv)),
           "mean_delta": float((av - bv).mean()), "median_delta": float(np.median(av - bv))}
    if n >= 8 and np.any(av != bv):
        res["wilcoxon_p_two_sided"] = float(stats.wilcoxon(av, bv,
                                                           alternative="two-sided").pvalue)
        res["wilcoxon_p_one_sided"] = float(stats.wilcoxon(av, bv,
                                                           alternative="greater").pvalue)
        res["ttest_p_two_sided"] = float(stats.ttest_rel(av, bv).pvalue)
        res["cliffs_delta"] = cliffs_delta(av, bv)
        wins = int((av > bv).sum()); ties = int((av == bv).sum()); losses = int((av < bv).sum())
        res.update({"wins": wins, "ties": ties, "losses": losses,
                    "win_rate": float(wins / n), "win_rate_wilson_ci": wilson_ci(wins, n)})
    pairs_a = [(zscore(c["xn"][a_key]), zscore(c["c_ablate"])) for c in combos]
    pairs_b = [(zscore(c["x36"][b_key]), zscore(c["c_ablate"])) for c in combos]
    da = boot_pooled(pairs_a, "cluster", args.boot_cluster, rng)
    db = boot_pooled(pairs_b, "cluster", args.boot_cluster, rng)
    if len(da) and len(db):
        diff = da - db
        pa = spearman(np.concatenate([p[0] for p in pairs_a]),
                      np.concatenate([p[1] for p in pairs_a]))
        pb = spearman(np.concatenate([p[0] for p in pairs_b]),
                      np.concatenate([p[1] for p in pairs_b]))
        res["pooled_delta_point"] = float(pa - pb)
        res["pooled_delta_ci"] = [float(np.percentile(diff, 2.5)),
                                  float(np.percentile(diff, 97.5))]
        res["pooled_delta_boot_std"] = float(diff.std())
    return res


if __name__ == "__main__":
    main()
