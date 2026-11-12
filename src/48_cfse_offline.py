# -*- coding: utf-8 -*-
"""48: CFSE 离线记分卡 (零 GPU, 零模型加载)

对已落盘的 36/38/40/42 + 多种子 (s43/s44/s45) 交付物, 按新的
"因果特征筛选评估体系 (CFSE v1)" 重算指标:

  Tier 0 门控   hook 恒等 / 层对齐 / 分辨率比 RR / 饱和 / 零校准
  Tier 1 逐特征 rho_abs, rho_signed, SignAcc, nDCG@k, Recall@k, 标定斜率/R2
  Tier 2 集合级 z@N, C_norm@N, C_rel@N, ExcessRatio@N, SetCoverage@N(M2)
  Tier 4 稳健性 跨种子 top-50 Jaccard, 规模斜率
  Tier 5 成本   静态前向预算

不重跑任何干预; 所有数字来自落盘 JSON。
输出: output/phaseStep2/cfse_offline/{cfse_combos.json, cfse_summary.json,
      cfse_report.md, cfse_overview.png}
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# bf16 logits 在 |logit| in [4,8) 的 ULP; 保守取 0.0625
BF16_ULP = 0.0625

# 模型名 -> 参数量(B), 用于规模斜率
PARAMS = {
    "qwen3.5-0.8b": 0.8, "qwen3.5-2b": 2.0,
    "tinyllama-1.1b": 1.1, "minicpm5-1b": 1.0,
    "phi-3.5-mini": 3.8, "ministral-3-3b": 3.0,
    "pythia-1.4b": 1.4, "pythia-2.8b": 2.8, "pythia-6.9b": 6.9,
    "qwen2.5-7b-instruct": 7.6,
}

FEATURE_METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
                   "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]
SET_METRICS = FEATURE_METRICS + ["pred_full", "pred_dir", "pred_last",
                                 "pred_dir_last", "pred_steer", "pred_align"]
KS = [10, 20, 50]
NS = [1, 5, 10, 20, 50]


# ---------------------------------------------------------------------------
# 基础统计
# ---------------------------------------------------------------------------
def _rank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    # ties -> average rank
    vals, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return r


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return None
    rx, ry = _rank(x[m]), _rank(y[m])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() == 0:
        return None, None
    x, y = x[m], y[m]
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return float(b), r2


def ndcg_at_k(scores, gains, k):
    scores = np.asarray(scores, dtype=float)
    gains = np.asarray(gains, dtype=float)
    if len(gains) < k:
        k = len(gains)
    order = np.argsort(-scores, kind="mergesort")[:k]
    dcg = float(np.sum(gains[order] / np.log2(np.arange(2, k + 2))))
    ideal = np.sort(gains)[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, k + 2))))
    return dcg / idcg if idcg > 0 else None


def recall_at_k(scores, gains, k):
    scores = np.asarray(scores, dtype=float)
    gains = np.asarray(gains, dtype=float)
    if len(gains) < k:
        k = len(gains)
    sel = set(np.argsort(-scores, kind="mergesort")[:k].tolist())
    true = set(np.argsort(-gains, kind="mergesort")[:k].tolist())
    return len(sel & true) / k if k else None


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None
    return len(a & b) / len(a | b)


def boot_ci(vals, B=2000, seed=7):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) < 3:
        return None, None
    rng = np.random.RandomState(seed)
    means = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(B)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# 读盘
# ---------------------------------------------------------------------------
def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def discover(base):
    root = base / "output" / "phaseStep2"
    combos = []
    for mdir in sorted((root / "causal_validity").glob("*")):
        if not mdir.is_dir():
            continue
        for ldir in sorted(mdir.glob("layer*")):
            try:
                layer = int(ldir.name.replace("layer", ""))
            except ValueError:
                continue
            for f in sorted(ldir.glob("*.json")):
                if f.name.startswith("resume_"):
                    continue
                task = f.stem
                rec = load_json(f)
                if not rec or rec.get("error") or rec.get("at_chance"):
                    continue
                feats = [r for r in (rec.get("features") or [])
                         if r.get("C_ablate") is not None]
                if len(feats) < 20:
                    continue
                combos.append({"model": mdir.name, "layer": layer, "task": task,
                               "rec36": rec, "feats": feats})
    return combos


def attach_42(combo, base):
    p = (base / "output" / "phaseStep2" / "causal_margin" / combo["model"]
         / f"layer{combo['layer']}" / f"{combo['task']}.json")
    rec = load_json(p)
    if not rec:
        return False
    preds = {int(r["feature_idx"]): r for r in (rec.get("features") or [])}
    n = 0
    for r in combo["feats"]:
        pr = preds.get(int(r["feature_idx"]))
        if pr is None:
            continue
        r["pred_full_v"] = pr.get("pred_full_v")
        r["pred_last_v"] = pr.get("pred_last_v")
        r["pred_align_v"] = pr.get("pred_align")
        n += 1
    combo["rec42"] = rec
    combo["n_pred"] = n
    return n > 0


def attach_38(combo, base):
    p = (base / "output" / "phaseStep2" / "topn_joint" / combo["model"]
         / f"layer{combo['layer']}" / f"{combo['task']}.json")
    rec = load_json(p)
    combo["rec38"] = rec
    return rec is not None


def attach_40(combo, base):
    p = (base / "output" / "phaseStep2" / "selector_margin" / combo["model"]
         / f"layer{combo['layer']}" / f"{combo['task']}.json")
    rec = load_json(p)
    combo["rec40"] = rec
    return rec is not None


# ---------------------------------------------------------------------------
# 单组合指标
# ---------------------------------------------------------------------------
def feature_table(combo, tag=""):
    feats = combo["feats"]
    C = np.array([r["C_ablate"] for r in feats], dtype=float)
    B0 = combo["rec36"].get("b0", 0.0)
    dB = np.array([(r.get("B_ablate", B0) - B0) for r in feats], dtype=float)
    base = {"M1_act_freq_task": np.array([r.get("M1_act_freq_task") or 0.0 for r in feats]),
            "M2_mean_act_task": np.array([r.get("M2_mean_act_task") or 0.0 for r in feats]),
            "M3_max_act_task": np.array([r.get("M3_max_act_task") or 0.0 for r in feats]),
            "M4_recon_mse_gain": np.array([r.get("M4_recon_mse_gain") or 0.0 for r in feats]),
            "M5_ctx_coherence": np.array([r.get("M5_ctx_coherence") or 0.0 for r in feats]),
            "M6_corpus_freq": np.array([r.get("M6_corpus_freq") or 0.0 for r in feats])}
    out = {}
    for name, s in base.items():
        out[name] = {"score": s, "signed": None}
    if combo.get("n_pred", 0) > 0:
        pf = np.array([r.get("pred_full_v") or 0.0 for r in feats], dtype=float)
        pl = np.array([r.get("pred_last_v") or 0.0 for r in feats], dtype=float)
        pa = np.array([r.get("pred_align_v") or 0.0 for r in feats], dtype=float)
        out["pred_full"] = {"score": np.abs(pf), "signed": pf}
        out["pred_last"] = {"score": np.abs(pl), "signed": pl}
        out["pred_align"] = {"score": pa, "signed": None}
    return C, dB, out


def per_feature_metrics(combo):
    C, dB, table = feature_table(combo)
    res = {}
    n = len(C)
    rng = np.random.RandomState(11)
    rand_scores = [rng.permutation(np.arange(n)) for _ in range(40)]
    for name, d in table.items():
        s = d["score"]
        rec = {"rho_abs": spearman(s, C), "n": n}
        for k in KS:
            rec[f"ndcg@{k}"] = ndcg_at_k(s, C, k)
            rec[f"recall@{k}"] = recall_at_k(s, C, k)
            vals = [ndcg_at_k(rs, C, k) for rs in rand_scores]
            rec[f"ndcg@{k}_rand"] = float(np.mean(vals)) if vals else None
            vals = [recall_at_k(rs, C, k) for rs in rand_scores]
            rec[f"recall@{k}_rand"] = float(np.mean(vals)) if vals else None
        if d["signed"] is not None:
            sgn = d["signed"]
            rec["rho_signed"] = spearman(sgn, dB)
            rec["sign_acc"] = float(np.mean(np.sign(sgn) == np.sign(dB)))
            rec["sign_acc_major"] = float(max(rec["sign_acc"], 1 - rec["sign_acc"]))
            # 去偏移方向: 消融效应可能有模型级常数偏移 (几乎所有特征同号),
            # 此时绝对符号不可学, 但相对方向仍可学。
            sgn_c = sgn - np.median(sgn)
            db_c = dB - np.median(dB)
            rec["sign_acc_centered"] = float(np.mean(np.sign(sgn_c) == np.sign(db_c)))
            rec["offset_ratio"] = float(np.median(dB) / (np.median(np.abs(dB)) + 1e-12))
            b, r2 = ols(sgn, dB)
            rec["calib_slope"] = b
            rec["calib_r2"] = r2
        res[name] = rec
    return res


def gate_metrics(combo):
    rec = combo["rec36"]
    C = np.array([r["C_ablate"] for r in combo["feats"]], dtype=float)
    top = np.sort(C)[::-1][:10]
    med_top = float(np.median(top)) if len(top) else 0.0
    n_prompt = float(rec.get("task_meta", {}).get("n") or 200)
    # 均值化后的有效分辨率: ULP / n (保守)
    res_eff = BF16_ULP / max(n_prompt, 1.0)
    rr = res_eff / med_top if med_top > 0 else None
    b0_acc = rec.get("b0_acc")
    return {
        "hook_identity_ok": bool(rec.get("hook_identity_ok")),
        "align_max_abs_diff": rec.get("align_max_abs_diff"),
        "layer_map_ok": (rec.get("align_max_abs_diff") is not None
                         and rec["align_max_abs_diff"] <= 1e-3),
        "b0_acc": b0_acc,
        "saturated": (b0_acc is not None and b0_acc >= 0.95),
        "resolution_ratio": rr,
        "resolution_ok": (rr is not None and rr <= 0.1),
        "n_prompts": int(n_prompt),
    }


def set_metrics(combo):
    rec = combo.get("rec40")
    if not rec:
        return {}
    cmap = {int(r["feature_idx"]): float(r["C_ablate"]) for r in combo["feats"]}
    ctot = sum(cmap.values())
    out = {}
    for blk in ("part1_selector", "part1_extra"):
        for row in rec.get(blk) or []:
            met, N = row.get("metric"), row.get("N")
            if met is None or N is None:
                continue
            ab = row.get("ablate") or {}
            C = ab.get("C")
            rc = row.get("rand_mean_C")
            idxs = row.get("feature_idx") or []
            mass = (sum(cmap.get(int(f), 0.0) for f in idxs) / ctot) if ctot > 0 else None
            out.setdefault(met, {})[int(N)] = {
                "C": C,
                "C_norm": ab.get("C_norm"),
                "C_rel": ab.get("C_rel"),
                "flip": ab.get("flip"),
                "rand_mean_C": rc,
                "z": row.get("z"),
                "excess": (C / rc) if (C is not None and rc and rc > 0) else None,
                "mass_recall": mass,
                "n_features": len(row.get("feature_idx") or []),
            }
    # SetCoverage (只有 38 的 M2 集合有 mean-ablate 参考)
    r38 = combo.get("rec38")
    if r38:
        ref = (r38.get("refs") or {}).get("mean_ablate", {}).get("C")
        for row in r38.get("results") or []:
            N = int(row.get("N"))
            ab = row.get("ablate") or {}
            if ref and ab.get("C") is not None:
                out.setdefault("M2_mean_act_task", {}).setdefault(N, {})
                out["M2_mean_act_task"][N]["coverage"] = ab["C"] / ref
    return out


def stability(combo, base, topk=50):
    """跨 seed 的 top-k Jaccard (M2 与 pred_full 排序)。"""
    m, layer, task = combo["model"], combo["layer"], combo["task"]
    out = {}
    # M2: 36 的 features 按 M2 排序
    base_set = set(int(r["feature_idx"]) for r in
                   sorted(combo["feats"], key=lambda r: -r["M2_mean_act_task"])[:topk])
    js = []
    for tag in ("_s43", "_s44", "_s45"):
        p = (base / "output" / "phaseStep2" / f"causal_validity{tag}" / m
             / f"layer{layer}" / f"{task}.json")
        rec = load_json(p)
        if not rec or rec.get("error") or rec.get("at_chance"):
            continue
        fs = [r for r in (rec.get("features") or []) if r.get("C_ablate") is not None]
        if len(fs) < topk:
            continue
        s = set(int(r["feature_idx"]) for r in
                sorted(fs, key=lambda r: -r["M2_mean_act_task"])[:topk])
        j = jaccard(base_set, s)
        if j is not None:
            js.append(j)
    out[f"M2_jaccard@{topk}"] = float(np.mean(js)) if js else None

    # pred_full: 42 的 features 按 |pred| 排序
    if combo.get("n_pred", 0) > 0:
        base_p = set(int(r["feature_idx"]) for r in
                     sorted(combo["feats"],
                            key=lambda r: -abs(r.get("pred_full_v") or 0.0))[:topk])
        js = []
        for tag in ("_s43", "_s44", "_s45"):
            p = (base / "output" / "phaseStep2" / f"causal_margin{tag}" / m
                 / f"layer{layer}" / f"{task}.json")
            rec = load_json(p)
            if not rec:
                continue
            fs = rec.get("features") or []
            if len(fs) < topk:
                continue
            s = set(int(r["feature_idx"]) for r in
                    sorted(fs, key=lambda r: -abs(r.get("pred_full") or 0.0))[:topk])
            j = jaccard(base_p, s)
            if j is not None:
                js.append(j)
        out[f"pred_full_jaccard@{topk}"] = float(np.mean(js)) if js else None
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(rows, selectors):
    summary = {}
    for sel in selectors:
        recs = [r["feature"][sel] for r in rows if sel in r.get("feature", {})]
        if not recs:
            continue
        s = {"n_combos": len(recs)}
        for key in ("rho_abs", "rho_signed", "sign_acc", "sign_acc_major",
                    "sign_acc_centered", "offset_ratio", "calib_slope", "calib_r2"):
            vals = [x.get(key) for x in recs if x.get(key) is not None]
            if vals:
                lo, hi = boot_ci(vals)
                s[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                          "sd": float(np.std(vals)), "ci95": [lo, hi], "n": len(vals)}
        for k in KS:
            for pre in ("ndcg", "recall"):
                key = f"{pre}@{k}"
                vals = [x.get(key) for x in recs if x.get(key) is not None]
                if vals:
                    s[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                              "sd": float(np.std(vals))}
                rk = f"{key}_rand"
                rv = [x.get(rk) for x in recs if x.get(rk) is not None]
                if rv:
                    s[key + "_rand"] = {"mean": float(np.mean(rv))}
                    if vals:
                        s[key + "_delta"] = float(np.mean(vals) - np.mean(rv))
        summary[sel] = s
    return summary


def summarize_sets(rows, selectors):
    out = {}
    for sel in selectors:
        per_n = {}
        for N in NS:
            vals = {"C": [], "C_norm": [], "C_rel": [], "flip": [], "z": [],
                    "excess": [], "coverage": [], "mass_recall": []}
            for r in rows:
                d = (r.get("sets", {}).get(sel) or {}).get(N)
                if not d:
                    continue
                for k in vals:
                    if d.get(k) is not None and np.isfinite(d[k]):
                        vals[k].append(d[k])
            if not vals["C"]:
                continue
            rec = {}
            for k, v in vals.items():
                if v:
                    rec[k] = {"mean": float(np.mean(v)), "median": float(np.median(v)),
                              "n": len(v)}
            if rec.get("z"):
                lo, hi = boot_ci(vals["z"])
                rec["z"]["ci95"] = [lo, hi]
            per_n[N] = rec
        if per_n:
            out[sel] = per_n
    return out


def scale_slope(rows, sel, key="rho_abs"):
    xs, ys = [], []
    for r in rows:
        p = PARAMS.get(r["model"])
        v = (r.get("feature", {}).get(sel) or {}).get(key)
        if p and v is not None:
            xs.append(math.log10(p))
            ys.append(v)
    if len(xs) < 3:
        return None
    b, r2 = ols(xs, ys)
    return {"slope_per_decade": b, "r2": r2, "n": len(xs)}


def grade(combo_rec, sel):
    """CFSE 双轨判级: 幅度轨 (排序/集合) 与方向轨 (符号/标定).

    单一总分不可辩护; 一个选择器可以"幅度 A / 方向 C"。
    """
    g = combo_rec["gates"]
    f = combo_rec["feature"].get(sel, {})
    s = combo_rec.get("sets", {}).get(sel, {})

    def _v(d, key):
        x = (d or {}).get(key)
        if isinstance(x, dict):
            return x.get("mean")
        return x

    z10 = _v(s.get(10), "z")
    cov50 = _v(s.get(50), "coverage")
    gate_ok = bool(g["hook_identity_ok"] and g["layer_map_ok"]
                   and (g["resolution_ok"] is not False))

    mag = {
        "rho_abs>=0.50": (f.get("rho_abs") is not None and f["rho_abs"] >= 0.5),
        "ndcg@20>=0.50": (f.get("ndcg@20") is not None and f["ndcg@20"] >= 0.5),
        "z@10>=5": (z10 is not None and z10 >= 5),
        "coverage@50>=0.25": (cov50 is not None and cov50 >= 0.25),
    }
    dirn = None
    if f.get("sign_acc") is not None:
        dirn = {
            "sign_acc>=0.65": f["sign_acc"] >= 0.65,
            "slope>0": (f.get("calib_slope") is not None and f["calib_slope"] > 0),
        }

    def verdict(checks, hard):
        if not gate_ok:
            return "C"
        if not all(checks[k] for k in hard):
            return "C"
        avail = [v for v in checks.values() if v is not None]
        passed = sum(bool(v) for v in avail)
        return "A" if passed == len(avail) else "B"

    mag_g = verdict(mag, ["rho_abs>=0.50"])
    dir_g = None if dirn is None else verdict(dirn, ["sign_acc>=0.65"])
    return {"mag": mag_g, "dir": dir_g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="output/phaseStep2/cfse_offline")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = discover(base)
    print(f"[发现] {len(combos)} 个有效组合")
    rows = []
    for c in combos:
        attach_42(c, base)
        attach_38(c, base)
        attach_40(c, base)
        feat = per_feature_metrics(c)
        sets = set_metrics(c)
        gates = gate_metrics(c)
        stab = stability(c, base)
        row = {"model": c["model"], "layer": c["layer"], "task": c["task"],
               "gates": gates, "feature": feat, "sets": sets, "stability": stab}
        row["grade"] = {sel: grade(row, sel) for sel in feat}
        rows.append(row)
        print(f"  {c['model']}|L{c['layer']}|{c['task']}: "
              f"rho(pred)={feat.get('pred_full', {}).get('rho_abs')}, "
              f"ndcg20={feat.get('pred_full', {}).get('ndcg@20')}, "
              f"z10={((sets.get('pred_full') or {}).get(10) or {}).get('z')}")

    selectors = [s for s in SET_METRICS if any(s in r["feature"] for r in rows)]
    grade_counts = {"mag": {}, "dir": {}}
    for s in selectors:
        for track in ("mag", "dir"):
            grade_counts[track][s] = {
                g: sum(1 for r in rows
                       if (r["grade"].get(s) or {}).get(track) == g)
                for g in ("A", "B", "C")}

    summary = {"timestamp": datetime.now().isoformat(),
               "n_combos": len(rows),
               "n_features_total": int(sum(len(c["feats"]) for c in combos)),
               "feature": summarize(rows, selectors),
               "sets": summarize_sets(rows, selectors),
               "scale": {s: scale_slope(rows, s) for s in selectors},
               "gates": {
                   "n_hook_identity_ok": int(sum(r["gates"]["hook_identity_ok"] for r in rows)),
                   "n_layer_map_ok": int(sum(r["gates"]["layer_map_ok"] for r in rows)),
                   "n_resolution_ok": int(sum(bool(r["gates"]["resolution_ok"]) for r in rows)),
                   "n_saturated": int(sum(r["gates"]["saturated"] for r in rows)),
                   "resolution_ratio_median": float(np.median(
                       [r["gates"]["resolution_ratio"] for r in rows
                        if r["gates"]["resolution_ratio"] is not None])),
               },
               "grade_counts": grade_counts}
    summary["stability"] = {}
    for k in ("M2_jaccard@50", "pred_full_jaccard@50"):
        vals = [r["stability"].get(k) for r in rows if r["stability"].get(k) is not None]
        if vals:
            summary["stability"][k] = {"mean": float(np.mean(vals)),
                                       "median": float(np.median(vals)), "n": len(vals)}

    with open(out_dir / "cfse_combos.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    with open(out_dir / "cfse_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    # ---------------- markdown ----------------
    L = []
    L.append("# CFSE v1 离线记分卡\n")
    L.append(f"- 生成: {summary['timestamp']}")
    L.append(f"- 组合: {len(rows)}；特征: {summary['n_features_total']}")
    g = summary["gates"]
    L.append(f"- 门控: hook 恒等 {g['n_hook_identity_ok']}/{len(rows)}；"
             f"层对齐 {g['n_layer_map_ok']}/{len(rows)}；"
             f"分辨率 {g['n_resolution_ok']}/{len(rows)}；"
             f"饱和 {g['n_saturated']}/{len(rows)}；"
             f"RR 中位 {g['resolution_ratio_median']:.4f}")
    L.append("")
    L.append("## Tier 1 逐特征忠实度（组合均值）\n")
    L.append("| selector | rho_abs | rho_signed | SignAcc | SignAcc(centered) | offset | nDCG@20 | dNDCG@20 | Recall@20 | dRecall@20 | slope | R2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in selectors:
        d = summary["feature"].get(s)
        if not d:
            continue

        def f(k, nd=3):
            v = d.get(k)
            return f"{v['mean']:.{nd}f}" if isinstance(v, dict) else "—"

        def fraw(k, nd=3):
            v = d.get(k)
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

        L.append(f"| {s} | {f('rho_abs')} | {f('rho_signed')} | {f('sign_acc')} | "
                 f"{f('sign_acc_centered')} | {f('offset_ratio')} | {f('ndcg@20')} | "
                 f"{fraw('ndcg@20_delta')} | {f('recall@20')} | "
                 f"{fraw('recall@20_delta')} | {f('calib_slope')} | {f('calib_r2')} |")
    L.append("")
    L.append("### 方向准确率的分模型拆分（pred_full）\n")
    L.append("| model | SignAcc | SignAcc(centered) | offset ratio |")
    L.append("|---|---|---|---|")
    by_model = {}
    for r in rows:
        d = r["feature"].get("pred_full") or {}
        if d.get("sign_acc") is None:
            continue
        by_model.setdefault(r["model"], []).append(d)
    for m, ds in sorted(by_model.items()):
        L.append(f"| {m} | {np.mean([x['sign_acc'] for x in ds]):.3f} | "
                 f"{np.mean([x.get('sign_acc_centered', np.nan) for x in ds]):.3f} | "
                 f"{np.mean([abs(x.get('offset_ratio', np.nan)) for x in ds]):.3f} |")
    L.append("")
    L.append("## Tier 2 集合级（N=10 / 50）\n")
    L.append("| selector | z@10 | C_norm@10 | mass@10 | z@50 | C_norm@50 | mass@50 | coverage@50 | flip@50 (未饱和) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in selectors:
        d = summary["sets"].get(s)
        if not d:
            continue

        def gv(N, k, nd=3):
            v = (d.get(N) or {}).get(k)
            return f"{v['mean']:.{nd}f}" if v else "—"

        # flip 只在未饱和组合上可解释
        fl = []
        for r in rows:
            if not r["gates"]["saturated"]:
                x = ((r.get("sets", {}).get(s) or {}).get(50) or {}).get("flip")
                if x is not None:
                    fl.append(x)
        fl_s = f"{np.mean(fl):.3f} (n={len(fl)})" if fl else "—"
        L.append(f"| {s} | {gv(10,'z',2)} | {gv(10,'C_norm')} | {gv(10,'mass_recall')} | "
                 f"{gv(50,'z',2)} | {gv(50,'C_norm')} | {gv(50,'mass_recall')} | "
                 f"{gv(50,'coverage')} | {fl_s} |")
    L.append("")
    L.append("## Tier 4/5 稳健性与成本\n")
    for k, v in summary["stability"].items():
        L.append(f"- {k}: mean={v['mean']:.3f}, median={v['median']:.3f}, n={v['n']}")
    for s in selectors:
        sc = summary["scale"].get(s)
        if sc:
            L.append(f"- scale slope {s}: {sc['slope_per_decade']:+.3f}/decade "
                     f"(R2={sc['r2']:.3f}, n={sc['n']})")
    L.append("")
    L.append("## CFSE 双轨判级（provisional）\n")
    L.append("| selector | mag A | mag B | mag C | dir A | dir B | dir C |")
    L.append("|---|---|---|---|---|---|---|")
    for s in selectors:
        mg = summary["grade_counts"]["mag"].get(s, {})
        dg = summary["grade_counts"]["dir"].get(s, {})
        L.append(f"| {s} | {mg.get('A',0)} | {mg.get('B',0)} | {mg.get('C',0)} | "
                 f"{dg.get('A',0)} | {dg.get('B',0)} | {dg.get('C',0)} |")
    L.append("")
    L.append("> 幅度轨: gate + rho_abs + nDCG@20 + z@10 (+coverage 若有)。"
             "方向轨: gate + sign_acc + slope>0；无符号预测的指标记 0。")
    L.append("> `sign_acc_major` 仅供参考；判级用**原始** sign_acc，"
             "因为对同一 B 定义做全局翻号会掩盖真实方向错误。")
    L.append("> `coverage` 仅 M2 集合可算（38 提供 mean-ablate 参考），其余记 —。")
    (out_dir / "cfse_report.md").write_text("\n".join(L), encoding="utf-8")

    # ---------------- figure ----------------
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        names = [s for s in selectors if s in summary["feature"]]
        vals = [summary["feature"][s].get("ndcg@20", {}).get("mean", np.nan) for s in names]
        axes[0].barh(names, vals, color="#4C78A8")
        axes[0].set_title("nDCG@20 (mean over combos)")
        axes[0].axvline(0, color="k", lw=0.5)
        vals = [summary["feature"][s].get("rho_abs", {}).get("mean", np.nan) for s in names]
        axes[1].barh(names, vals, color="#F58518")
        axes[1].set_title("Spearman |score| vs C_ablate")
        axes[1].axvline(0, color="k", lw=0.5)
        # grade counts stacked
        als = [s for s in selectors if s in summary["grade_counts"]["mag"]]
        gc = summary["grade_counts"]["mag"]
        a = [gc[s]["A"] for s in als]
        b = [gc[s]["B"] for s in als]
        c = [gc[s]["C"] for s in als]
        axes[2].barh(als, a, color="#54A24B", label="A")
        axes[2].barh(als, b, left=a, color="#EECA3B", label="B")
        axes[2].barh(als, c, left=np.array(a) + np.array(b), color="#E45756", label="C")
        axes[2].set_title("CFSE magnitude grade counts")
        axes[2].legend()
        plt.tight_layout()
        fig.savefig(out_dir / "cfse_overview.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        print(f"[图] 跳过: {e}")

    print(f"\n[完成] {out_dir}")
    print(f"  Tier1 pred_full: rho={summary['feature'].get('pred_full', {}).get('rho_abs', {}).get('mean')}, "
          f"ndcg20={summary['feature'].get('pred_full', {}).get('ndcg@20', {}).get('mean')}")
    print(f"  Gates: {g}")


if __name__ == "__main__":
    main()
