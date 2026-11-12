# -*- coding: utf-8 -*-
"""E7 — 统计单元重算（零机时，全部离线）

回应四条审稿意见：

  C07  (组合, split) 对被当作独立样本做符号检验 -> 伪重复
       -> 改为组合级检验：每组合先对两 split 求均值，再对组合做符号检验/Wilcoxon
       -> 同时给出 cluster bootstrap 的 p 值（以组合为重采样单元）
       -> 输出「聚类前 vs 聚类后」对照表

  C08  |dB| 折叠 + Spearman/sqrt(Spearman) 尺度混用 + 假定 rel_X=1
       -> 同时报告 rel(signed) 与 rel(folded)
       -> 计算 rel(pred) 与 rel(M2)（在各自半样本内部再分半）
       -> 改用 disattenuated: rho / sqrt(rel_X * rel_Y)
       -> 保留原 ratio 但改名并给限定

  C09  闸门阈值在同一批点上选、又用同一批点报 20/20 与 46/48
       -> LOCO（留一组合）交叉验证：用其余点重选阈值，预测被留出者
       -> 给出固定预算版本（rel_pred 分位点）的一致率
       -> 输出阈值 / 预测值 / 观测值 三层语义的统一表

  C19  pooled n 三处写成 9,600 / 10,200 / 12,000
       -> 统一为 9,600（48 组合）与 10,200（60 组合）

输出:
  output/phaseStep2/cfse_e7_stats/e7_stats.json
  output/phaseStep2/cfse_e7_stats/e7_report.md
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
GRID_DIR = BASE / "cfse_grid_heldout"
GATE_DIR = BASE / "cfse_gate"
OUT = BASE / "cfse_e7_stats"

N_SPLITS = 20
SEED0 = 4242
KS = (5, 10, 20, 50, 100)

# 闸门阈值（来自 cfse_gate/gate_report.json 的现行设定）
TH_REL_PRED = 0.25      # Level 1: 真值不可测
TH_KAPPA_REJECT = 0.10  # Level 2: reject
TH_KAPPA_ADMIT = 0.30   # Level 2: admit


# ---------------------------------------------------------------- 统计工具
def _rank(a: np.ndarray) -> np.ndarray:
    """平均秩（ties 取均值），与 scipy 的 rankdata 一致。"""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sa = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（ties 取平均秩），与项目内 34/48 号脚本口径一致。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if den <= 0:
        return float("nan")
    return float((ra * rb).sum() / den)


def ndcg_at_k(score: np.ndarray, gain: np.ndarray, k: int) -> float:
    """以 gain 为相关度、score 排序的 nDCG@k。"""
    score = np.asarray(score, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    k = min(k, score.size)
    idx = np.argsort(-score, kind="mergesort")[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gain[idx] * disc).sum())
    ideal = np.sort(gain)[::-1][:k]
    idcg = float((ideal * disc).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def sign_test(pairs) -> dict:
    """符号检验：正号计 win，零计 tie。返回双侧二项 p（不含零）。"""
    pos = sum(1 for a, b in pairs if a > b)
    neg = sum(1 for a, b in pairs if a < b)
    tie = len(pairs) - pos - neg
    n = pos + neg
    if n == 0:
        return {"wins": 0, "losses": 0, "ties": tie, "n": 0, "p": 1.0}
    # 双侧二项检验：p = 2 * P(X >= max(pos,neg))，截断到 1
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    p = min(1.0, 2.0 * tail)
    return {"wins": pos, "losses": neg, "ties": tie, "n": n, "p": p}


def wilcoxon_signed_rank(pairs) -> dict:
    """双侧 Wilcoxon 符号秩（正态近似 + 连续性校正）。"""
    d = [a - b for a, b in pairs if a != b]
    n = len(d)
    if n == 0:
        return {"n": 0, "p": 1.0, "w_plus": 0.0}
    r = _rank(np.abs(np.asarray(d)))
    w_plus = float(r[np.asarray(d) > 0].sum())
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma <= 0:
        return {"n": n, "p": 1.0, "w_plus": w_plus}
    z = (w_plus - mu) / sigma
    # 双侧正态近似
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n": n, "p": float(min(1.0, p)), "w_plus": w_plus}


def cluster_bootstrap_p(pairs, n_boot=5000, seed=7):
    """以「组合」为重采样单元的 bootstrap p 值。

    pairs: [(combo_key, delta_a, delta_b), ...]
    对每个组合先求均值，再对组合有放回重采样，统计均值差 <= 0 的比例。
    """
    if not pairs:
        return {"p": 1.0, "n_boot": n_boot}
    by_combo: dict[str, list[float]] = {}
    for key, a, b in pairs:
        by_combo.setdefault(key, []).append(a - b)
    keys = list(by_combo)
    means = np.array([float(np.mean(by_combo[k])) for k in keys])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(means), size=(n_boot, len(means)))
    boot = means[idx].mean(axis=1)
    # 观测统计量与 0 的比较用双侧：bootstrap 均值分布中落在 <=0 或 >=0 的比例
    obs = float(means.mean())
    if obs >= 0:
        p = float((boot <= 0).mean())
    else:
        p = float((boot >= 0).mean())
    return {"p": float(min(1.0, 2.0 * min(p, 1.0 - p))), "n_boot": n_boot,
            "observed_mean": obs}


def split_half_rel(E: np.ndarray, sel: np.ndarray, ev: np.ndarray,
                   mode: str = "folded") -> float:
    """两半之间的真值秩相关。

    mode='folded' : |mean_p dB| （现稿口径）
    mode='signed' : mean_p dB （带符号）
    """
    a = E[:, sel].mean(axis=1)
    b = E[:, ev].mean(axis=1)
    if mode == "signed":
        return spearman(a, b)
    return spearman(np.abs(a), np.abs(b))


def score_rel(score_per_prompt: np.ndarray, sel: np.ndarray, rng: np.random.RandomState,
              n_sub=10) -> float:
    """分数自身的 split-half 可靠性：在选择半内部再随机分半，取中位秩相关。

    score_per_prompt: (n_prompt, n_feat)
    """
    h = sel.size // 2
    vals = []
    for _ in range(n_sub):
        perm = rng.permutation(sel)
        a, b = perm[:h], perm[h:]
        va = score_per_prompt[a].mean(axis=0)
        vb = score_per_prompt[b].mean(axis=0)
        vals.append(spearman(va, vb))
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if vals else float("nan")


def noise_model(E: np.ndarray, half: int = 100, n_draws: int = 200, seed: int = 0) -> float:
    """零拟合噪声模型预测的 split-half 可靠性（§7.4 的 rel_pred）。

    若真值不可测的来源只是独立采样噪声，则两半各自的均值是
    mu + (sigma/sqrt(half)) * z，抽 200 次取平均秩相关。

    注意：本函数与 66_measurability_model.noise_model 同定义，但**用带符号**的
    绝对值折叠（与现稿 §7.4 口径一致）。
    """
    mu = E.mean(axis=1)
    sd = E.std(axis=1, ddof=1)
    se = np.maximum(sd / np.sqrt(half), 1e-12)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_draws):
        x1 = np.abs(mu + se * rng.randn(len(mu)))
        x2 = np.abs(mu + se * rng.randn(len(mu)))
        r = spearman(x1, x2)
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------- 载入网格
def load_grid():
    recs = []
    for npz_p in sorted(GRID_DIR.glob("*/*/*.npz")):
        meta_p = npz_p.with_suffix(".json")
        meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
        d = np.load(npz_p, allow_pickle=True)
        recs.append({
            "key": f"{meta.get('model', npz_p.parent.parent.name)}|"
                   f"{meta.get('layer', npz_p.parent.name.replace('layer',''))}|"
                   f"{meta.get('task', npz_p.stem)}",
            "model": meta.get("model", npz_p.parent.parent.name),
            "layer": int(meta.get("layer", npz_p.parent.name.replace("layer", ""))),
            "task": meta.get("task", npz_p.stem),
            "E32": np.asarray(d["E32"], dtype=np.float64),
            "E16": np.asarray(d["E16"], dtype=np.float64),
            "pred": np.asarray(d["pred_p"], dtype=np.float64),
            "M2": np.asarray(d["M2_p"], dtype=np.float64),
            "M3": np.asarray(d["M3_p"], dtype=np.float64),
            "kappa": meta.get("kappa_f32"),
            "b0_acc": meta.get("b0_acc"),
        })
    return recs


# ---------------------------------------------------------------- 主流程
def analyse(rec, n_splits=N_SPLITS, seed0=SEED0):
    """对单个组合做逐 split 分析，返回每个 split 的全部统计量。"""
    E = rec["E32"]
    pred, M2 = rec["pred"], rec["M2"]
    n_feat, n_prompt = E.shape
    h = n_prompt // 2
    rows = []
    for s in range(n_splits):
        rs = seed0 + 1000 * s
        rng = np.random.RandomState(rs)
        perm = rng.permutation(n_prompt)
        sel, ev = perm[:h], perm[h:]

        c_fold_sel = np.abs(E[:, sel].mean(axis=1))
        c_fold_ev = np.abs(E[:, ev].mean(axis=1))
        c_sign_sel = E[:, sel].mean(axis=1)
        c_sign_ev = E[:, ev].mean(axis=1)

        rel_folded = spearman(c_fold_sel, c_fold_ev)
        rel_signed = spearman(c_sign_sel, c_sign_ev)

        pred_sel = pred[sel].mean(axis=0)
        m2_sel = M2[sel].mean(axis=0)

        # 分数自身的可靠性（选择半内部再分半）
        rel_pred = score_rel(np.abs(pred), sel, np.random.RandomState(rs + 1))
        rel_m2 = score_rel(M2, sel, np.random.RandomState(rs + 2))

        row = {"split": s,
               "rel_folded": rel_folded, "rel_signed": rel_signed,
               "rel_pred": rel_pred, "rel_m2": rel_m2}
        for name, sc in (("pred_full", pred_sel), ("M2", m2_sel)):
            rho_fold = spearman(np.abs(sc), c_fold_ev)
            rho_sign = spearman(sc, c_sign_ev)
            ceil_fold = math.sqrt(max(rel_folded, 0.0))
            ceil_sign = math.sqrt(max(rel_signed, 0.0))
            row[name] = {
                "rho_folded": rho_fold,
                "rho_signed": rho_sign,
                "ratio_folded": rho_fold / max(ceil_fold, 1e-9),
                "ratio_signed": rho_sign / max(ceil_sign, 1e-9),
                "disatten_folded": rho_fold / max(math.sqrt(max(rel_folded, 0.0)
                                                            * max(rel_pred if name == "pred_full"
                                                                   else rel_m2, 1e-12)), 1e-9),
                "disatten_signed": rho_sign / max(math.sqrt(max(rel_signed, 0.0)
                                                            * max(rel_pred if name == "pred_full"
                                                                   else rel_m2, 1e-12)), 1e-9),
                "ndcg@20_folded": ndcg_at_k(np.abs(sc), c_fold_ev, 20),
            }
        rows.append(row)
    return rows


def aggregate(rows):
    agg = {}
    for key in ("rel_folded", "rel_signed", "rel_pred", "rel_m2"):
        v = np.array([r[key] for r in rows], dtype=float)
        agg[key] = {"mean": float(np.nanmean(v)),
                    "sd": float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0}
    for name in ("pred_full", "M2"):
        for mk in ("rho_folded", "rho_signed", "ratio_folded", "ratio_signed",
                   "disatten_folded", "disatten_signed", "ndcg@20_folded"):
            v = np.array([r[name][mk] for r in rows], dtype=float)
            agg[f"{name}/{mk}"] = {"mean": float(np.nanmean(v)),
                                   "sd": float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0}
    return agg


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_grid()
    print(f"[E7] 载入 {len(recs)} 个组合")

    per_combo = {}
    for i, rec in enumerate(recs):
        rows = analyse(rec)
        per_combo[rec["key"]] = {"model": rec["model"], "layer": rec["layer"],
                                 "task": rec["task"], "kappa": rec["kappa"],
                                 "agg": aggregate(rows)}
        if (i + 1) % 12 == 0:
            print(f"   ... {i+1}/{len(recs)}")

    # ---------------- C07：聚类前 vs 聚类后 ----------------
    # 逐 split 对（组合, split）：现有稿的统计单元
    pair_pred_m2 = []      # (combo, split, rho_pred, rho_m2)
    pair_ndcg = []
    for rec in recs:
        rows = analyse(rec)
        for r in rows:
            pair_pred_m2.append((rec["key"], r["split"],
                                 r["pred_full"]["rho_folded"], r["M2"]["rho_folded"]))
            pair_ndcg.append((rec["key"], r["split"],
                              r["pred_full"]["ndcg@20_folded"], r["M2"]["ndcg@20_folded"]))

    unclustered_rho = sign_test([(a, b) for _, _, a, b in pair_pred_m2])
    unclustered_ndcg = sign_test([(a, b) for _, _, a, b in pair_ndcg])

    # 组合级：每组合两 split 求均值
    combo_rho = []
    combo_ndcg = []
    for rec in recs:
        agg = per_combo[rec["key"]]["agg"]
        combo_rho.append((rec["key"],
                          agg["pred_full/rho_folded"]["mean"], agg["M2/rho_folded"]["mean"]))
        combo_ndcg.append((rec["key"],
                           agg["pred_full/ndcg@20_folded"]["mean"], agg["M2/ndcg@20_folded"]["mean"]))

    clustered_rho = sign_test([(a, b) for _, a, b in combo_rho])
    clustered_ndcg = sign_test([(a, b) for _, a, b in combo_ndcg])
    w_rho = wilcoxon_signed_rank([(a, b) for _, a, b in combo_rho])
    w_ndcg = wilcoxon_signed_rank([(a, b) for _, a, b in combo_ndcg])
    boot_rho = cluster_bootstrap_p([(k, a, b) for k, a, b in combo_rho])
    boot_ndcg = cluster_bootstrap_p([(k, a, b) for k, a, b in combo_ndcg])

    # ---------------- C08：折叠 vs 带符号 ----------------
    rel_fold = np.array([per_combo[k]["agg"]["rel_folded"]["mean"] for k in per_combo])
    rel_sign = np.array([per_combo[k]["agg"]["rel_signed"]["mean"] for k in per_combo])
    rel_pred = np.array([per_combo[k]["agg"]["rel_pred"]["mean"] for k in per_combo])
    rel_m2 = np.array([per_combo[k]["agg"]["rel_m2"]["mean"] for k in per_combo])

    admissible = []
    for k, v in per_combo.items():
        kappa = v["kappa"]
        if kappa is None:
            continue
        if kappa >= TH_KAPPA_ADMIT:
            admissible.append(k)

    def mean_over(keys, path):
        vals = []
        for k in keys:
            node = per_combo[k]["agg"]
            for part in path:
                node = node[part]
            vals.append(node["mean"])
        return float(np.mean(vals)) if vals else float("nan")

    c08 = {
        "rel_folded_mean": float(np.nanmean(rel_fold)),
        "rel_signed_mean": float(np.nanmean(rel_sign)),
        "rel_folded_vs_signed_diff": float(np.nanmean(rel_fold) - np.nanmean(rel_sign)),
        "n_better_signed": int((rel_sign > rel_fold).sum()),
        "n_total_rel": int(rel_fold.size),
        "rel_pred_mean": float(np.nanmean(rel_pred)),
        "rel_M2_mean": float(np.nanmean(rel_m2)),
        "admissible_n": len(admissible),
        "admissible_rel_folded_mean": mean_over(admissible, ["rel_folded"]),
        "admissible_rel_signed_mean": mean_over(admissible, ["rel_signed"]),
        "admissible_ratio_folded_pred": mean_over(admissible, ["pred_full/ratio_folded"]),
        "admissible_ratio_signed_pred": mean_over(admissible, ["pred_full/ratio_signed"]),
        "admissible_disatten_folded_pred": mean_over(admissible, ["pred_full/disatten_folded"]),
        "admissible_disatten_signed_pred": mean_over(admissible, ["pred_full/disatten_signed"]),
        "admissible_ratio_folded_M2": mean_over(admissible, ["M2/ratio_folded"]),
        "admissible_ratio_signed_M2": mean_over(admissible, ["M2/ratio_signed"]),
        "admissible_disatten_folded_M2": mean_over(admissible, ["M2/disatten_folded"]),
        "admissible_disatten_signed_M2": mean_over(admissible, ["M2/disatten_signed"]),
    }

    # ---------------- C09：闸门 LOCO ----------------
    # rel_pred = 零拟合噪声模型预测的 split-half 可靠性（与 §7.4 同一定义）
    gate_rows = json.loads((GATE_DIR / "gate_report.json").read_text(encoding="utf-8"))
    gate_obs = []
    for g in gate_rows:
        key = f"{g.get('model')}|{g.get('layer')}|{g.get('task')}"
        gate_obs.append({
            "key": key, "kappa": g.get("kappa"), "rel_pred": g.get("rel_pred"),
            "verdict": g.get("verdict"), "path": g.get("path"),
        })

    # 直接从 gate_report 指向的 npz 算实测 rel（与该点自身口径完全可比）
    for g in gate_obs:
        p = Path(g["path"].replace("\\", "/"))
        if p.exists():
            z = np.load(p, allow_pickle=True)
            E = np.asarray(z["E"], dtype=np.float64)
            g["rel_measured"] = split_half_rel(E, np.arange(E.shape[1] // 2),
                                               np.arange(E.shape[1] // 2, E.shape[1]))
            g["npz"] = str(p)

    def gate_verdict(rel_pred, kappa):
        """闸门的三级判定，与 cfse_gate 的 rank_ok/degraded/reject 对齐。"""
        if rel_pred is None or kappa is None:
            return None
        if rel_pred < TH_REL_PRED:
            return "reject"
        if kappa < TH_KAPPA_REJECT:
            return "reject"
        if kappa < TH_KAPPA_ADMIT:
            return "degrade"
        return "admit"

    def observed_verdict(mrel, kappa):
        """观测判定：真值不可测 -> reject；可测但 kappa 偏低 -> degrade；否则 admit。"""
        if mrel is None:
            return None
        if mrel < 0.25:
            return "reject"
        if kappa is not None and kappa < TH_KAPPA_ADMIT:
            return "degrade"
        return "admit"

    # 内样本一致率（现行稿口径）
    in_sample_ok, in_sample_n, loco_ok, loco_n = 0, 0, 0, 0
    loco_detail = []
    for i, g in enumerate(gate_obs):
        v = gate_verdict(g["rel_pred"], g["kappa"])
        obs = observed_verdict(g.get("rel_measured"), g["kappa"])
        if v is None or obs is None:
            continue
        in_sample_n += 1
        if v == obs:
            in_sample_ok += 1
        # LOCO：用其余点重选 rel_pred 阈值（10% 分位，与 §7.4 的保守口径同族）
        others = [x["rel_pred"] for j, x in enumerate(gate_obs)
                  if j != i and x["rel_pred"] is not None]
        if others:
            thr = max(TH_REL_PRED, float(np.quantile(others, 0.10)))
            v_loco = gate_verdict(g["rel_pred"], g["kappa"])
            if g["rel_pred"] < thr:
                v_loco = "reject"
            loco_n += 1
            if v_loco == obs:
                loco_ok += 1
            loco_detail.append({"key": g["key"], "verdict": v_loco, "obs": obs,
                                "ok": v_loco == obs, "thr": thr})

    c09 = {
        "n_matched": in_sample_n,
        "in_sample_agree": f"{in_sample_ok}/{in_sample_n}",
        "in_sample_rate": in_sample_ok / in_sample_n if in_sample_n else float("nan"),
        "loco_agree": f"{loco_ok}/{loco_n}",
        "loco_rate": loco_ok / loco_n if loco_n else float("nan"),
        "loco_detail": loco_detail,
        "thresholds": {"rel_pred": TH_REL_PRED, "kappa_reject": TH_KAPPA_REJECT,
                       "kappa_admit": TH_KAPPA_ADMIT},
        "note": "rel_measured 由 gate_report 指向的同一 npz 计算，与该点 rel_pred 同口径",
    }

    # ---------------- 48 网格上的闸门一致率 ----------------
    # 用 §7.4 的零拟合噪声模型在网格自身的效应矩阵上算 rel_pred
    grid_points = []
    for rec in recs:
        rel_pred_grid = noise_model(rec["E32"])
        kappa = rec["kappa"]
        grid_points.append({
            "key": rec["key"], "kappa": kappa, "rel_pred": rel_pred_grid,
            "rel_measured": per_combo[rec["key"]]["agg"]["rel_folded"]["mean"],
            "verdict": gate_verdict(rel_pred_grid, kappa),
        })
    # 判定一致性：闸门说 admit 而实测 rel<0.25 -> 不一致（假准入）
    #             闸门说 reject 而实测 rel>=0.25 -> 不一致（假拒绝）
    grid_ok = sum(1 for g in grid_points
                  if (g["verdict"] == "admit" and g["rel_measured"] >= 0.25)
                  or (g["verdict"] == "degrade" and g["rel_measured"] >= 0.25)
                  or (g["verdict"] == "reject" and g["rel_measured"] < 0.25))
    grid_false_admit = sum(1 for g in grid_points
                           if g["verdict"] in ("admit", "degrade") and g["rel_measured"] < 0.25)
    grid_false_reject = sum(1 for g in grid_points
                            if g["verdict"] == "reject" and g["rel_measured"] >= 0.25)
    c09["grid_agree"] = f"{grid_ok}/{len(grid_points)}"
    c09["grid_false_admit"] = grid_false_admit
    c09["grid_false_reject"] = grid_false_reject
    c09["grid_points"] = grid_points

    # ---------------- C19：pooled n ----------------
    n_main, n_desat = 48, 12
    c19 = {
        "n_48_main": n_main * 200,
        "n_60_all": n_main * 200 + n_desat * 50,
        "erroneous_12000": 60 * 200,
        "note": "12 个去饱和组合的池为 top-50，不是 200；60x200=12000 多算 1800 条",
    }

    out = {
        "schema": "cfse_e7_stats/v1",
        "n_combos": len(recs),
        "n_splits": N_SPLITS,
        "C07_clustering": {
            "unclustered_pairs": {
                "rho": unclustered_rho, "ndcg@20": unclustered_ndcg,
                "n_units": len(pair_pred_m2),
                "note": "现有稿口径：把 (组合, split) 当独立样本",
            },
            "clustered_combinations": {
                "rho": clustered_rho, "ndcg@20": clustered_ndcg,
                "wilcoxon_rho": w_rho, "wilcoxon_ndcg": w_ndcg,
                "bootstrap_rho": boot_rho, "bootstrap_ndcg": boot_ndcg,
                "n_units": len(combo_rho),
                "note": "正确口径：先按组合对两 split 求均值，再以组合为单元",
            },
        },
        "C08_reliability_semantics": c08,
        "C09_gate_calibration": c09,
        "C19_pooled_n": c19,
        "per_combination": per_combo,
    }
    (OUT / "e7_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 报告 ----------------
    md = []
    md.append("# E7 统计单元重算报告\n")
    md.append(f"组合数 {len(recs)}，每组合 {N_SPLITS} 个 split。全部离线，无机时。\n")

    md.append("\n## C07 伪重复：聚类前 vs 聚类后\n")
    md.append("| 统计量 | 单元 | n | 胜/负/平 | p |")
    md.append("|---|---|---|---|---|")
    md.append(f"| ρ(pred, M2) | (组合, split) 对 [现稿] | {unclustered_rho['n']} | "
              f"{unclustered_rho['wins']}/{unclustered_rho['losses']}/{unclustered_rho['ties']} | "
              f"{unclustered_rho['p']:.3g} |")
    md.append(f"| ρ(pred, M2) | 组合 [聚类] | {clustered_rho['n']} | "
              f"{clustered_rho['wins']}/{clustered_rho['losses']}/{clustered_rho['ties']} | "
              f"{clustered_rho['p']:.3g} |")
    md.append(f"| nDCG@20 | (组合, split) 对 [现稿] | {unclustered_ndcg['n']} | "
              f"{unclustered_ndcg['wins']}/{unclustered_ndcg['losses']}/{unclustered_ndcg['ties']} | "
              f"{unclustered_ndcg['p']:.3g} |")
    md.append(f"| nDCG@20 | 组合 [聚类] | {clustered_ndcg['n']} | "
              f"{clustered_ndcg['wins']}/{clustered_ndcg['losses']}/{clustered_ndcg['ties']} | "
              f"{clustered_ndcg['p']:.3g} |")
    md.append(f"\nWilcoxon（组合级）ρ：p={w_rho['p']:.3g}；nDCG@20：p={w_ndcg['p']:.3g}")
    md.append(f"cluster bootstrap（组合为单元）ρ：p={boot_rho['p']:.3g}；"
              f"nDCG@20：p={boot_ndcg['p']:.3g}\n")

    md.append("\n## C08 折叠 vs 带符号的可靠性\n")
    md.append("| 量 | 值 |")
    md.append("|---|---|")
    md.append(f"| rel(folded) 均值 | {c08['rel_folded_mean']:.4f} |")
    md.append(f"| rel(signed) 均值 | {c08['rel_signed_mean']:.4f} |")
    md.append(f"| 差（folded − signed） | {c08['rel_folded_vs_signed_diff']:+.4f} |")
    md.append(f"| signed 更高的组合数 | {c08['n_better_signed']}/{c08['n_total_rel']} |")
    md.append(f"| rel(pred) 均值 | {c08['rel_pred_mean']:.4f} |")
    md.append(f"| rel(M2) 均值 | {c08['rel_M2_mean']:.4f} |")
    md.append(f"\n准入组合 {c08['admissible_n']} 个：")
    md.append("| 口径 | pred | M2 |")
    md.append("|---|---|---|")
    md.append(f"| ratio (folded, 现稿) | {c08['admissible_ratio_folded_pred']:.4f} | "
              f"{c08['admissible_ratio_folded_M2']:.4f} |")
    md.append(f"| ratio (signed) | {c08['admissible_ratio_signed_pred']:.4f} | "
              f"{c08['admissible_ratio_signed_M2']:.4f} |")
    md.append(f"| disattenuated (folded) | {c08['admissible_disatten_folded_pred']:.4f} | "
              f"{c08['admissible_disatten_folded_M2']:.4f} |")
    md.append(f"| disattenuated (signed) | {c08['admissible_disatten_signed_pred']:.4f} | "
              f"{c08['admissible_disatten_signed_M2']:.4f} |")
    md.append(f"\n准入组合 rel(folded) 均值 {c08['admissible_rel_folded_mean']:.4f}，"
              f"rel(signed) 均值 {c08['admissible_rel_signed_mean']:.4f}\n")

    md.append("\n## C09 闸门标定\n")
    md.append(f"- 内样本一致率：{c09['in_sample_agree']}（{c09['in_sample_rate']:.3f}）")
    md.append(f"- LOCO 一致率：{c09['loco_agree']}（{c09['loco_rate']:.3f}）")
    md.append(f"- 48 网格一致：{c09['grid_agree']}（假准入 {c09['grid_false_admit']}，"
              f"假拒绝 {c09['grid_false_reject']}）")
    md.append(f"- 阈值：rel_pred < {TH_REL_PRED} 判 reject；κ < {TH_KAPPA_REJECT} reject；"
              f"{TH_KAPPA_REJECT} ≤ κ < {TH_KAPPA_ADMIT} degrade；κ ≥ {TH_KAPPA_ADMIT} admit\n")

    md.append("\n## C19 pooled n\n")
    md.append(f"- 48 主组合：{c19['n_48_main']:,}")
    md.append(f"- 60 组合合计：{c19['n_60_all']:,}")
    md.append(f"- 现有稿误写的 60×200 = {c19['erroneous_12000']:,}（多算 1,800 条）\n")

    (OUT / "e7_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n[E7] 输出 -> {OUT}")


if __name__ == "__main__":
    main()
