# -*- coding: utf-8 -*-
"""E1-post — 把 E1 的「逐元素噪声」拆成两级：单元级 vs 聚合级

E1 直接比较 8 种配置下效应矩阵的**逐元素**相关，得到的相关只有 0.1--0.9。
但稿件里没有任何一个结论建立在单个 (特征, prompt) 单元上：
所有推断都在**聚合级**（特征对 prompt 的平均效应），以及**秩**
（Spearman over 200 个特征）。

因此本脚本对同一批 npz 做第二次读数：

  (A) 单元级：所有 (f, p) 单元的重测噪声 sigma_cell 与中位 |dB| 的比值
  (B) 特征级：每特征对 prompt 求均值后的向量，配置间相关
  (C) prompt 级：每 prompt 对特征求均值后的向量，配置间相关
  (D) 秩级：每特征均值效应的跨配置 Spearman（这才是稿件实际用的量）

同时按 dtype 是否相同、batch 是否等于 1 把配置对分成三类，
因为「同 dtype 换 batch」才是审稿意见真正质疑的那件事。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "phaseStep2" / "cfse_e1_batchnoise"


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(float)
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(float)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() < 1e-15 or b.std() < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _dtype(k):
    return k.split("_")[0]


def _batch(k):
    return int(k.split("_b")[1])


def classify(rows):
    """按 dtype / batch 把配置对分成三类。

    不能用字符串包含判断 "b1"：'b16' 与 'bf16' 都含 'b1'，
    必须解析出整数 batch 再比较。
    """
    same, b1p, cross = [], [], []
    for r in rows:
        da, db = _dtype(r["a"]), _dtype(r["b"])
        ba, bb = _batch(r["a"]), _batch(r["b"])
        if ba == 1 or bb == 1:
            b1p.append(r)
        elif da == db:
            same.append(r)
        elif ba == bb:
            cross.append(r)
    return same, b1p, cross


def med(rs, k):
    v = [r[k] for r in rs if np.isfinite(r[k])]
    return float(np.median(v)) if v else float("nan")


def pair_stats(mats, keys):
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A, B = mats[a], mats[b]
            fA, fB = np.abs(A.mean(axis=1)), np.abs(B.mean(axis=1))
            pA, pB = np.abs(A.mean(axis=0)), np.abs(B.mean(axis=0))
            rows.append({
                "a": a, "b": b,
                "cell_pearson": pearson(A, B),
                "feat_pearson_absmean": pearson(fA, fB),
                "feat_spearman_absmean": spearman(fA, fB),
                "prompt_pearson_absmean": pearson(pA, pB),
                "sigma_cell": float(np.std(A - B)),
                "median_abs_effect": float(np.median(np.abs(A))),
            })
    for r in rows:
        r["cell_ratio"] = r["sigma_cell"] / max(r["median_abs_effect"], 1e-12)
    return rows


def main():
    recs = []
    for jp in sorted(OUT.glob("*/*/*_e1.json")):
        npz_p = jp.with_suffix(".npz")
        if not npz_p.exists():
            continue
        d = json.load(open(jp, encoding="utf-8"))
        z = np.load(npz_p)
        keys = sorted([k for k in z.files if k.startswith(("bf16_b", "f32_b"))])
        mats = {k: z[k].astype(np.float64) for k in keys}
        rows = pair_stats(mats, keys)
        same, b1p, cross = classify(rows)

        rec = {
            "model": d["model"], "layer": d["layer"], "task": d["task"],
            "n_features": d["n_features"], "n_prompts": d["n_prompts"],
            "n_pairs": len(rows),
            "same_dtype_batch_ge8": {
                "n_pairs": len(same),
                "median_cell_pearson": med(same, "cell_pearson"),
                "median_feat_spearman": med(same, "feat_spearman_absmean"),
                "median_feat_pearson": med(same, "feat_pearson_absmean"),
                "median_prompt_pearson": med(same, "prompt_pearson_absmean"),
                "median_cell_ratio": med(same, "cell_ratio"),
                "min_feat_spearman": float(np.nanmin(
                    [r["feat_spearman_absmean"] for r in same])) if same else float("nan"),
            },
            "batch1_pairs": {
                "n_pairs": len(b1p),
                "median_cell_pearson": med(b1p, "cell_pearson"),
                "median_feat_spearman": med(b1p, "feat_spearman_absmean"),
                "median_cell_ratio": med(b1p, "cell_ratio"),
            },
            "cross_dtype_same_batch": {
                "n_pairs": len(cross),
                "median_cell_pearson": med(cross, "cell_pearson"),
                "median_feat_spearman": med(cross, "feat_spearman_absmean"),
                "median_cell_ratio": med(cross, "cell_ratio"),
            },
            "pairs": rows,
        }
        recs.append(rec)
        s, b, c = (rec["same_dtype_batch_ge8"], rec["batch1_pairs"],
                   rec["cross_dtype_same_batch"])
        print(f"{rec['model']}|{rec['layer']}|{rec['task']:12s} "
              f"同dtype换batch(n={s['n_pairs']}): 单元r={s['median_cell_pearson']:.3f} "
              f"特征秩rho={s['median_feat_spearman']:.3f} "
              f"prompt r={s['median_prompt_pearson']:.3f} "
              f"噪声/效应={s['median_cell_ratio']:.2f} | "
              f"batch1(n={b['n_pairs']}): 特征秩rho={b['median_feat_spearman']:.3f} | "
              f"跨dtype: 单元r={c['median_cell_pearson']:.3f} "
              f"特征秩rho={c['median_feat_spearman']:.3f}")

    if recs:
        agg = {
            "n_combos": len(recs),
            "median_feat_spearman_same_dtype": med(
                [r["same_dtype_batch_ge8"] for r in recs], "median_feat_spearman"),
            "min_feat_spearman_same_dtype": float(np.nanmin(
                [r["same_dtype_batch_ge8"]["median_feat_spearman"] for r in recs])),
            "median_cell_pearson_same_dtype": med(
                [r["same_dtype_batch_ge8"] for r in recs], "median_cell_pearson"),
            "median_cell_ratio_same_dtype": med(
                [r["same_dtype_batch_ge8"] for r in recs], "median_cell_ratio"),
            "median_feat_spearman_batch1": med(
                [r["batch1_pairs"] for r in recs], "median_feat_spearman"),
            "median_feat_spearman_cross_dtype": med(
                [r["cross_dtype_same_batch"] for r in recs], "median_feat_spearman"),
        }
        (OUT / "e1_aggregate.json").write_text(json.dumps(
            {"results": recs, "aggregate": agg}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print("\n聚合读数：同 dtype 换 batch 的特征级秩相关中位 "
              f"{agg['median_feat_spearman_same_dtype']:.3f}"
              f"（组合间最小 {agg['min_feat_spearman_same_dtype']:.3f}）；"
              f"单元级 Pearson 中位 {agg['median_cell_pearson_same_dtype']:.3f}，"
              f"噪声/效应中位 {agg['median_cell_ratio_same_dtype']:.2f}")
        print(f"batch=1 对 特征级秩相关中位 {agg['median_feat_spearman_batch1']:.3f}；"
              f"跨 dtype 特征级秩相关中位 "
              f"{agg['median_feat_spearman_cross_dtype']:.3f}")


if __name__ == "__main__":
    main()
