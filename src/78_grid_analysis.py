# -*- coding: utf-8 -*-
"""E21 — 48 组网格的 held-out 分析（§5 换协议）

用 77 落盘的逐 prompt 数据，按 §7 的协议重算 §5 的主表：

  分数（选择半集重算）
    pred_full : 逐 prompt 预测在选择半集上取平均，再取绝对值
    M2 / M3   : 逐 prompt 激活均值 / 最大值在选择半集上取平均
  真值（评估半集）
    E32 : |mean_p E|，f32 口径（主口径）
    E16 : |mean_p E|，bf16 口径（对照）
  rel : Spearman(选择半集真值, 评估半集真值)
  ratio = ρ / sqrt(rel)

输出：
  output/phaseStep2/cfse_grid_heldout/grid_heldout.json / .md
"""
from __future__ import annotations

import json
import time
from importlib import import_module
from pathlib import Path

import numpy as np

_m48 = import_module("48_cfse_offline")
spearman = _m48.spearman
ndcg_at_k = _m48.ndcg_at_k
jaccard = _m48.jaccard

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_grid_heldout"

N_SPLITS = 2
SEED0 = 42
KS = (5, 10, 20, 50, 100)


def load_all():
    recs = []
    for npz_p in sorted(OUT.glob("*/*/*.npz")):
        d = np.load(npz_p, allow_pickle=True)
        meta_p = npz_p.with_suffix(".json")
        meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
        recs.append({
            "model": meta.get("model", npz_p.parent.parent.name),
            "layer": meta.get("layer", int(npz_p.parent.name.replace("layer", ""))),
            "task": meta.get("task", npz_p.stem),
            "E32": np.asarray(d["E32"], dtype=np.float64),
            "E16": np.asarray(d["E16"], dtype=np.float64),
            "pred": np.asarray(d["pred_p"], dtype=np.float64),
            "M2": np.asarray(d["M2_p"], dtype=np.float64),
            "M3": np.asarray(d["M3_p"], dtype=np.float64),
            "b0_acc": meta.get("b0_acc"),
            "kappa": meta.get("kappa_f32"),
            "kappa_bf16": meta.get("kappa_bf16"),
        })
    return recs


def analyse(rec, tag="f32", n_splits=N_SPLITS, seed0=SEED0):
    E = rec["E32"] if tag == "f32" else rec["E16"]
    pred = rec["pred"]
    M2 = rec["M2"]
    M3 = rec["M3"]
    n_feat, n_prompt = E.shape
    h = n_prompt // 2
    rows = []
    for s in range(n_splits):
        rs = seed0 + 1000 * s
        rng = np.random.RandomState(rs)
        perm = rng.permutation(n_prompt)
        sel, ev = perm[:h], perm[h:]
        c_sel = np.abs(E[:, sel].mean(axis=1))
        c_ev = np.abs(E[:, ev].mean(axis=1))
        db_ev = E[:, ev].mean(axis=1)
        rel = float(spearman(c_sel, c_ev))
        ceil = float(np.sqrt(max(rel, 0.0)))
        # 分数在**选择半集**上重算
        sc = {
            "pred_full": np.abs(pred[sel].mean(axis=0)),
            "M2": M2[sel].mean(axis=0),
            "M3": M3[sel].mean(axis=0),
        }
        row = {"split": s, "rel": rel, "ceiling": ceil}
        for name, v in sc.items():
            rho = float(spearman(v, c_ev))
            row[name] = {
                "rho": rho,
                "ratio": rho / max(ceil, 1e-9),
                "ndcg@20": float(ndcg_at_k(v, c_ev, 20)),
                "jac50": float(jaccard(np.argsort(-v, kind="mergesort")[:50],
                                       np.argsort(-c_ev, kind="mergesort")[:50])),
            }
            for k in KS:
                row[name][f"ndcg@{k}"] = float(ndcg_at_k(v, c_ev, k))
        row["sign_acc_pred"] = float(np.mean(
            np.sign(pred[sel].mean(axis=0)) == np.sign(db_ev)))
        rows.append(row)
    agg = {}
    for key in ("rel", "ceiling", "sign_acc_pred"):
        v = np.asarray([r[key] for r in rows], dtype=float)
        agg[key] = {"mean": float(v.mean()),
                    "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    for name in ("pred_full", "M2", "M3"):
        for mk in ("rho", "ratio", "ndcg@20", "jac50") + tuple(f"ndcg@{k}" for k in KS):
            v = np.asarray([r[name][mk] for r in rows], dtype=float)
            agg[f"{name}/{mk}"] = {"mean": float(v.mean()),
                                   "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return {"agg": agg, "splits": rows}


def sign_test(pairs):
    """pairs: list of (a, b)。返回 (wins, n, p_two_sided)。"""
    from math import comb
    wins = sum(1 for a, b in pairs if a > b)
    ties = sum(1 for a, b in pairs if a == b)
    n = wins + sum(1 for a, b in pairs if a < b)
    if n == 0:
        return wins, ties, 1.0
    k = min(wins, n - wins)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return wins, ties, min(1.0, p)


def render(recs, results, out_tag="f32"):
    L = [f"# E21 — {len(recs)} 组网格的 held-out（{out_tag} 口径）", "",
         f"> 生成 {time.strftime('%Y-%m-%d %H:%M')}｜{N_SPLITS} 分割｜"
         f"分数=选择半集重算，真值=评估半集", "",
         "| model | L | task | κ | rel | √rel | pred | M2 | M3 | pred/M2 | 结论 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(recs, key=lambda x: -(x["kappa"] if x["kappa"] else -1)):
        a = results[(r["model"], r["layer"], r["task"])]["agg"]
        p = a["pred_full/rho"]["mean"]
        q = a["M2/rho"]["mean"]
        verdict = "pred 胜" if p > q + 0.05 else ("M2 胜" if q > p + 0.05 else "持平")
        L.append(
            f"| {r['model']} | {r['layer']} | {r['task']} | "
            f"{r['kappa']:.3f} | {a['rel']['mean']:.3f} | {a['ceiling']['mean']:.3f} | "
            f"{p:+.3f} ({a['pred_full/ratio']['mean']:.2f}×) | "
            f"{q:+.3f} ({a['M2/ratio']['mean']:.2f}×) | "
            f"{a['M3/rho']['mean']:+.3f} ({a['M3/ratio']['mean']:.2f}×) | "
            f"{p/max(q,1e-9):.2f} | {verdict} |")
    md = OUT / f"grid_heldout_{out_tag}.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[写出] {md}")
    return md


def main():
    recs = load_all()
    if not recs:
        print("没有找到 77 的落盘数据")
        return
    print(f"载入 {len(recs)} 组")

    results = {}
    for r in recs:
        key = (r["model"], r["layer"], r["task"])
        results[key] = analyse(r, "f32")
        results[(key[0], key[1], key[2], "bf16")] = analyse(r, "bf16")

    # ---- 汇总统计 ----
    pred_ratios = [results[k]["agg"]["pred_full/ratio"]["mean"] for k in results
                   if len(k) == 3]
    m2_ratios = [results[k]["agg"]["M2/ratio"]["mean"] for k in results if len(k) == 3]
    wins, ties, p = sign_test([(a, b) for a, b in zip(pred_ratios, m2_ratios)])
    ndcg_wins, ndcg_ties, ndcg_p = sign_test(
        [(results[k]["agg"]["pred_full/ndcg@20"]["mean"],
          results[k]["agg"]["M2/ndcg@20"]["mean"]) for k in results if len(k) == 3])

    rels = [results[k]["agg"]["rel"]["mean"] for k in results if len(k) == 3]
    kappas = [r["kappa"] for r in recs]

    summary = {
        "n_combos": len(recs),
        "n_splits": N_SPLITS,
        "rho": {
            "pred_mean": float(np.mean(pred_ratios)),
            "pred_min": float(np.min(pred_ratios)),
            "pred_max": float(np.max(pred_ratios)),
            "M2_mean": float(np.mean(m2_ratios)),
            "M2_min": float(np.min(m2_ratios)),
            "M2_max": float(np.max(m2_ratios)),
        },
        "rho_sign_test": {"wins": wins, "ties": ties, "n": len(pred_ratios), "p": p},
        "ndcg20_sign_test": {"wins": ndcg_wins, "ties": ndcg_ties,
                             "n": len(pred_ratios), "p": ndcg_p},
        "rel": {"mean": float(np.mean(rels)), "min": float(np.min(rels)),
                "max": float(np.max(rels))},
        "kappa": {"mean": float(np.mean(kappas)),
                  "min": float(np.min(kappas)), "max": float(np.max(kappas))},
        "R_kappa_rel": (float(spearman(kappas, rels))
                        if len(kappas) >= 3 else None),
    }

    # κ 分层
    bands = {"reject(<0.10)": [], "degraded(0.10-0.30)": [], "admit(>=0.30)": []}
    for r in recs:
        k = r["kappa"]
        key = ("reject(<0.10)" if k < 0.10 else
               ("degraded(0.10-0.30)" if k < 0.30 else "admit(>=0.30)"))
        bands[key].append((r, results[(r["model"], r["layer"], r["task"])]["agg"]))
    summary["bands"] = {}
    for name, lst in bands.items():
        if not lst:
            continue
        w = sum(1 for _, a in lst if a["pred_full/rho"]["mean"] > a["M2/rho"]["mean"])
        t = sum(1 for _, a in lst if a["pred_full/rho"]["mean"] == a["M2/rho"]["mean"])
        l = len(lst) - w - t
        summary["bands"][name] = {
            "n": len(lst),
            "pred_ratio_mean": float(np.mean([a["pred_full/ratio"]["mean"] for _, a in lst])),
            "M2_ratio_mean": float(np.mean([a["M2/ratio"]["mean"] for _, a in lst])),
            "pred_wins": w, "ties": t, "M2_wins": l,
        }
        print(f"\n[{name}] n={len(lst)}: pred 胜 {w} 平 {t} 负 {l}")
        print(f"   pred ratio 均值 {summary['bands'][name]['pred_ratio_mean']:.3f} | "
              f"M2 ratio 均值 {summary['bands'][name]['M2_ratio_mean']:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    json_p = OUT / "grid_heldout.json"
    json_p.write_text(json.dumps({
        "summary": summary,
        "records": [
            {"model": r["model"], "layer": r["layer"], "task": r["task"],
             "kappa": r["kappa"], "b0_acc": r["b0_acc"],
             "agg": results[(r["model"], r["layer"], r["task"])]["agg"]}
            for r in recs],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {json_p}")

    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    render(recs, results, "f32")


if __name__ == "__main__":
    main()
