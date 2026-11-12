# -*- coding: utf-8 -*-
"""E7b — 把所有 (组合, split) 对级显著性检验改到组合级（C07）

现稿把 (combination, split) 对当作独立样本做符号检验，但 CI 已经用了
combination-clustered bootstrap。这是内部不一致：两个 split 是同一批 prompt 的
互补两半，共享特征、模型与任务。

本脚本对每一处报告过的比较，给出**同一数据**下的两套 p 值：
  - unclustered：现稿口径，(组合, split) 对为独立单元
  - clustered  ：正确口径，每组合两 split 先求均值，再以组合为单元

覆盖的现稿位置：
  §5.2  48 网格 rho 与 nDCG@20：44/48 与 46/48
  §5.4  N=50 上 z 的 86/96
  §5.4  2x2 符号设计：36/11、57/10、41/1、58/8
  §6    归因基线：21/10、27/5、24/8、28/4、32/0、31/1
  §7.6  top-k：35/36、34/36
  §7.5  8/8

输出 output/phaseStep2/cfse_e7_stats/e7b_clustered_tests.json
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_e7_stats"
GRID_DIR = BASE / "cfse_grid_heldout"
ATTR_DIR = BASE / "cfse_attrib_heldout"
SETLEVEL = BASE / "cfse_setlevel_heldout"
SETLEVEL_STEER = BASE / "cfse_setlevel_steer_heldout"
RECALL = BASE / "cfse_recall"

_m48 = import_module("48_cfse_offline")
spearman = _m48.spearman


def _rank(a):
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


def sign_test(pairs):
    pos = sum(1 for a, b in pairs if a > b)
    neg = sum(1 for a, b in pairs if a < b)
    tie = len(pairs) - pos - neg
    n = pos + neg
    if n == 0:
        return {"wins": 0, "losses": 0, "ties": tie, "n": 0, "p": 1.0}
    k = max(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return {"wins": pos, "losses": neg, "ties": tie, "n": n,
            "p": min(1.0, 2.0 * tail)}


def wilcoxon(pairs):
    d = [a - b for a, b in pairs if a != b]
    n = len(d)
    if n == 0:
        return {"n": 0, "p": 1.0}
    r = _rank(np.abs(np.asarray(d)))
    w = float(r[np.asarray(d) > 0].sum())
    mu = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mu) / sd if sd > 0 else 0.0
    return {"n": n, "p": float(min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))}


def cluster_bootstrap(pairs, n_boot=5000, seed=7):
    by = {}
    for k, a, b in pairs:
        by.setdefault(k, []).append(a - b)
    keys = list(by)
    means = np.array([float(np.mean(by[k])) for k in keys])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(means), size=(n_boot, len(means)))
    boot = means[idx].mean(axis=1)
    obs = float(means.mean())
    p = float((boot <= 0).mean()) if obs >= 0 else float((boot >= 0).mean())
    return {"p": float(min(1.0, 2.0 * min(p, 1.0 - p))), "observed": obs,
            "n_combos": len(means)}


def report(name, rows, results):
    """rows: [(combo_key, a, b), ...]"""
    by_combo = {}
    for k, a, b in rows:
        by_combo.setdefault(k, []).append((a, b))
    combo_rows = [(k, float(np.mean([x[0] for x in v])),
                   float(np.mean([x[1] for x in v]))) for k, v in by_combo.items()]
    results[name] = {
        "unclustered": sign_test([(a, b) for _, a, b in rows]),
        "clustered": sign_test([(a, b) for _, a, b in combo_rows]),
        "clustered_wilcoxon": wilcoxon([(a, b) for _, a, b in combo_rows]),
        "clustered_bootstrap": cluster_bootstrap(combo_rows),
        "n_units_unclustered": len(rows),
        "n_units_clustered": len(combo_rows),
    }


def ndcg_at_k(score, gain, k):
    score = np.asarray(score, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    k = min(k, score.size)
    idx = np.argsort(-score, kind="mergesort")[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gain[idx] * disc).sum())
    ideal = np.sort(gain)[::-1][:k]
    idcg = float((ideal * disc).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    # ================= 1) 48 网格：pred vs M2 的 rho 与 nDCG@20 =================
    recs = []
    for npz_p in sorted(GRID_DIR.glob("*/*/*.npz")):
        meta_p = npz_p.with_suffix(".json")
        meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
        d = np.load(npz_p, allow_pickle=True)
        recs.append({
            "key": f"{meta.get('model', npz_p.parent.parent.name)}|"
                   f"{meta.get('layer', npz_p.parent.name.replace('layer',''))}|"
                   f"{meta.get('task', npz_p.stem)}",
            "E32": np.asarray(d["E32"], dtype=np.float64),
            "pred": np.asarray(d["pred_p"], dtype=np.float64),
            "M2": np.asarray(d["M2_p"], dtype=np.float64),
        })

    N_SPLITS, SEED0 = 20, 4242
    rows_rho, rows_ndcg = [], []
    for rec in recs:
        E, pred, M2 = rec["E32"], rec["pred"], rec["M2"]
        n_feat, n_prompt = E.shape
        h = n_prompt // 2
        for s in range(N_SPLITS):
            rng = np.random.RandomState(SEED0 + 1000 * s)
            perm = rng.permutation(n_prompt)
            sel, ev = perm[:h], perm[h:]
            c_ev = np.abs(E[:, ev].mean(axis=1))
            p_s = np.abs(pred[sel].mean(axis=0))
            m_s = M2[sel].mean(axis=0)
            rows_rho.append((rec["key"], float(spearman(p_s, c_ev)),
                             float(spearman(m_s, c_ev))))
            rows_ndcg.append((rec["key"], float(ndcg_at_k(p_s, c_ev, 20)),
                              float(ndcg_at_k(m_s, c_ev, 20))))
    report("grid48_rho_pred_vs_M2", rows_rho, results)
    report("grid48_ndcg20_pred_vs_M2", rows_ndcg, results)

    # ================= 2) 集合层：N=50 上 z 的 86/96 =================
    sl_rows = []
    for p in sorted(SETLEVEL.glob("*/*/*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        key = f"{d.get('model', p.parent.parent.name)}|{d.get('layer', p.parent.name.replace('layer',''))}|{d.get('task', p.stem)}"
        for sp in d.get("splits", []):
            for res in sp.get("results", []):
                if res.get("N") != 50:
                    continue
                sel = res.get("selectors", {})
                if "pred_dir" in sel and "M2" in sel:
                    sl_rows.append((key, float(sel["pred_dir"]["z"]),
                                    float(sel["M2"]["z"])))
    if sl_rows:
        report("setlevel_N50_z_pred_dir_vs_M2", sl_rows, results)
    else:
        print("[E7b] 集合层 z 未找到，检查结构")

    # ================= 3) 2x2 符号设计 =================
    two_rows = []
    for p in sorted(SETLEVEL_STEER.glob("*/*/*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        key = f"{d.get('model', p.parent.parent.name)}|{d.get('layer', p.parent.name.replace('layer',''))}|{d.get('task', p.stem)}"
        for sp in d.get("splits", []):
            t = sp.get("two_by_two")
            if not t:
                continue
            try:
                ab = t["ablate"]
                st = t["steer"]
                two_rows.append((key + "|ablate",
                                 float(ab["pred_dir"]["flip"]), float(ab["pred_steer"]["flip"])))
                two_rows.append((key + "|steer",
                                 float(st["pred_steer"]["flip"]), float(st["pred_dir"]["flip"])))
            except (KeyError, TypeError):
                continue
    if two_rows:
        report("two_by_two_diag_vs_offdiag", two_rows, results)

    # ================= 4) 归因基线 =================
    attr_rows = []
    for p in sorted(ATTR_DIR.glob("*/*/*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        key = f"{d.get('model', p.parent.parent.name)}|{d.get('layer', p.parent.name.replace('layer',''))}|{d.get('task', p.stem)}"
        for sp in d.get("splits", []):
            for base, node in (("full", sp.get("corr_full")),
                               ("shap", sp.get("corr_shap"))):
                if not isinstance(node, dict):
                    continue
                for a, b in (("pred_full", "KernelSHAP"), ("pred_full", "IG"),
                             ("pred_full", "attention"), ("pred_full", "M2"),
                             ("pred_full", "random")):
                    if a in node and b in node:
                        attr_rows.append((f"{key}|{base}|{a}-{b}",
                                          float(node[a]), float(node[b])))
    if attr_rows:
        report("attrib_paired", attr_rows, results)
    else:
        print("[E7b] 归因结构未匹配")
        for p in sorted(ATTR_DIR.glob("*/*/*.json"))[:1]:
            dd = json.loads(p.read_text(encoding="utf-8"))
            if "splits" in dd and dd["splits"]:
                s0 = dd["splits"][0]
                print("   corr_full:", json.dumps(s0.get("corr_full"), ensure_ascii=False)[:220])
                print("   corr_shap:", json.dumps(s0.get("corr_shap"), ensure_ascii=False)[:220])

    # ================= 5) recall（top-k）=================
    rec_rows = []
    for p in sorted(RECALL.glob("**/*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            continue
        for ck, cd in d.items():
            if isinstance(cd, dict) and "splits" in cd:
                for sp in cd["splits"]:
                    v = sp.get("pred_full", {}), sp.get("M2", {})
                    if all(isinstance(x, dict) for x in v):
                        rec_rows.append((f"{p.stem}|{ck}",
                                         float(v[0].get("found", 0)),
                                         float(v[1].get("found", 0))))
    if rec_rows:
        report("recall_topk_pred_vs_M2", rec_rows, results)

    (OUT / "e7b_clustered_tests.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 78)
    for k, v in results.items():
        u, c = v.get("unclustered", {}), v.get("clustered", {})
        print(f"\n{k}")
        print(f"  unclustered  n={u.get('n'):4d}  {u.get('wins')}/{u.get('losses')}"
              f"/{u.get('ties')}  p={u.get('p', 0):.3g}")
        print(f"  clustered    n={c.get('n'):4d}  {c.get('wins')}/{c.get('losses')}"
              f"/{c.get('ties')}  p={c.get('p', 0):.3g}")
        print(f"  clustered Wilcoxon p={v['clustered_wilcoxon']['p']:.3g}"
              f" | bootstrap p={v['clustered_bootstrap']['p']:.3g}")
    print("=" * 78)


if __name__ == "__main__":
    main()
