# -*- coding: utf-8 -*-
"""E10 — SAE 训练种子方差：ρ 与选择集是否随字典重训而漂移

回应审稿意见 C14（R1-M3）。

问题
----
每个 (model, layer) 只训了一个 SAE，"SAE 质量"因此是固定条件而非变量。审稿人问：
如果换一个训练种子重训字典，M1--M6 与效应的关系、以及"指标无用"的结论还成立吗？

设计
----
同一层、同一批激活、同一套超参，只换种子训 3 个 SAE（seed 42/43/44），各自跑一遍
单特征干预，比较：

  1. 每个种子的 ρ(metric, |ΔB|)（n=200），看离散度；
  2. "结论是否翻向"：是否有种子让某个代理指标超过 0.30 的可报告门槛；
  3. 选择集稳定性：每个种子内 top-50(按 |ΔB|) 与 top-50(按 M2) 的 Jaccard。

关于第 3 点的一个说明：独立训练的两个字典的特征编号互不对应，跨种子比较"同一批
特征"的 Jaccard 没有定义。因此这里报的是**种子内**选择器的命中率稳定性，而不是
跨字典的特征同一性——后者只能通过解码方向匹配来定义，不在本文范围内。

用法
----
  python 9J_c14_seeds.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
OUT = BASE / "cfse_e10_saeseeds"
MODEL, LAYER, TASK = "pythia-1.4b", 16, "ioi"

RUNS = [("layer_sae", "causal_validity", "s42"),
        ("layer_sae_s43", "causal_validity_s43", "s43"),
        ("layer_sae_s44", "causal_validity_s44", "s44")]

METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
           "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def load_run(sae_dir, cv_dir):
    p = BASE / cv_dir / MODEL / f"layer{LAYER}" / f"{TASK}.json"
    if not p.exists():
        return None
    d = json.load(open(p, encoding="utf-8"))
    mfile = BASE / sae_dir / MODEL / f"layer{LAYER}" / "per_layer_sae_metrics.json"
    m = json.load(open(mfile, encoding="utf-8")) if mfile.exists() else {}
    feats = [r for r in d["features"] if r.get("C_ablate") is not None]
    return {"rec": d, "metrics": m, "feats": feats}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, per_seed = [], {}
    for sae_dir, cv_dir, tag in RUNS:
        got = load_run(sae_dir, cv_dir)
        if got is None:
            print(f"[缺] {tag}: 无 {cv_dir}/{MODEL}/layer{LAYER}/{TASK}.json")
            continue
        feats = got["feats"]
        eff = np.array([abs(r["B_ablate"] - got["rec"]["b0"]) for r in feats])
        rec = {"tag": tag,
               "sae_val_cos": got["metrics"].get("val_cos"),
               "sae_val_mse": got["metrics"].get("val_recon_mse"),
               "sae_dead": got["metrics"].get("dead_features"),
               "n_features": len(feats), "rho": {}}
        for m in METRICS:
            v = np.array([r.get(m) if r.get(m) is not None else np.nan for r in feats],
                         dtype=np.float64)
            rec["rho"][m] = spearman(v, eff)
        # 选择器命中率：top-50 by |ΔB| 与 top-50 by 各指标
        top_eff = set(np.argsort(-eff)[:50].tolist())
        rec["jaccard_top50"] = {}
        for m in METRICS:
            v = np.array([r.get(m) if r.get(m) is not None else -np.inf for r in feats],
                         dtype=np.float64)
            rec["jaccard_top50"][m] = float(jaccard(np.argsort(-v)[:50].tolist(), top_eff))
        per_seed[tag] = rec
        rows.append(rec)
        print(f"[{tag}] cos={rec['sae_val_cos']:.3f} "
              f"rho(M2)={rec['rho']['M2_mean_act_task']:+.3f} "
              f"rho(M1)={rec['rho']['M1_act_freq_task']:+.3f}", flush=True)

    if not rows:
        print("没有任何种子结果")
        return

    summary = {}
    for m in METRICS:
        v = [r["rho"][m] for r in rows]
        j = [r["jaccard_top50"][m] for r in rows]
        summary[m] = {"rho_mean": float(np.mean(v)), "rho_std": float(np.std(v, ddof=1))
                      if len(v) > 1 else 0.0,
                      "rho_min": float(np.min(v)), "rho_max": float(np.max(v)),
                      "jaccard_mean": float(np.mean(j)),
                      "n_seeds": len(v)}
    res = {"model": MODEL, "layer": LAYER, "task": TASK,
           "seeds": list(per_seed), "per_seed": per_seed, "summary": summary}
    (OUT / "e10_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                          encoding="utf-8")

    print("\n指标            ρ 均值±标准差      min     max    top-50 Jaccard")
    for m in METRICS:
        s = summary[m]
        print(f"{m:22s} {s['rho_mean']:+.3f}±{s['rho_std']:.3f}  "
              f"{s['rho_min']:+.3f}  {s['rho_max']:+.3f}   {s['jaccard_mean']:.2f}")


if __name__ == "__main__":
    main()
