# -*- coding: utf-8 -*-
"""E30 — pooled ρ 的三口径对照（本轮最重要的口径审计）

问题
----
§5.1 报告的 "Pooled Spearman rises from +0.206 (M2) to +0.725 (pred_full)" 用的是
**pooled_spearman**：把 60 组 × 200 特征 = 10200 个 (分数, 真值) 对拼在一起算一个 ρ。
而 §5.1 同一段还报告了 "per-combination mean 0.675±0.303"。

held-out 之后，pooled 与逐组合两个尺度给出**很不一样的倍数**：

  逐组合（每组内部排 200 个特征）: pred 0.843 vs M2 0.225 → 3.74×
  pooled（38 组合拼成 7600 对）   : pred 0.907 vs M2 0.633 → 1.43×

原因：pooled 拼组后，**组间的系统性差异**（某模型/层的特征整体更强）主导了相关性，
组内区分度被稀释。这不是 bug，是尺度的定义差异，但论文必须说清楚用的是哪个。

本脚本算齐三口径，给出可直接引用的对照表：

  A. 同 prompt，逐组合（60 组）        —— §5.1 现在的数字
  B. 同 prompt，pooled（10200 对）     —— §5.1 的 0.725 / 0.206
  C. held-out，逐组合（48 / 38 组）
  D. held-out，pooled（48×200×2 / 38×200×2 对）

用法：python 87_pooled_audit.py
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
BASE = ROOT / "output" / "phaseStep2"

_m48 = import_module("48_cfse_offline")
sp = _m48.spearman


def same_prompt():
    """A/B：旧口径（stepA_offline，60 组，同 prompt）。"""
    d = json.load(open(BASE / "causal_margin" / "stepA_offline.json",
                       encoding="utf-8"))
    p = d["pooled"]
    out = {}
    for name in ("pred_full", "pred_last", "pred_align",
                 "M2_mean_act_task", "M3_max_activation"):
        if name in p:
            v = p[name]
            out[name] = {"pooled": v["pooled_spearman"],
                         "mean_combo": v["mean_combo_spearman"],
                         "std_combo": v.get("std_combo_spearman")}
    return out, len(d["per_combo"])


def admissible_set():
    """门控放行的 (model, layer, task)：held-out κ ≥ 0.30。"""
    adm = set()
    for f in sorted(BASE.glob("cfse_grid_heldout/*/*/*.json")):
        m = json.load(open(f, encoding="utf-8"))
        k = m.get("kappa_f32")
        if k is not None and k >= 0.30:
            adm.add((m["model"], m["layer"], m["task"]))
    return adm


def same_prompt_admissible(adm):
    """附录 Table 的 'same prompt, admissible' 行。"""
    d = json.load(open(BASE / "causal_margin" / "stepA_offline.json",
                       encoding="utf-8"))
    want = {f"{a}|L{b}|{c}" for a, b, c in adm}
    sub = [r for r in d["per_combo"] if r["combo"] in want]
    out = {"n": len(sub)}
    for name, key in (("pred_full", "pred_full"), ("M2", "M2_mean_act_task")):
        vals = [r["spearman"][key] for r in sub]
        out[name] = {"mean_combo": float(np.mean(vals))}
    return out


def held_out():
    """C/D：held-out 口径，逐组合与 pooled 都算。"""
    per = {"pred_full": [], "M2": [], "M3": []}
    pooled = {"pred_full": [], "M2": [], "M3": []}
    pool_all = []
    n_combos = 0
    for f in sorted(BASE.glob("cfse_grid_heldout/*/*/*.npz")):
        meta_p = f.with_suffix(".json")
        meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
        k = meta.get("kappa_f32")
        adm = (k is not None and k >= 0.30)
        d = np.load(f, allow_pickle=True)
        E32 = np.asarray(d["E32"], dtype=np.float64)
        pred = np.asarray(d["pred_p"], dtype=np.float64)
        M2 = np.asarray(d["M2_p"], dtype=np.float64)
        M3 = np.asarray(d["M3_p"], dtype=np.float64)
        n = E32.shape[1]
        h = n // 2
        for s in range(2):
            rng = np.random.RandomState(42 + 1000 * s)
            perm = rng.permutation(n)
            sel, ev = perm[:h], perm[h:]
            c_ev = np.abs(E32[:, ev].mean(axis=1))
            scores = {"pred_full": np.abs(pred[sel].mean(axis=0)),
                      "M2": M2[sel].mean(axis=0),
                      "M3": M3[sel].mean(axis=0)}
            for name, v in scores.items():
                per[name].append(sp(v, c_ev))
                if adm:
                    pooled[name].append((v, c_ev))
        if adm:
            n_combos += 1
    out = {"per_combo_all48": {k: float(np.mean(v)) for k, v in per.items()},
           "per_combo_all48_n": len(per["pred_full"]),
           "pooled_admit38": {},
           "n_admit": n_combos}
    for name, lst in pooled.items():
        E = np.concatenate([b for _, b in lst])
        S = np.concatenate([a for a, _ in lst])
        out["pooled_admit38"][name] = {"pooled": float(sp(S, E)),
                                       "n_pairs": int(len(E))}
    # 组间层面：把每个 (组合, 分割) 的均值当成一个点再相关（附录 tab:scales 末段）
    pm = {"pred_full": [], "M2": [], "effect": []}
    for f in sorted(BASE.glob("cfse_grid_heldout/*/*/*.npz")):
        meta_p = f.with_suffix(".json")
        meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
        k = meta.get("kappa_f32")
        if not (k is not None and k >= 0.30):
            continue
        d = np.load(f, allow_pickle=True)
        E32 = np.asarray(d["E32"], dtype=np.float64)
        pred = np.asarray(d["pred_p"], dtype=np.float64)
        M2 = np.asarray(d["M2_p"], dtype=np.float64)
        n = E32.shape[1]
        h = n // 2
        for s in range(2):
            rng = np.random.RandomState(42 + 1000 * s)
            perm = rng.permutation(n)
            sel, ev = perm[:h], perm[h:]
            pm["effect"].append(float(np.abs(E32[:, ev].mean(axis=1)).mean()))
            pm["pred_full"].append(float(np.abs(pred[sel].mean(axis=0)).mean()))
            pm["M2"].append(float(M2[sel].mean(axis=0).mean()))
    out["group_level"] = {
        "n_points": len(pm["effect"]),
        "spearman": {n: float(sp(np.array(pm[n]), np.array(pm["effect"])))
                     for n in ("pred_full", "M2")},
        "between_combo_sd": {n: float(np.std(pm[n], ddof=1))
                             for n in ("pred_full", "M2", "effect")},
    }
    return out


def main():
    sp_old, n_old = same_prompt()
    adm = admissible_set()
    sp_adm = same_prompt_admissible(adm)
    ho = held_out()

    print("=" * 74)
    print("pooled ρ 的三口径对照")
    print("=" * 74)

    print("\n[A] 同 prompt，逐组合（%d 组）—— §5.1 的 0.675 就是这个" % n_old)
    for name in ("pred_full", "pred_last", "pred_align",
                 "M2_mean_act_task", "M3_max_activation"):
        if name in sp_old:
            v = sp_old[name]
            print(f"    {name:18s} {v['mean_combo']:+.3f} ± {v['std_combo']:.3f}")

    print("\n[B] 同 prompt，pooled（%d 对）—— §5.1 的 0.725 / 0.206 就是这个"
          % (n_old * 200))
    for name in ("pred_full", "pred_last", "pred_align",
                 "M2_mean_act_task", "M3_max_activation"):
        if name in sp_old:
            print(f"    {name:18s} {sp_old[name]['pooled']:+.3f}")

    print("\n[C] held-out，逐组合（%d 组）" % ho["per_combo_all48_n"])
    for name, v in ho["per_combo_all48"].items():
        print(f"    {name:18s} {v:+.3f}")

    print("\n[D] held-out，pooled（门控内 %d 组，%d 对）"
          % (ho["n_admit"], ho["pooled_admit38"]["pred_full"]["n_pairs"]))
    for name, v in ho["pooled_admit38"].items():
        print(f"    {name:18s} {v['pooled']:+.3f}")

    gl = ho["group_level"]
    print("\n[E] 组间层面（%d 个组合×分割点，均值再相关）" % gl["n_points"])
    for name, v in gl["spearman"].items():
        print(f"    {name:18s} {v:+.3f}")
    print("    组间 sd: " + "  ".join(
        f"{k}={v:.4f}" for k, v in gl["between_combo_sd"].items()))

    print("\n[F] 同 prompt，门控内 %d 组逐组合" % sp_adm["n"])
    for name in ("pred_full", "M2"):
        print(f"    {name:18s} {sp_adm[name]['mean_combo']:+.3f}")

    # 倍数对照
    print("\n[倍数对照]")
    a_p, a_m = sp_old["pred_full"]["mean_combo"], sp_old["M2_mean_act_task"]["mean_combo"]
    b_p, b_m = sp_old["pred_full"]["pooled"], sp_old["M2_mean_act_task"]["pooled"]
    c_p, c_m = ho["per_combo_all48"]["pred_full"], ho["per_combo_all48"]["M2"]
    d_p = ho["pooled_admit38"]["pred_full"]["pooled"]
    d_m = ho["pooled_admit38"]["M2"]["pooled"]
    print(f"    同 prompt 逐组合:  {a_p:+.3f} / {a_m:+.3f} = {a_p/a_m:.2f}x")
    print(f"    同 prompt pooled:  {b_p:+.3f} / {b_m:+.3f} = {b_p/b_m:.2f}x")
    print(f"    held-out 逐组合:  {c_p:+.3f} / {c_m:+.3f} = {c_p/c_m:.2f}x")
    print(f"    held-out pooled:  {d_p:+.3f} / {d_m:+.3f} = {d_p/d_m:.2f}x")

    out = {"same_prompt": sp_old, "n_combos_same_prompt": n_old,
           "same_prompt_admissible": sp_adm,
           "held_out": ho,
           "ratios": {"same_prompt_per_combo": a_p / a_m,
                      "same_prompt_pooled": b_p / b_m,
                      "held_out_per_combo": c_p / c_m,
                      "held_out_pooled": d_p / d_m}}
    p = BASE / "cfse_grid_measurability" / "pooled_audit.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[写出] {p}")


if __name__ == "__main__":
    main()
