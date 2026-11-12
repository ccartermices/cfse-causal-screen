# -*- coding: utf-8 -*-
"""50_cfse_attenuation.py — CFSE 测量学层: 可靠性 + 衰减校正

目的
----
E2 pilot 出现一个反直觉结果: 用「选择集上实测的逐特征效应」(oracle) 去排序,
其 held-out 相关 0.405(qwen) 反而低于平滑的一阶预测器 pred_full 的 0.604。
两种候选的差别不在"信号强度", 而在**测量噪声被衰减的次数**:

  - oracle 分数 = c_sel, 本身是 C 的含噪测量, 与 c_ev 的噪声独立
        rho(c_sel, c_ev) = rho_true * sqrt(rel_sel * rel_ev) = rho_true * rel
  - pred 分数是确定性函数, 只有目标端噪声
        rho(|pred|, c_ev) = rho_true * sqrt(rel_ev) = rho_true * sqrt(rel)

其中 rel = 同一组合两个 prompt 半样本上逐特征效应的相关 (split-half reliability)。
因此本脚本给出三个数:
  1. rel        —— 逐特征因果真值本身的测量可靠性
  2. rho_ceiling= sqrt(rel) —— 任何选择器在 n 条 prompt 上能拿到的 rho_abs 上限
  3. rho_true   —— 去衰减后的真实效度 (pred 用 sqrt(rel) 校正, oracle 用 rel 校正)

再用 Spearman-Brown 把 200-prompt 的 rel 外推到 100-prompt, 检验 pilot 的
观测 rho 是否与预测一致 —— 若一致, "oracle 输给 pred" 就确认为测量学现象而非
"实测真值没用"。

用法
----
python 50_cfse_attenuation.py                 # 自动读 cfse_heldout 与 cfse_heldout_e2b
python 50_cfse_attenuation.py --dirs cfse_heldout_e2b
"""
import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return None
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_brown_down(rel_2n):
    """把 2n 条 prompt 的可靠性降为 n 条: rel_n = rel_2n / (2 - rel_2n)."""
    if rel_2n is None or rel_2n <= 0:
        return None
    return float(rel_2n / (2.0 - rel_2n))


def spearman_brown_up(rel_n, n, target):
    """达到目标可靠性所需的 prompt 数: N = target*n*(1-r) / (r*(1-target))."""
    if rel_n is None or not (0 < rel_n < 1) or not (0 < target < 1):
        return None
    if rel_n >= target:
        return 0
    return float(target * n * (1 - rel_n) / (rel_n * (1 - target)))


def load_dir(root, tag):
    """返回 {(model, layer, task): {'n_prompts':.., 'splits':[...]}}"""
    out = {}
    pat = str(Path(root) / "output" / "phaseStep2" / (f"cfse_heldout{tag}") /
              "*" / "layer*" / "*.json")
    for f in sorted(glob.glob(pat)):
        if "summary" in Path(f).name:
            continue
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not j.get("splits"):
            continue
        key = (j["model"], int(j["layer"]), j["task"])
        out[key] = {"n_prompts": j.get("n_prompts"), "tag": tag,
                    "splits": j["splits"]}
    return out


def analyze(rec):
    """单个 model/layer/task 的可靠性 + 衰减校正。

    可靠性需要逐特征效应数组 (c_sel/c_eval); 早期 pilot 未落盘时 rel=None,
    此时仍返回观测 rho, 供跨尺度检验使用。
    """
    rels, rows = [], []
    for sp in rec["splits"]:
        cs, ce = sp.get("c_sel") or [], sp.get("c_eval") or []
        rel = sp.get("reliability_rho")
        if rel is None and cs and ce and len(cs) == len(ce):
            rel = spearman(cs, ce)
        if rel is not None:
            rels.append(float(rel))
        f = sp.get("feature") or {}
        row = {"split": sp.get("split"), "rel": rel,
               "n_sel": len(cs), "n_ev": len(ce)}
        for sel in ("pred_full", "M2", "M3", "oracle", "random"):
            d = f.get(sel) or {}
            row[sel] = d.get("rho_abs")
            row[f"{sel}_ndcg20"] = d.get("ndcg@20")
        if any(row.get(s) is not None for s in ("pred_full", "oracle")):
            rows.append(row)

    if not rows:
        return None
    rel = float(np.mean(rels)) if rels else None
    # 半样本大小 = 每条 split 的 eval prompt 数 = n_prompts // 2
    n_half = int(rec["n_prompts"] // 2) if rec.get("n_prompts") else None

    out = {"n_prompts": rec.get("n_prompts"), "n_splits": len(rows),
           "rel_mean": rel, "rel_splits": rels,
           "rel_n_half": n_half,
           "ceiling_rho_abs": (float(np.sqrt(max(rel, 0.0)))
                               if rel is not None else None)}
    if n_half and rel is not None:
        out["rel_at_half_n"] = spearman_brown_down(rel)
        out["n_for_rel"] = {str(t): spearman_brown_up(rel, n_half, t)
                            for t in (0.7, 0.8, 0.9)}

    for sel in ("pred_full", "M2", "M3", "oracle", "random"):
        vals = [r.get(sel) for r in rows if r.get(sel) is not None]
        if not vals:
            continue
        obs = float(np.mean(vals))
        # pred/M2/M3/random 是确定性分数: 只受目标端噪声 -> /sqrt(rel)
        # oracle 本身是含噪测量: 双端噪声 -> /rel
        if rel is None:
            out[sel] = {"rho_obs": obs, "denom": None, "rho_true": None}
            continue
        denom = rel if sel == "oracle" else np.sqrt(max(rel, 1e-9))
        out[sel] = {"rho_obs": obs, "denom": float(denom),
                    "rho_true": float(obs / denom) if denom > 0 else None}
    out["stability"] = stability(rec)
    return out


def stability(rec):
    """跨 split 的集合重叠 (Jaccard@N): 换一批 prompt 后选出的集合是否还一样。"""
    per = {}
    splits = [sp for sp in rec["splits"] if sp.get("sets")]
    for sel in ("pred_full", "M2", "M3", "oracle", "random"):
        for N in ("10", "20", "50"):
            sets = []
            for sp in splits:
                d = (sp["sets"].get(sel) or {}).get(N) or {}
                idx = d.get("feature_idx")
                if idx:
                    sets.append(set(int(x) for x in idx))
            if len(sets) < 2:
                continue
            js = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    u = len(sets[i] | sets[j])
                    js.append(len(sets[i] & sets[j]) / u if u else None)
            js = [x for x in js if x is not None]
            if js:
                per.setdefault(sel, {})[N] = {
                    "jaccard_mean": float(np.mean(js)),
                    "n_pairs": len(js)}
    return per


def cross_scale_check(small, big):
    """用大样本可靠性预测小样本的观测 rho, 检验两尺度是否同一衰减机制。"""
    if not small or not big:
        return None
    rel_big = big["rel_mean"]
    rel_small_pred = spearman_brown_down(rel_big)
    if not rel_small_pred:
        return None
    rows = []
    for sel, kind in (("pred_full", "sqrt"), ("oracle", "full")):
        a = (small.get(sel) or {}).get("rho_obs")
        b = (big.get(sel) or {})
        if a is None or b.get("rho_true") is None:
            continue
        denom = (np.sqrt(max(rel_small_pred, 1e-9)) if kind == "sqrt"
                 else rel_small_pred)
        pred = b["rho_true"] * denom
        rows.append({"selector": sel, "rho_obs_small": a,
                     "rho_pred_from_big": float(pred),
                     "abs_err": float(abs(a - pred))})
    return {"rel_big": rel_big, "rel_small_est": rel_small_pred, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["cfse_heldout_e2b"])
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    ref = load_dir(base, "")          # n=100 pilot
    big = {}
    for tag in args.dirs:
        big.update(load_dir(base, tag))

    res = {"timestamp": datetime.now().isoformat(), "pilot": {}, "big": {},
           "cross_scale": {}}

    for key in sorted(set(ref) | set(big)):
        name = f"{key[0]}_L{key[1]}_{key[2]}"
        a = analyze(ref[key]) if key in ref else None
        b = analyze(big[key]) if key in big else None
        if a:
            res["pilot"][name] = a
        if b:
            res["big"][name] = b
        cs = cross_scale_check(a, b)
        if cs:
            res["cross_scale"][name] = cs

    out_dir = base / "output" / "phaseStep2" / "cfse_attenuation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "attenuation.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    L = ["# CFSE 测量学层：可靠性、天花板与衰减校正\n",
         f"- 生成时间：{res['timestamp']}",
         "- `rel`：逐特征效应在两个 prompt 半样本上的 Spearman 相关（split-half 可靠性）",
         "- `ceiling`：`sqrt(rel)`，任何选择器在该 prompt 数下能取得的 `rho_abs` 上限",
         "- `rho_true`：去衰减效度；pred/M2 除以 `sqrt(rel)`。oracle 的分数就是含噪实测值，",
         "其去衰减值按定义恒为 1，属循环，故只列观测值（= rel）。\n"]

    for group, title in (("big", "大样本 (E2b)"), ("pilot", "小样本 (E2 pilot)")):
        if not res[group]:
            continue
        L.append(f"## {title}\n")
        L.append("| combo | n_total | rel | ceiling | pred obs→true | M2 obs→true | oracle (obs = rel) |")
        L.append("|---|---|---|---|---|---|---|")
        for name, d in res[group].items():
            def cell(sel):
                x = d.get(sel)
                if not x:
                    return "—"
                if sel == "oracle":
                    # oracle 分数=c_sel, 其 held-out 相关按定义恒等于 rel,
                    # 去衰减值必为 1, 属循环, 不展示为结果
                    return f"{x['rho_obs']:.3f} (= rel)"
                t = x.get("rho_true")
                return (f"{x['rho_obs']:.3f} → {t:.3f}" if t is not None
                        else f"{x['rho_obs']:.3f}")
            rel_s = f"{d['rel_mean']:.3f}" if d.get("rel_mean") is not None else "—"
            ceil_s = (f"{d['ceiling_rho_abs']:.3f}"
                      if d.get("ceiling_rho_abs") is not None else "—")
            L.append(f"| {name} | {d.get('n_prompts')} | {rel_s} | "
                     f"{ceil_s} | {cell('pred_full')} | "
                     f"{cell('M2')} | {cell('oracle')} |")
        L.append("")

    stab_rows = []
    for group, label in (("big", "E2b"), ("pilot", "pilot")):
        for name, d in (res.get(group) or {}).items():
            st = d.get("stability") or {}
            for sel, per in st.items():
                stab_rows.append((label, name, sel, per))
    if stab_rows:
        L.append("## 跨 split 稳定性 (Jaccard@N)\n")
        L.append("| 数据 | combo | selector | @10 | @20 | @50 | 对数 |")
        L.append("|---|---|---|---|---|---|---|")
        for label, name, sel, per in stab_rows:
            def j(N):
                x = per.get(N)
                return f"{x['jaccard_mean']:.3f}" if x else "—"
            npair = max((per[N]["n_pairs"] for N in per), default=0)
            L.append(f"| {label} | {name} | {sel} | {j('10')} | {j('20')} | "
                     f"{j('50')} | {npair} |")
        L.append("")

    need = [(n, d) for n, d in res["big"].items() if d.get("n_for_rel")]
    if need:
        L.append("## 可靠达到标所需的 prompt 数（Spearman-Brown 外推）\n")
        L.append("| combo | rel(n) | n | → rel=0.7 | → rel=0.8 | → rel=0.9 |")
        L.append("|---|---|---|---|---|---|")
        for name, d in need:
            nr = d["n_for_rel"]

            def f(t):
                v = nr.get(t)
                return "已达" if v == 0 else (f"{v:.0f}" if v else "—")
            L.append(f"| {name} | {d['rel_mean']:.3f} | {d.get('rel_n_half')} | "
                     f"{f('0.7')} | {f('0.8')} | {f('0.9')} |")
        L.append("")

    if res["cross_scale"]:
        L.append("## 跨尺度检验（用 E2b 的 rho_true 预测 pilot 的观测 rho）\n")
        L.append("| combo | selector | pilot obs | 由大样本预测 | 误差 |")
        L.append("|---|---|---|---|---|")
        for name, cs in res["cross_scale"].items():
            L.append(f"- {name}: rel(大样本)={cs['rel_big']:.3f} → "
                     f"rel(小样本估计)={cs['rel_small_est']:.3f}")
            for r in cs["rows"]:
                L.append(f"| {name} | {r['selector']} | {r['rho_obs_small']:.3f} | "
                         f"{r['rho_pred_from_big']:.3f} | {r['abs_err']:.3f} |")
            L.append("")

    (out_dir / "attenuation.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
