# -*- coding: utf-8 -*-
"""66 — E14：把 κ 从"门控"升级为"预测器"——测量噪声生成模型

问题
----
E3（§17）证明 κ = sd_between/sd_within 与 rel 的秩相关 0.934（n=14），
但**必要非充分**：qwen3.5-2b L22 的 κ=0.445 > L8 的 0.339，rel 反而更低
（0.681 < 0.744）。原因是 κ 只用了二阶矩（均值的离散度 / 特征内 sd），
丢掉了两件东西：

  1. **形状**：|μ_f| 的分布尾部（少数特征主导排序 vs 均匀分布）；
  2. **异方差**：大效应特征是否同时伴随大的 prompt 间波动。

做法
----
不做回归拟合，而是写一个**生成模型**：把每特征的半样本均值看作
`μ̂_f = μ_f + (σ_f/√h)·z`，从 E 里估 μ_f、σ_f，模拟两个互斥半样本，
算 |mean| 的 Spearman，取均值即预测的 split-half 信度 rel_pred。
这是"如果波动只是独立采样噪声，信度应该等于多少"的零模型。

- 若 rel_obs ≈ rel_pred：该组合的不可测性完全由采样噪声解释，κ 之外的形状
  信息没有额外作用，κ 已够用；
- 若 rel_obs > rel_pred：prompt 间存在跨特征的共同结构（提示主效应），
  半样本的偏差被共享，信度高于独立噪声预测；
- 若 rel_obs < rel_pred：效应在 prompt 间有**交互**（特征×prompt），
  独立噪声模型低估了真实波动。

同时给出三个形状量，检验它们能否在 κ 之外补充解释力：
  - tail   = median|μ_f| / sd(μ_f)      （分布形状；高斯≈0.674）
  - het    = corr(log|μ_f|, log σ_f)    （异方差；>0 说明大效应更不稳）
  - frac2  = #{|μ_f|/SE_f > 2} / F      （可分辨特征占比）

评估：14 个组合上算 Spearman 与留一交叉验证 MAE，比较
  A: 只用 κ        B: 只用 rel_pred（零拟合）
  C: log κ + het   D: rel_pred + het

输出
----
output/phaseStep2/cfse_shapemodel/shapemodel.json / shapemodel.md
_paper_build/figures/fig7_measurability_model.{png,pdf}
"""
import json
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "output" / "phaseStep2" / "cfse_shapemodel"
FIG = BASE / "_paper_build" / "figures"
RES = BASE / "output" / "phaseStep2" / "cfse_resolution"
KAP = BASE / "output" / "phaseStep2" / "cfse_kappa"

_spec = importlib.util.spec_from_file_location("m64", BASE / "64_e3_figure.py")
_m64 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m64)
rel100 = _m64.rel100
_spearman = _m64._spearman

COLOR = _m64.COLOR
MARK = _m64.MARK

# 14 个组合（与 §17.2 表一致）
POINTS = [
    ("qwen3.5-2b", 8, "ioi", "E3"),
    ("qwen3.5-2b", 16, "ioi", "E10"),
    ("qwen3.5-2b", 22, "ioi", "E3"),
    ("qwen3.5-2b", 16, "gt", "E10"),
    ("qwen3.5-2b", 16, "natural", "E10"),
    ("qwen3.5-0.8b", 16, "ioi", "E3"),
    ("qwen2.5-7b-instruct", 18, "ioi", "E3"),
    ("pythia-1.4b", 16, "ioi", "E10"),
    ("pythia-1.4b", 16, "natural", "E10"),
    ("pythia-2.8b", 21, "ioi", "E3"),
    ("pythia-6.9b", 21, "ioi", "E10"),
    ("minicpm5-1b", 16, "ioi", "E3"),
    ("tinyllama-1.1b", 14, "ioi", "E3"),
    ("phi-3.5-mini", 21, "gt", "E3"),
]


def E_path(m, layer, task, src):
    return (RES if src == "E10" else KAP) / m / f"layer{layer}" / f"{task}_E_f32.npz"


def collect_points():
    """自动收集：E10 主表 JSON + E3/E11 的 kappa_scan.json（新模型无需手改）。"""
    pts, seen = [], set()
    e10 = BASE / "output" / "phaseStep2" / "cfse_e10"
    default_layer = {"qwen3.5-2b": 16, "pythia-1.4b": 16, "pythia-6.9b": 21}
    for f in sorted(e10.glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        layer = d.get("layer") or default_layer.get(d["model"])
        if layer is None:
            continue
        key = (d["model"], layer, d["task"])
        if key in seen or not E_path(*key, "E10").exists():
            continue
        seen.add(key)
        pts.append((*key, "E10"))
    ks = KAP / "kappa_scan.json"
    if ks.exists():
        for r in json.load(open(ks, encoding="utf-8")).get("runs", []):
            key = (r["model"], r["layer"], r["task"])
            if key in seen or not E_path(*key, "E3").exists():
                continue
            seen.add(key)
            pts.append((*key, "E3"))
    return pts


def kappa_of(E):
    sb = float(E.mean(axis=1).std(ddof=1))
    sw = float(E.std(axis=1, ddof=1).mean())
    return sb / sw if sw > 0 else np.nan


def noise_model(E, half=100, n_draws=200, seed=0):
    """独立采样噪声下的 split-half 信度预测（不经任何拟合）。"""
    mu = E.mean(axis=1)
    sd = E.std(axis=1, ddof=1)
    se = np.maximum(sd / np.sqrt(half), 1e-12)
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_draws):
        x1 = np.abs(mu + se * rng.randn(len(mu)))
        x2 = np.abs(mu + se * rng.randn(len(mu)))
        r = _spearman(x1, x2)
        if r is not None:
            vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def shape_terms(E, half=100):
    mu = E.mean(axis=1)
    sd = E.std(axis=1, ddof=1)
    se = np.maximum(sd / np.sqrt(half), 1e-12)
    amu = np.abs(mu)
    tail = float(np.median(amu) / max(np.std(mu, ddof=1), 1e-12))
    eps = 1e-9
    het = float(np.corrcoef(np.log(amu + eps), np.log(sd + eps))[0, 1])
    frac2 = float(np.mean(amu / se > 2.0))
    return {"tail": tail, "het": het, "frac2": frac2}


def loo_mae(y, X, ridge=1e-6):
    """线性回归的留一交叉验证 MAE（X 已含常数列）。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n = len(y)
    preds = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        XtX = X[m].T @ X[m] + ridge * np.eye(X.shape[1])
        w = np.linalg.solve(XtX, X[m].T @ y[m])
        preds.append(float(X[i] @ w))
    preds = np.asarray(preds)
    return float(np.mean(np.abs(preds - y))), preds


def main():
    rows = []
    for m, layer, task, src in collect_points():
        p = E_path(m, layer, task, src)
        if not p.exists():
            print(f"[缺] {p}")
            continue
        E = np.load(p)["E"].astype(float)
        rel_obs = rel100(E)
        rows.append({"model": m, "layer": layer, "task": task, "src": src,
                     "kappa": kappa_of(E), "rel_obs": rel_obs,
                     "rel_pred": noise_model(E), **shape_terms(E)})

    if not rows:
        raise SystemExit("没有可用数据点")

    y = np.array([r["rel_obs"] for r in rows])
    kap = np.array([r["kappa"] for r in rows])
    prd = np.array([r["rel_pred"] for r in rows])
    het = np.array([r["het"] for r in rows])

    def stat(name, x):
        r = _spearman(x, y)
        bias = float(np.mean(x - y))
        mae = float(np.mean(np.abs(x - y)))
        return {"name": name, "spearman": r, "mae": mae, "bias": bias}

    stats = [stat("kappa", kap), stat("noise_model", prd)]

    ones = np.ones_like(kap)
    mae_k, _ = loo_mae(y, np.c_[ones, np.log(kap)])
    mae_kh, _ = loo_mae(y, np.c_[ones, np.log(kap), het])
    mae_ph, _ = loo_mae(y, np.c_[ones, prd, het])
    mae_p, _ = loo_mae(y, np.c_[ones, prd])
    # 校准斜率（观测 ~ 预测）
    A = np.c_[ones, prd]
    w = np.linalg.solve(A.T @ A + 1e-9 * np.eye(2), A.T @ y)
    resid = y - prd

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shapemodel.json").write_text(json.dumps(
        {"points": rows, "stats": stats,
         "loo_mae": {"kappa": mae_k, "log_kappa+het": mae_kh,
                     "noise_model": mae_p, "noise_model+het": mae_ph},
         "calibration": {"intercept": float(w[0]), "slope": float(w[1])}},
        ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- markdown ----------
    L = ["# E14 — 测量噪声生成模型：能否用 κ 预测 rel", "",
         f"点数 n={len(rows)}｜rel 口径：half=100、20 次重采样（与 fig6 完全一致）｜"
         "rel_pred 为**零拟合**预测", "",
         "| model | L | task | κ | tail | het | frac2 | rel_obs | rel_pred | 残差 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["kappa"]):
        L.append(f"| {r['model']} | {r['layer']} | {r['task']} | {r['kappa']:.3f} | "
                 f"{r['tail']:.3f} | {r['het']:+.3f} | {r['frac2']:.2f} | "
                 f"{r['rel_obs']:.3f} | {r['rel_pred']:.3f} | {r['rel_obs'] - r['rel_pred']:+.3f} |")
    L += ["", "## 预测力", "",
          "| 预测器 | Spearman | MAE | 偏差(预测−观测) |", "|---|---|---|---|"]
    for s in stats:
        L.append(f"| {s['name']} | {s['spearman']:.3f} | {s['mae']:.3f} | {s['bias']:+.3f} |")
    L += ["", "## 留一交叉验证（线性模型，样本外 MAE）", "",
          "| 模型 | LOO MAE |", "|---|---|",
          f"| log κ | {mae_k:.3f} |",
          f"| log κ + het | {mae_kh:.3f} |",
          f"| rel_pred | {mae_p:.3f} |",
          f"| rel_pred + het | {mae_ph:.3f} |",
          "", f"标定：obs = {w[0]:.3f} + {w[1]:.3f} × pred", ""]
    md = OUT / "shapemodel.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # ---------- figure ----------
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    lim = (-0.05, 1.02)
    ax.plot(lim, lim, ls="--", lw=1, color="#999", zorder=1)
    for r in rows:
        ax.scatter(r["rel_pred"], r["rel_obs"], s=64, alpha=0.9, zorder=3,
                   color=COLOR.get(r["model"], "#555"), marker=MARK.get(r["task"], "o"),
                   edgecolor="white", linewidth=0.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("predicted rel (independent noise model)", fontsize=9)
    ax.set_ylabel("observed rel (100/100 split-half)", fontsize=9)
    ax.set_title(f"(a) zero-fit prediction: $\\rho$={stats[1]['spearman']:.3f}, "
                 f"MAE={stats[1]['mae']:.3f}", fontsize=10)
    ax.grid(alpha=0.25, ls=":")
    # 模型配色图例（只列出现过的模型）+ 任务形状图例
    seen = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    handles = [plt.Line2D([], [], ls="", marker="o", markersize=6,
                          markerfacecolor=COLOR.get(mo, "#555"), markeredgecolor="white",
                          label=mo) for mo in seen]
    ax.legend(handles=handles, fontsize=6, ncol=2, loc="lower right",
              frameon=False, handletextpad=0.3, columnspacing=0.8, labelspacing=0.25)
    tasks = sorted({r["task"] for r in rows})
    ax.text(0.02, 0.02, "   ".join(f"{MARK.get(t, 'o')} {t}" for t in tasks),
            transform=ax.transAxes, fontsize=6.5, color="#555", ha="left", va="bottom")

    ax = axes[1]
    ax.axhline(0, color="#999", lw=1)
    for r in rows:
        ax.scatter(r["kappa"], r["rel_obs"] - r["rel_pred"], s=64, alpha=0.9, zorder=3,
                   color=COLOR.get(r["model"], "#555"), marker=MARK.get(r["task"], "o"),
                   edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"measurability ratio $\kappa$", fontsize=9)
    ax.set_ylabel("residual  obs − pred", fontsize=9)
    ax.set_title("(b) where the iid noise model breaks", fontsize=10)
    ax.grid(alpha=0.25, ls=":")

    # 只在 (b) 标出残差最大与最小的两个点
    order = sorted(rows, key=lambda r: r["rel_obs"] - r["rel_pred"])
    for r in (order[0], order[-1]):
        ax.annotate(f"{r['model']}/L{r['layer']}",
                    (r["kappa"], r["rel_obs"] - r["rel_pred"]),
                    textcoords="offset points", xytext=(8, 4), fontsize=7, color="#333")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig7_measurability_model.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "shapemodel.png", dpi=170, bbox_inches="tight", facecolor="white")
    print(f"\n[图] {FIG / 'fig7_measurability_model.png'}")
    print(f"[写出] {md}")


if __name__ == "__main__":
    main()
