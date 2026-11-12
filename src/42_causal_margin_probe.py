# -*- coding: utf-8 -*-
"""Step A: 一阶因果边距预测器 —— 离线判定

动机
----
36 号实验已证明: 常用 SAE 指标 (M1..M6) 几乎无法预测单特征干预的因果效应量
(pooled spearman(M2, C_ablate) ≈ +0.23)。原因是这些指标只测「特征被激活的幅度」,
完全不含「该特征的 decoder 方向是否与下游任务读出方向对齐」的信息。

一阶假设
--------
在层 L 的 block 输出处, 单特征 ablate 的干预为
    delta(t) = -s_f(t) * (W_dec[:,f] ⊙ std)          (36 make_feature_hook, mode=ablate)
把行为 B = logit(correct) - logit(distractor) 对 h 做一阶展开:
    ΔB ≈ Σ_t <∂B/∂h(t), delta(t)> = -Σ_t s_f(t) * (W_dec[:,f] ⊙ std)·g(t)
其中 g(t) = ∂B/∂h(t) 只需**一次反向传播**就能拿到 (与 f 无关)。

于是可定义**零干预**可算的预测器 (只用 prompt 上的一次前向+反向, 不做任何干预):
    pred_signed[f] = -mean_prompt[ Σ_t s_f(t) * (W_dec[:,f] ⊙ std)·g(t) ]
    pred_mag[f]    = |pred_signed[f]|

本脚本不训练任何东西, 只判定: pred_mag 与实测 C_ablate = |B_ablate - b0| 的
spearman 是否显著超过 M2 的 ~+0.23。

变体
----
  pred_full  全位置求和 (理论公式)
  pred_last  只用末位 token (任务读出通常集中在末位)
  pred_align 纯对齐项 |(W_dec[:,f] ⊙ std)·ḡ|, ḡ = 平均梯度, 不含激活幅度

资源: CPU 严格节流 (OMP/MKL=1, torch threads=1), 串行, 按模型只加载一次基座,
      独显 cuda:0, 不打扰同机其他实验。
输出: output/phaseStep2/causal_margin/{model}/layer{L}/{task}.json
      output/phaseStep2/causal_margin/stepA_offline.json
      output/phaseStep2/causal_margin/stepA_offline.md
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from importlib import import_module

torch.set_num_threads(1)

from model_utils import (load_config, resolve_model_cfg, model_needs_fp8,
                         dequantize_fp8_model)

_m05 = import_module("05_train_text_sae")
LogSaver = _m05.LogSaver

_m27 = import_module("27_h6_semantic_injection")
find_decoder_layers = _m27.find_decoder_layers
load_model_for_generation = _m27.load_model_for_generation

_m36 = import_module("36_causal_validity")
TASK_BUILDERS = _m36.TASK_BUILDERS
_encode_with_prefix = _m36._encode_with_prefix
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae

_m34 = import_module("34_step1_metric_validity")
spearman = _m34.spearman
pearson = _m34.pearson

_save = _m36._save

METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
           "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]


# ----------------------------------------------------------------------------
# 组合发现: 扫描 36 的落盘结果, 只保留有效组合
# ----------------------------------------------------------------------------
OUT_TAG = ""  # 由 --out_tag 设置; 同时决定读取 36 与写出 42 的目录后缀


def discover_combos(base_dir, models, tasks):
    root = base_dir / "output" / "phaseStep2" / f"causal_validity{OUT_TAG}"
    combos = []
    for m in models:
        mdir = root / m
        if not mdir.is_dir():
            continue
        for ldir in sorted(mdir.glob("layer*")):
            try:
                layer = int(ldir.name.replace("layer", ""))
            except ValueError:
                continue
            for task in tasks:
                f = ldir / f"{task}.json"
                if not f.exists():
                    continue
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        rec = json.load(fh)
                except Exception:
                    continue
                if rec.get("error") or rec.get("at_chance"):
                    continue
                feats = rec.get("features") or []
                feats = [r for r in feats if r.get("C_ablate") is not None]
                if len(feats) < 5:
                    continue
                combos.append({"model": m, "layer": layer, "task": task,
                               "block_idx": rec.get("block_idx"),
                               "b0": rec.get("b0"), "b0_acc": rec.get("b0_acc"),
                               "src": str(f), "n_features": len(feats)})
    return combos


def load_annotations(src_path):
    with open(src_path, "r", encoding="utf-8") as fh:
        rec = json.load(fh)
    rows = []
    for r in rec.get("features") or []:
        if r.get("C_ablate") is None:
            continue
        rows.append({
            "feature_idx": int(r["feature_idx"]),
            "C_ablate": float(r["C_ablate"]),
            "B_ablate": float(r["B_ablate"]) if r.get("B_ablate") is not None else None,
            "M2_mean_act_task": float(r["M2_mean_act_task"]) if r.get("M2_mean_act_task") is not None else None,
            "M1_act_freq_task": r.get("M1_act_freq_task"),
            "M3_max_act_task": r.get("M3_max_act_task"),
            "M4_recon_mse_gain": r.get("M4_recon_mse_gain"),
            "M5_ctx_coherence": r.get("M5_ctx_coherence"),
            "M6_corpus_freq": r.get("M6_corpus_freq"),
        })
    return rec, rows


# ----------------------------------------------------------------------------
# 一次前向 + 一次反向: 得到 g = dB/dh 与 sparse
# ----------------------------------------------------------------------------
def gradient_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t, device,
                   batch_size=16, feat_idx=None):
    """返回 (g_sum_list, ...) 与逐特征预测。

    pred_full[b,f], pred_last[b,f], g_bar (D,), b0_here (list)
    """
    D = sae.decoder.weight.shape[0]
    W = sae.decoder.weight.detach().float()            # (D, hidden_dim)
    if feat_idx is None:
        Wsel = W
    else:
        idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
        Wsel = W.index_select(1, idx_t)                # (D, F)
    Wsc = Wsel * std_t.view(-1, 1)                     # (D, F)

    F = Wsc.shape[1]
    b0_list = []
    term_full_sum = np.zeros(F, dtype=np.float64)
    term_last_sum = np.zeros(F, dtype=np.float64)
    g_bar_sum = np.zeros(D, dtype=np.float64)
    n_prompt = 0
    n_tok = 0

    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
        cap = {}

        def hk(module, args, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            cap["h"] = hs

        handle = layer_mod.register_forward_hook(hk)
        try:
            logits = model(input_ids=input_ids, attention_mask=attn).logits[:, -1, :].float()
        finally:
            handle.remove()
        h = cap["h"]
        cid = torch.tensor([it["correct_id"] for it in chunk], device=logits.device)
        did = torch.tensor([it["distractor_id"] for it in chunk], device=logits.device)
        B = (logits.gather(1, cid[:, None]) - logits.gather(1, did[:, None])).squeeze(1)
        b0_list.extend(B.detach().cpu().tolist())

        g = torch.autograd.grad(B.sum(), h)[0].float()          # (B,T,D)
        mask = attn.unsqueeze(-1).float()
        g = g * mask

        with torch.no_grad():
            normed = (h.detach().float() - mean_t) / std_t
            sparse, _ = sae.encode(normed)
            if feat_idx is None:
                s_sel = sparse
            else:
                s_sel = sparse.index_select(-1, torch.tensor(feat_idx, dtype=torch.long, device=device))

            # 全位置: (B,F)
            tf = torch.einsum("btd,df,btf->bf", g, Wsc, s_sel)
            # 末位 token (left padding -> 位置 -1 恒为真实 token): (B,F)
            tl = torch.einsum("bd,df,bf->bf", g[:, -1, :], Wsc, s_sel[:, -1, :])

            term_full_sum += tf.sum(0).cpu().double().numpy()
            term_last_sum += tl.sum(0).cpu().double().numpy()
            g_bar_sum += g.sum(dim=(0, 1)).cpu().double().numpy()
            n_prompt += int(tf.shape[0])
            n_tok += int(mask.sum().item())
        del g, logits, B, h, cap, sparse, s_sel, tf, tl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pred_full = -(term_full_sum / max(n_prompt, 1))            # (F,)
    pred_last = -(term_last_sum / max(n_prompt, 1))
    g_bar = g_bar_sum / max(n_tok, 1)
    Wsc_np = Wsc.detach().cpu().double().numpy()
    pred_align = np.abs(Wsc_np.T @ g_bar)                      # (F,)
    pred_align_signed = -(Wsc_np.T @ g_bar)
    return {
        "pred_full": pred_full, "pred_last": pred_last,
        "pred_align": pred_align, "pred_align_signed": pred_align_signed,
        "b0_here": float(np.mean(b0_list)), "b0_here_list_n": len(b0_list),
    }


# ----------------------------------------------------------------------------
# fp8 反量化 (fp8 Triton 内核无 autograd, 无法求 dB/dh)
# ----------------------------------------------------------------------------
def fp8_dequant_selfcheck(model, tokenizer, device):
    """把模型 fp8 权重反量化为 bf16, 并用同一 prompt 校验反量化前后 logits 一致。

    fp8 前向对激活做动态量化, 与 bf16 前向本就有小幅数值差, 故只校验「量级一致」。
    返回反量化后的模块数。
    """
    probe = "When Mary and John went to the store, John gave a drink to"
    enc = tokenizer([probe], return_tensors="pt").to(device)
    with torch.no_grad():
        before = model(**enc).logits[:, -1, :].float().cpu()
    n = dequantize_fp8_model(model, torch.bfloat16)
    with torch.no_grad():
        after = model(**enc).logits[:, -1, :].float().cpu()
    diff = (before - after).abs().max().item()
    scale = max(before.abs().max().item(), 1e-6)
    print(f"  [fp8] 反量化 {n} 个模块 -> bf16; logits 最大绝对差={diff:.4g} "
          f"(相对={diff / scale:.3%}); |logits|max before={before.abs().max().item():.3f} "
          f"after={after.abs().max().item():.3f}")
    if diff / scale > 0.5:
        print("  [fp8][警告] 反量化前后 logits 差异过大, 结果可能不可靠")
    return n


# ----------------------------------------------------------------------------
# 单组合
# ----------------------------------------------------------------------------
def run_one(args, config, base_dir, m, combo, tag):
    layer, task = combo["layer"], combo["task"]
    out_dir = (base_dir / "output" / "phaseStep2" / f"causal_margin{OUT_TAG}"
               / m / f"layer{layer}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"
    resume_file = out_dir / f"resume_42_{task}.json"
    if out_file.exists() and resume_file.exists() and not args.force:
        try:
            with open(resume_file, "r", encoding="utf-8") as fh:
                if json.load(fh).get("status") == "completed":
                    print(f"  [跳过] {m} L{layer} {task} 已完成")
                    return json.load(open(out_file, encoding="utf-8"))
        except Exception:
            pass

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rec36, ann = load_annotations(combo["src"])
    feat_idx = [r["feature_idx"] for r in ann]
    result = {"model": m, "layer": layer, "task": task,
              "timestamp": datetime.now().isoformat(),
              "block_idx": combo["block_idx"], "n_features": len(feat_idx),
              "b0_36": combo["b0"], "b0_acc_36": combo["b0_acc"],
              "n_prompts": args.n_prompts}

    loaded = load_per_layer_sae(base_dir, m, layer, device)
    if loaded is None:
        result["error"] = "缺少 per-layer SAE"
        _save(out_file, result)
        return result
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    model, tokenizer, layer_mod = tag["model"], tag["tokenizer"], tag["layer_mod"][layer]

    items, task_meta = TASK_BUILDERS[task](tokenizer, args.n_prompts, seed=args.seed)
    if not items:
        result["error"] = f"任务构造失败: {task_meta}"
        _save(out_file, result)
        return result
    result["task_meta_n"] = len(items)

    probe = gradient_probe(model, tokenizer, items, layer_mod, sae, mean_t, std_t,
                           device, args.batch_size, feat_idx)
    pred_full = probe["pred_full"]
    pred_last = probe["pred_last"]
    pred_align = probe["pred_align"]

    b0 = combo["b0"] if combo["b0"] is not None else probe["b0_here"]
    result["b0_here"] = probe["b0_here"]
    result["b0_abs_diff_vs_36"] = abs(probe["b0_here"] - (combo["b0"] or 0.0))

    C = np.array([r["C_ablate"] for r in ann], dtype=np.float64)
    dB = np.array([(r["B_ablate"] - b0) if r["B_ablate"] is not None else np.nan
                   for r in ann], dtype=np.float64)

    # ---- 相关性 (幅度: |pred| vs C) ----
    corr = {}
    for name, p in (("pred_full", pred_full), ("pred_last", pred_last),
                    ("pred_align", pred_align)):
        corr[name] = {"spearman": spearman(np.abs(p).tolist(), C.tolist())}
    for mm in METRICS:
        xs = [r[mm] for r in ann if r.get(mm) is not None]
        ys = [r["C_ablate"] for r in ann if r.get(mm) is not None]
        corr[mm] = {"spearman": spearman(xs, ys), "n": len(xs)}

    # ---- 有符号: signed pred vs ΔB ----
    ok = ~np.isnan(dB)
    signed = {}
    for name, p in (("pred_full", pred_full), ("pred_last", pred_last),
                    ("pred_align_signed", probe["pred_align_signed"])):
        signed[name] = {
            "pearson_vs_dB": pearson(p[ok].tolist(), dB[ok].tolist()),
            "spearman_vs_dB": spearman(p[ok].tolist(), dB[ok].tolist()),
            "sign_acc": float(np.mean((p[ok] > 0) == (dB[ok] > 0))),
        }

    # ---- 落盘逐特征 ----
    rows = []
    for i, r in enumerate(ann):
        row = {
            "feature_idx": r["feature_idx"], "C_ablate": r["C_ablate"],
            "dB_measured": float(dB[i]) if not np.isnan(dB[i]) else None,
            "pred_full_v": float(pred_full[i]), "pred_full": float(abs(pred_full[i])),
            "pred_last_v": float(pred_last[i]), "pred_last": float(abs(pred_last[i])),
            "pred_align": float(pred_align[i]),
        }
        for mm in METRICS:
            row[mm] = r.get(mm)
        rows.append(row)
    result["correlations"] = corr
    result["signed"] = signed
    result["features"] = rows
    _save(out_file, result)
    _save(resume_file, {"status": "completed", "completed_at": datetime.now().isoformat()})
    top = max(rows, key=lambda r: r["pred_full"]) if rows else None
    print(f"  [预测] spearman: full={corr['pred_full']['spearman']:.3f} "
          f"last={corr['pred_last']['spearman']:.3f} align={corr['pred_align']['spearman']:.3f} "
          f"| M2={corr['M2_mean_act_task']['spearman']:.3f} | "
          f"sign_acc(full)={signed['pred_full']['sign_acc']:.3f}"
          + (f" | top-pred feat={top['feature_idx']}" if top else ""))
    return result


# ----------------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------------
def _pooled(records, xkey, ykey):
    xs_all, ys_all = [], []
    per_combo = []
    for r in records:
        rows = [row for row in (r.get("features") or [])
                if row.get(xkey) is not None and row.get(ykey) is not None]
        if len(rows) < 5:
            continue
        xs = np.array([row[xkey] for row in rows], dtype=np.float64)
        ys = np.array([row[ykey] for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
            continue
        sx, sy = xs.std(), ys.std()
        if sx <= 1e-12 or sy <= 1e-12:
            continue
        xs_all.append((xs - xs.mean()) / sx)
        ys_all.append((ys - ys.mean()) / sy)
        per_combo.append(spearman(xs.tolist(), ys.tolist()))
    if not xs_all:
        return None
    X = np.concatenate(xs_all)
    Y = np.concatenate(ys_all)
    return {"pooled_spearman": spearman(X.tolist(), Y.tolist()),
            "pooled_pearson": pearson(X.tolist(), Y.tolist()),
            "mean_combo_spearman": float(np.mean(per_combo)),
            "std_combo_spearman": float(np.std(per_combo)),
            "n_combos": len(per_combo), "n_features": int(len(X))}


def summarize(records, out_dir, meta):
    pairs = {"pred_full": "C_ablate", "pred_last": "C_ablate",
             "pred_align": "C_ablate"}
    summary = {"meta": meta, "pooled": {}, "per_combo": []}
    for xk in list(pairs.keys()) + METRICS:
        s = _pooled(records, xk, "C_ablate")
        if s:
            summary["pooled"][xk] = s
    for r in records:
        summary["per_combo"].append({
            "combo": f"{r['model']}|L{r['layer']}|{r['task']}",
            "n_features": r.get("n_features"),
            "b0": r.get("b0_36"), "b0_here": r.get("b0_here"),
            "spearman": {k: v.get("spearman") for k, v in (r.get("correlations") or {}).items()},
            "sign_acc_full": (r.get("signed") or {}).get("pred_full", {}).get("sign_acc"),
            "pearson_dB_full": (r.get("signed") or {}).get("pred_full", {}).get("pearson_vs_dB"),
        })
    _save(out_dir / "stepA_offline.json", summary)

    lines = ["# Step A: 一阶因果边距预测器 —— 离线判定", "",
             f"- 生成时间: {meta['timestamp']}",
             f"- 组合数: {len(records)} / 特征总数: {meta['n_features_total']}",
             f"- 口径: n_prompts={meta['n_prompts']}, seed={meta['seed']}, "
             f"实测口径 C_ablate=|B_ablate-b0| (来自 36)", "",
             "## 1. Pooled 相关 (逐组合 z-score 后合并)", "",
             "| 分数 | pooled spearman | pooled pearson | 组合均值±std | n |",
             "|---|---|---|---|---|"]
    for k, s in summary["pooled"].items():
        lines.append(f"| {k} | {s['pooled_spearman']:.3f} | {s['pooled_pearson']:.3f} | "
                     f"{s['mean_combo_spearman']:.3f}±{s['std_combo_spearman']:.3f} | {s['n_features']} |")
    lines += ["", "## 2. 逐组合 spearman(|pred| vs C_ablate)", "",
              "| 组合 | n | b0 | pred_full | pred_last | pred_align | M2 | sign_acc |",
              "|---|---|---|---|---|---|---|---|"]
    for r in summary["per_combo"]:
        sp = r["spearman"]
        lines.append(f"| {r['combo']} | {r['n_features']} | "
                     f"{(r['b0'] if r['b0'] is not None else float('nan')):.3f} | "
                     f"{sp.get('pred_full', float('nan')):.3f} | "
                     f"{sp.get('pred_last', float('nan')):.3f} | "
                     f"{sp.get('pred_align', float('nan')):.3f} | "
                     f"{sp.get('M2_mean_act_task', float('nan')):.3f} | "
                     f"{(r['sign_acc_full'] if r['sign_acc_full'] is not None else float('nan')):.3f} |")
    # 判定
    pf = summary["pooled"].get("pred_full", {}).get("pooled_spearman")
    pa = summary["pooled"].get("pred_align", {}).get("pooled_spearman")
    m2 = summary["pooled"].get("M2_mean_act_task", {}).get("pooled_spearman")
    lines += ["", "## 3. 判定", ""]
    if pf is not None and m2 is not None:
        lines.append(f"- M2 baseline pooled spearman = {m2:.3f}")
        lines.append(f"- pred_full pooled spearman = {pf:.3f} (Δ={pf - m2:+.3f})")
        if pa is not None:
            lines.append(f"- pred_align pooled spearman = {pa:.3f} (Δ={pa - m2:+.3f})")
        best = max(v for v in (pf, pa) if v is not None)
        if best >= m2 + 0.15:
            lines.append(f"- **结论: 通过** —— 一阶预测器显著优于标准指标, 进入 Step B (端到端 selector 横评)。")
        elif best >= m2 + 0.05:
            lines.append(f"- **结论: 临界** —— 有增益但幅度有限, 需 Step B 端到端验证是否有 z 提升。")
        else:
            lines.append(f"- **结论: 不通过** —— 一阶边距增益不足, 退回「评估协议/负面结果+方法学」路线。")
    else:
        lines.append("- 数据不足, 无法判定。")
    (out_dir / "stepA_offline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Step A: 一阶因果边距预测器离线判定")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--tasks", type=str, default="gt,ioi,ioi_distract")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tag", type=str, default="",
                    help="输出目录后缀, 如 _s43 -> causal_margin_s43 且读 causal_validity_s43")
    ap.add_argument("--limit", type=int, default=0, help="每模型最多处理的组合数 (调试用)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global OUT_TAG
    OUT_TAG = args.out_tag

    config = load_config(args.config)
    base_dir = Path(args.config).parent
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [args.model] if args.model else list(config["models"].keys())

    out_root = base_dir / "output" / "phaseStep2" / f"causal_margin{OUT_TAG}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(out_root / "logs" / f"42_causal_margin_{ts}.log"))
    sys.stdout = log_saver

    combos = discover_combos(base_dir, models, tasks)
    print(f"[发现] {len(combos)} 个有效组合: " + ", ".join(
        f"{c['model']}|L{c['layer']}|{c['task']}" for c in combos))
    if not combos:
        print("[退出] 没有可用组合")
        log_saver.close()
        sys.stdout = log_saver.terminal
        return

    records = []
    total_feat = 0
    try:
        by_model = {}
        for c in combos:
            by_model.setdefault(c["model"], []).append(c)
        for m, cs in by_model.items():
            if args.limit:
                cs = cs[:args.limit]
            _, model_cfg = resolve_model_cfg(config, m)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model_path = base_dir / model_cfg["local_path"]
            if not model_path.exists():
                for p in (base_dir / "models").glob("*/*"):
                    if p.is_dir() and (p / "config.json").exists():
                        model_path = p
                        break
            print(f"\n===== [{m}] 加载基座模型 ({len(cs)} 个组合) =====")
            if model_needs_fp8(model_cfg, model_path):
                try:
                    from huggingface_hub import constants as _hf_const
                    _hf_const.HF_HUB_OFFLINE = True
                except Exception as _e:
                    print(f"  [fp8] 设置离线模式失败: {_e}")
                try:
                    os.environ.setdefault(
                        "TRITON_CACHE_DIR",
                        os.path.join(os.path.expanduser("~"), ".cache", "triton"))
                except Exception:
                    pass
                try:
                    import huggingface_hub._snapshot_download as _sd
                    _sd.read_tree_cache = lambda *a, **k: None
                except Exception as _e:
                    print(f"  [fp8] patch read_tree_cache 失败: {_e}")
            model, tokenizer = load_model_for_generation(base_dir, model_cfg, model_path)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            if model_needs_fp8(model_cfg, model_path):
                fp8_dequant_selfcheck(model, tokenizer, device)
            try:
                model.config.use_cache = False
            except Exception:
                pass
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
            layers_mod = find_decoder_layers(model)
            probe = "When Mary and John went to the store, John gave a drink to"
            lmap = calibrate_layer_map(model, tokenizer, layers_mod, device, probe)
            layer_mod = {}
            for c in cs:
                bi = c["block_idx"]
                if bi is None:
                    bi = (lmap.get(str(c["layer"])) or {}).get("block_idx")
                if bi is None or bi >= len(layers_mod):
                    print(f"  [跳过] {m} L{c['layer']}: 层对齐失败")
                    continue
                layer_mod[c["layer"]] = layers_mod[bi]
            tag = {"model": model, "tokenizer": tokenizer, "layer_mod": layer_mod}
            for c in cs:
                if c["layer"] not in layer_mod:
                    continue
                key = f"{m}|L{c['layer']}|{c['task']}"
                print(f"\n[{key}] n_feat={c['n_features']} b0={c['b0']:.3f}")
                try:
                    res = run_one(args, config, base_dir, c["model"], c, tag)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[失败] {key}: {e}")
                    continue
                if res.get("error"):
                    print(f"  [错误] {res['error']}")
                    continue
                records.append(res)
                total_feat += int(res.get("n_features") or 0)
                # 逐组合增量落盘 (支持中断续跑)
                summarize(records, out_root, {
                    "timestamp": ts, "n_prompts": args.n_prompts, "seed": args.seed,
                    "n_features_total": total_feat, "models": models, "tasks": tasks})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        log_saver.close()
        sys.stdout = log_saver.terminal

    print(f"[完成] 有效组合={len(records)}, 特征总数={total_feat}")
    print(f"[输出] {out_root / 'stepA_offline.md'}")


if __name__ == "__main__":
    main()
