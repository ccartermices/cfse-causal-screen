# -*- coding: utf-8 -*-
"""49: CFSE held-out 选择-评估基准

现有 36/38/40 的共同缺口: 特征在同一批 200 条 prompt 上被选中、又在同一批上
评估, 存在选择泄漏, 且集合级只测过 M2 (38) 或同批 (40)。

本脚本做真正的 held-out:
  split A (100 prompts): 选特征 (pred_full / M2 / M3 / random / oracle)
  split B (100 prompts): 测这些特征的逐特征与联合因果效应

核心新量:
  - held-out nDCG@k / Recall@k: 选择分数来自 A, 真值来自 B
  - held-out Coverage@N = C_joint(B) / C_mean-ablate(B)
  - Selection optimism = C_norm(joint on A) - C_norm(joint on B)
  - Selection fidelity = Jaccard(selected top-N, oracle top-N on B)
  - 每个选择器的前向预算

输出: output/phaseStep2/cfse_heldout/{model}/layer{L}/{task}_s{split}.json
      output/phaseStep2/cfse_heldout/summary_49.{json,md}
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import math
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from importlib import import_module

torch.set_num_threads(1)

from model_utils import load_config, resolve_model_cfg, model_needs_fp8, dequantize_fp8_model

_m36 = import_module("36_causal_validity")
TASK_BUILDERS = _m36.TASK_BUILDERS
batch_behavior = _m36.batch_behavior
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae
find_decoder_layers = _m36.find_decoder_layers
load_model_for_generation = _m36.load_model_for_generation
make_feature_hook = _m36.make_feature_hook
make_mean_ablate_hook = _m36.make_mean_ablate_hook
make_all_feature_ablate_hook = _m36.make_all_feature_ablate_hook
capture_sparse = _m36.capture_sparse
_save = _m36._save

_m38 = import_module("38_topn_joint_intervention")
make_feature_set_hook = _m38.make_feature_set_hook

_m42 = import_module("42_causal_margin_probe")
gradient_probe = _m42.gradient_probe

_m48 = import_module("48_cfse_offline")
spearman = _m48.spearman
ols = _m48.ols
ndcg_at_k = _m48.ndcg_at_k
recall_at_k = _m48.recall_at_k
jaccard = _m48.jaccard

NS = [1, 5, 10, 20, 50]
KS = [10, 20, 50]
SELECTORS = ["pred_full", "M2", "M3", "random", "oracle"]


def _boot_ci(vals, B=2000, seed=7):
    vals = np.asarray([v for v in vals if v is not None], dtype=float)
    if len(vals) < 3:
        return None, None
    rng = np.random.RandomState(seed)
    ms = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(B)]
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def _behavior_pack(vals, b0_list, b0_mean, b0_std):
    flip = float(np.mean([1.0 if (v > 0) != (v0 > 0) else 0.0
                          for v, v0 in zip(vals, b0_list)]))
    C = float(abs(b0_mean - np.mean(vals)))
    return {"B": float(np.mean(vals)), "C": C,
            "C_norm": float(C / max(b0_std, 1e-8)), "flip": flip}


def _run_hook(model, tokenizer, items, layer_mod, hook, device, batch_size):
    vals, bm = batch_behavior(model, tokenizer, items, layer_mod, hook, device,
                              batch_size)
    return vals, bm


def run_one(args, config, base_dir, m, layer, task):
    _, model_cfg = resolve_model_cfg(config, m)
    out_dir = (base_dir / "output" / "phaseStep2" / f"cfse_heldout{getattr(args, 'tag', '')}"
               / m / f"layer{layer}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"
    if out_file.exists() and not args.force:
        print(f"  [跳过] {m} L{layer} {task} 已完成")
        return json.load(open(out_file, encoding="utf-8"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = {"model": m, "layer": layer, "task": task,
              "timestamp": datetime.now().isoformat(),
              "n_prompts": args.n_prompts, "n_features": args.n_features,
              "n_splits": args.n_splits}

    loaded = load_per_layer_sae(base_dir, m, layer, device)
    if loaded is None:
        result["error"] = "缺少 per-layer SAE"
        _save(out_file, result)
        return result
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    model_path = base_dir / model_cfg["local_path"]
    if not model_path.exists():
        for p in (base_dir / "models").glob("*/*"):
            if p.is_dir() and (p / "config.json").exists():
                model_path = p
                break
    print(f"\n[{m} L{layer} {task}] 加载基座模型...")
    if model_needs_fp8(model_cfg, model_path):
        try:
            from huggingface_hub import constants as _hf_const
            _hf_const.HF_HUB_OFFLINE = True
        except Exception:
            pass
        try:
            import huggingface_hub._snapshot_download as _sd
            _sd.read_tree_cache = lambda *a, **k: None
        except Exception:
            pass
    model, tokenizer = load_model_for_generation(base_dir, model_cfg, model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_needs_fp8(model_cfg, model_path):
        # fp8 无 autograd; 反量化为 bf16 后再做一阶梯度探针
        try:
            n = dequantize_fp8_model(model, torch.bfloat16)
            result["fp8_dequant_modules"] = n
            print(f"  [fp8] 反量化 {n} 个模块 -> bf16")
        except Exception as e:
            result["error"] = f"fp8 反量化失败: {e}"
            _save(out_file, result)
            return result

    layers_mod = find_decoder_layers(model)
    probe = "When Mary and John went to the store, John gave a drink to"
    lmap = calibrate_layer_map(model, tokenizer, layers_mod, device, probe)
    result["layer_map"] = lmap
    if str(layer) not in lmap:
        result["error"] = f"层对齐失败: hidden_states[{layer}]"
        _save(out_file, result)
        return result
    block_idx = lmap[str(layer)]["block_idx"]
    layer_mod = layers_mod[block_idx]
    result["block_idx"] = block_idx
    result["align_max_abs_diff"] = lmap[str(layer)]["max_abs_diff"]

    items, task_meta = TASK_BUILDERS[task](tokenizer, args.n_prompts, seed=args.seed)
    if not items:
        result["error"] = f"任务构造失败: {task_meta}"
        _save(out_file, result)
        return result
    result["task_meta"] = task_meta

    result["splits"] = []
    for sidx in range(args.n_splits):
        rs = args.seed + 1000 * sidx
        rng = np.random.RandomState(rs)
        perm = rng.permutation(len(items))
        n_sel = len(items) // 2
        sel_items = [items[i] for i in perm[:n_sel]]
        ev_items = [items[i] for i in perm[n_sel:]]
        print(f"  [split {sidx}] select={len(sel_items)} eval={len(ev_items)} seed={rs}",
              flush=True)

        b0_sel_list, b0_sel = batch_behavior(model, tokenizer, sel_items, layer_mod,
                                             None, device, args.batch_size)
        b0_ev_list, b0_ev = batch_behavior(model, tokenizer, ev_items, layer_mod,
                                           None, device, args.batch_size)
        acc_sel = float(np.mean([1.0 if v > 0 else 0.0 for v in b0_sel_list]))
        acc_ev = float(np.mean([1.0 if v > 0 else 0.0 for v in b0_ev_list]))
        std_ev = float(np.std(b0_ev_list))
        split_rec = {"split": sidx, "split_seed": rs,
                     "b0_sel": b0_sel, "b0_eval": b0_ev,
                     "b0_acc_sel": acc_sel, "b0_acc_eval": acc_ev,
                     "b0_std_eval": std_ev,
                     "saturated_eval": bool(acc_ev >= 0.95)}

        # ---- 候选池: 选择集上的 M2 top-K ----
        packs = capture_sparse(model, tokenizer, sel_items, layer_mod, sae,
                               mean_t, std_t, device, args.batch_size)
        S = np.concatenate([p[0] for p in packs], axis=0)
        M = np.concatenate([p[1] for p in packs], axis=0)
        S = S[M]
        mean_act = S.mean(axis=0)
        max_act = S.max(axis=0)
        act_freq = (S > 0).mean(axis=0)
        order = np.argsort(mean_act)[::-1][:args.n_features]
        cand = [int(x) for x in order.tolist()]
        split_rec["n_candidates"] = len(cand)
        split_rec["candidate_top_M2"] = cand[:10]

        # ---- pred_full (选择集, 一次前向 + 一次反向) ----
        gp = gradient_probe(model, tokenizer, sel_items, layer_mod, sae,
                            mean_t, std_t, device, args.batch_size, feat_idx=cand)
        pred = np.asarray(gp["pred_full"], dtype=float)
        scores = {
            "pred_full": np.abs(pred),
            "M2": np.asarray([mean_act[f] for f in cand], dtype=float),
            "M3": np.asarray([max_act[f] for f in cand], dtype=float),
            "random": rng.permutation(len(cand)).astype(float),
        }
        signed_pred = pred

        # ---- oracle: 选择集上逐特征消融 (上界) ----
        c_sel = []
        for f in cand:
            hk = make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
            _, bm = _run_hook(model, tokenizer, sel_items, layer_mod, hk, device,
                              args.batch_size)
            c_sel.append(abs(bm - b0_sel))
        c_sel = np.asarray(c_sel, dtype=float)
        scores["oracle"] = c_sel

        # ---- 评估集逐特征真值 (200 个候选) ----
        c_ev, b_ev = [], []
        for f in cand:
            hk = make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
            vals, bm = _run_hook(model, tokenizer, ev_items, layer_mod, hk, device,
                                 args.batch_size)
            c_ev.append(abs(bm - b0_ev))
            b_ev.append(bm)
        c_ev = np.asarray(c_ev, dtype=float)
        b_ev = np.asarray(b_ev, dtype=float)
        db_ev = b_ev - b0_ev
        split_rec["c_eval"] = [float(x) for x in c_ev]
        split_rec["c_sel"] = [float(x) for x in c_sel]
        split_rec["reliability_rho"] = spearman(c_sel, c_ev)

        # ---- 参考: mean ablation / all-feature ----
        _, bm_mean = _run_hook(model, tokenizer, ev_items, layer_mod,
                               make_mean_ablate_hook(mean_t), device, args.batch_size)
        c_mean = abs(bm_mean - b0_ev)
        _, bm_all = _run_hook(model, tokenizer, ev_items, layer_mod,
                              make_all_feature_ablate_hook(sae, mean_t, std_t),
                              device, args.batch_size)
        c_all = abs(bm_all - b0_ev)
        _, bm_mean_sel = _run_hook(model, tokenizer, sel_items, layer_mod,
                                   make_mean_ablate_hook(mean_t), device, args.batch_size)
        c_mean_sel = abs(bm_mean_sel - b0_sel)

        # ---- 逐特征排序指标 (分数来自 sel, 真值来自 eval) ----
        feat_metrics = {}
        for name in ("pred_full", "M2", "M3", "random", "oracle"):
            s = scores[name]
            rec = {"rho_abs": spearman(s, c_ev),
                   "n": len(cand)}
            for k in KS:
                rec[f"ndcg@{k}"] = ndcg_at_k(s, c_ev, k)
                rec[f"recall@{k}"] = recall_at_k(s, c_ev, k)
                top = np.argsort(-s, kind="mergesort")[:k]
                rec[f"mass_recall@{k}"] = float(c_ev[top].sum() / max(c_ev.sum(), 1e-12))
            if name == "pred_full":
                rec["rho_signed"] = spearman(signed_pred, db_ev)
                rec["sign_acc"] = float(np.mean(np.sign(signed_pred) == np.sign(db_ev)))
                b, r2 = ols(signed_pred, db_ev)
                rec["calib_slope"] = b
                rec["calib_r2"] = r2
            feat_metrics[name] = rec

        # ---- 集合级: 选中后分别在 sel / eval 上做联合消融 ----
        sets = {}
        rand_sets = {N: [rng.choice(len(cand), size=N, replace=False).tolist()
                         for _ in range(args.n_random_sets)]
                     for N in NS if N <= len(cand)}
        for name in SELECTORS:
            order_s = np.argsort(-scores[name], kind="mergesort")
            sets[name] = {}
            for N in NS:
                if N > len(cand):
                    continue
                idxs = [int(x) for x in order_s[:N].tolist()]
                feats = [cand[i] for i in idxs]
                hk = make_feature_set_hook(sae, mean_t, std_t, feats, "ablate", 1.0)
                vals_sel, bm_sel = _run_hook(model, tokenizer, sel_items, layer_mod,
                                             hk, device, args.batch_size)
                vals_ev, bm_ev = _run_hook(model, tokenizer, ev_items, layer_mod,
                                           hk, device, args.batch_size)
                # random 对照 (eval)
                rc = []
                for ridx in rand_sets[N]:
                    rf = [cand[i] for i in ridx]
                    rh = make_feature_set_hook(sae, mean_t, std_t, rf, "ablate", 1.0)
                    _, rbm = _run_hook(model, tokenizer, ev_items, layer_mod, rh,
                                       device, args.batch_size)
                    rc.append(abs(rbm - b0_ev))
                rc = np.asarray(rc, dtype=float)
                pack_ev = _behavior_pack(vals_ev, b0_ev_list, b0_ev, std_ev)
                pack_sel = _behavior_pack(vals_sel, b0_sel_list, b0_sel,
                                          float(np.std(b0_sel_list)))
                oracle_ev = set(np.argsort(-c_ev)[:N].tolist())
                sets[name][N] = {
                    "feature_idx": feats,
                    "eval": pack_ev,
                    "sel": pack_sel,
                    "rand_mean_C": float(rc.mean()),
                    "rand_std_C": float(rc.std()),
                    "z": float((pack_ev["C"] - rc.mean()) / max(rc.std(), 1e-8)),
                    "excess": float(pack_ev["C"] / max(rc.mean(), 1e-12)),
                    "coverage": float(pack_ev["C"] / max(c_mean, 1e-12)),
                    "coverage_sel": float(pack_sel["C"] / max(c_mean_sel, 1e-12)),
                    "optimism_norm": float(pack_sel["C_norm"] - pack_ev["C_norm"]),
                    "optimism_cov": float(pack_sel["C"] / max(c_mean_sel, 1e-12)
                                          - pack_ev["C"] / max(c_mean, 1e-12)),
                    "fidelity_oracle": jaccard(idxs, oracle_ev),
                }

        split_rec.update({
            "refs": {"mean_ablate_C_eval": c_mean, "all_feature_C_eval": c_all,
                     "mean_ablate_C_sel": c_mean_sel},
            "feature": feat_metrics,
            "sets": sets,
            "pred_stats": {"pred_signed_mean": float(np.mean(signed_pred)),
                           "pred_abs_median": float(np.median(np.abs(signed_pred)))},
            "cost": {"pred_full_passes": 2, "M2_passes": 0, "M3_passes": 0,
                     "random_passes": 0, "oracle_passes": len(cand),
                     "forward_per_feature_eval": len(cand),
                     "joint_forwards": len(SELECTORS) * len(NS) * 2
                                       + len(NS) * args.n_random_sets},
        })
        result["splits"].append(split_rec)
        print(f"    [pred] rho_abs={feat_metrics['pred_full']['rho_abs']:.3f} "
              f"sign={feat_metrics['pred_full']['sign_acc']:.3f} "
              f"ndcg20={feat_metrics['pred_full']['ndcg@20']:.3f}", flush=True)
        print(f"    [coverage@10] " + ", ".join(
            f"{k}={sets[k][10]['coverage']:.3f}" for k in SELECTORS if 10 in sets[k]),
            flush=True)
        _save(out_file, result)  # 每个 split 落盘, 支持中断

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _save(out_file, result)
    return result


def summarize(base, out_dir):
    rows = []
    for f in out_dir.glob("*/layer*/*.json"):
        if f.name.startswith("summary"):
            continue
        rec = json.load(open(f, encoding="utf-8"))
        if rec.get("error"):
            continue
        for sp in rec.get("splits") or []:
            sets_norm = {sel: {int(N): d for N, d in (per or {}).items()}
                         for sel, per in (sp.get("sets") or {}).items()}
            rows.append({"model": rec["model"], "layer": rec["layer"],
                         "task": rec["task"], "split": sp["split"],
                         "saturated": sp["saturated_eval"],
                         "feature": sp["feature"], "sets": sets_norm,
                         "b0_acc_eval": sp["b0_acc_eval"],
                         "reliability_rho": sp.get("reliability_rho")})
    if not rows:
        return None
    summary = {"timestamp": datetime.now().isoformat(), "n_runs": len(rows),
               "n_combos": len({(r["model"], r["layer"], r["task"]) for r in rows}),
               "feature": {}, "sets": {}, "grade": {}}

    rels = [r["reliability_rho"] for r in rows if r.get("reliability_rho") is not None]
    if rels:
        summary["reliability"] = {"mean": float(np.mean(rels)),
                                  "median": float(np.median(rels)), "n": len(rels)}
    dis = []
    for r in rows:
        rel = r.get("reliability_rho")
        rho = (r["feature"].get("pred_full") or {}).get("rho_abs")
        if rel and rel > 0 and rho is not None:
            dis.append(rho / math.sqrt(rel))
    if dis:
        summary["rho_disattenuated_pred_full"] = {"mean": float(np.mean(dis)),
                                                  "median": float(np.median(dis)),
                                                  "n": len(dis)}

    for sel in SELECTORS:
        vals = {}
        for key in ("rho_abs", "rho_signed", "sign_acc", "calib_slope", "calib_r2"):
            xs = [r["feature"][sel].get(key) for r in rows if sel in r["feature"]]
            xs = [x for x in xs if x is not None]
            if xs:
                lo, hi = _boot_ci(xs)
                vals[key] = {"mean": float(np.mean(xs)),
                             "median": float(np.median(xs)),
                             "sd": float(np.std(xs)), "ci95": [lo, hi], "n": len(xs)}
        for k in KS:
            for pre in ("ndcg", "recall"):
                key = f"{pre}@{k}"
                xs = [r["feature"][sel].get(key) for r in rows if sel in r["feature"]]
                xs = [x for x in xs if x is not None]
                if xs:
                    vals[key] = {"mean": float(np.mean(xs)),
                                 "median": float(np.median(xs)),
                                 "sd": float(np.std(xs))}
        summary["feature"][sel] = vals

        per_n = {}
        for N in NS:
            acc = {k: [] for k in ("z", "coverage", "excess", "flip", "C_norm",
                                   "optimism_norm", "optimism_cov", "fidelity_oracle")}
            for r in rows:
                d = (r["sets"].get(sel) or {}).get(N)
                if not d:
                    continue
                for k in acc:
                    if k == "flip":
                        if r["saturated"]:
                            continue
                        v = d["eval"].get("flip")
                    elif k in ("z", "coverage", "excess", "optimism_norm",
                               "fidelity_oracle"):
                        v = d.get(k)
                    elif k == "optimism_cov":
                        v = d.get("optimism_cov")
                        if v is None and d.get("coverage_sel") is not None:
                            v = d["coverage_sel"] - d["coverage"]
                    else:
                        v = d["eval"].get(k)
                    if v is not None:
                        acc[k].append(v)
            if acc["coverage"]:
                per_n[N] = {k: {"mean": float(np.mean(v)), "n": len(v)}
                            for k, v in acc.items() if v}
                if acc["z"]:
                    lo, hi = _boot_ci(acc["z"])
                    per_n[N]["z"]["ci95"] = [lo, hi]
        summary["sets"][sel] = per_n

        mag = []
        for r in rows:
            f = r["feature"].get(sel) or {}
            if f.get("rho_abs") is None:
                continue
            mag.append(f["rho_abs"] >= 0.5 and (f.get("ndcg@20") or 0) >= 0.5)
        if mag:
            summary["grade"][sel] = {"mag_pass_rate": float(np.mean(mag)),
                                     "n": len(mag)}

    # 分模型拆分（池化均值会掩盖 Pythia 这类失败）
    summary["by_model"] = {}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        bm = {"n": len(sub)}
        for sel in SELECTORS:
            def _mean(key):
                xs = [(r["feature"].get(sel) or {}).get(key) for r in sub]
                xs = [x for x in xs if x is not None]
                return float(np.mean(xs)) if xs else None
            def _cov(N):
                xs = []
                for r in sub:
                    d = (r["sets"].get(sel) or {}).get(N)
                    if d and d.get("coverage") is not None:
                        xs.append(d["coverage"])
                return float(np.mean(xs)) if xs else None
            bm[sel] = {"rho_abs": _mean("rho_abs"),
                       "ndcg@20": _mean("ndcg@20"),
                       "sign_acc": _mean("sign_acc"),
                       "coverage@10": _cov(10),
                       "coverage@50": _cov(50)}
        summary["by_model"][model] = bm
    with open(out_dir / "summary_49.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary


def render_md(summary, out_dir):
    L = ["# CFSE held-out 基准（实验 49）\n",
         f"- 生成: {summary['timestamp']}",
         f"- runs: {summary['n_runs']}；combos: {summary['n_combos']}",
         "- 口径: split A 选特征，split B 测效应；coverage 以 B 上 mean-ablate 为分母。\n"]
    if summary.get("reliability"):
        r = summary["reliability"]
        L.append(f"- split-half 可靠性（200 特征 C_ablate 的 A/B 相关）: "
                 f"mean={r['mean']:.3f}, median={r['median']:.3f}, n={r['n']}")
    if summary.get("rho_disattenuated_pred_full"):
        d = summary["rho_disattenuated_pred_full"]
        L.append(f"- pred_full 去衰减 ρ（除以 sqrt(可靠性)）: mean={d['mean']:.3f}, "
                 f"median={d['median']:.3f}, n={d['n']}")
    L += ["", "## 逐特征（held-out）\n",
         "| selector | rho_abs | rho_signed | SignAcc | nDCG@20 | Recall@20 | slope | R2 |",
         "|---|---|---|---|---|---|---|---|"]
    for sel in SELECTORS:
        d = summary["feature"].get(sel) or {}
        def f(k, nd=3):
            v = d.get(k)
            return f"{v['mean']:.{nd}f}" if v else "—"
        L.append(f"| {sel} | {f('rho_abs')} | {f('rho_signed')} | {f('sign_acc')} | "
                 f"{f('ndcg@20')} | {f('recall@20')} | {f('calib_slope')} | {f('calib_r2')} |")
    L.append("")
    if summary.get("by_model"):
        L.append("## 分模型拆分（held-out）\n")
        L.append("| model | selector | rho_abs | nDCG@20 | SignAcc | cov@10 | cov@50 |")
        L.append("|---|---|---|---|---|---|---|")
        for model, bm in summary["by_model"].items():
            for sel in SELECTORS:
                d = bm.get(sel) or {}
                def h(k, nd=3):
                    v = d.get(k)
                    return f"{v:.{nd}f}" if v is not None else "—"
                L.append(f"| {model} | {sel} | {h('rho_abs')} | {h('ndcg@20')} | "
                         f"{h('sign_acc')} | {h('coverage@10')} | {h('coverage@50')} |")
        L.append("")
    L.append("## 集合级（held-out）\n")
    L.append("| selector | N | z | coverage | excess | C_norm | optimism | fidelity | flip(未饱和) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for sel in SELECTORS:
        for N in NS:
            d = (summary["sets"].get(sel) or {}).get(N)
            if not d:
                continue
            def g(k, nd=3):
                v = d.get(k)
                return f"{v['mean']:.{nd}f}" if v else "—"
            fl = g("flip")
            L.append(f"| {sel} | {N} | {g('z',2)} | {g('coverage')} | {g('excess',2)} | "
                     f"{g('C_norm')} | {g('optimism_norm')} | {g('fidelity_oracle')} | {fl} |")
    L.append("")
    L.append("## 幅度轨通过率（rho>=0.5 且 nDCG@20>=0.5）\n")
    L.append("| selector | pass rate | n |")
    L.append("|---|---|---|")
    for sel, d in summary["grade"].items():
        L.append(f"| {sel} | {d['mag_pass_rate']:.3f} | {d['n']} |")
    (out_dir / "summary_49.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3.5-2b", "pythia-1.4b"])
    ap.add_argument("--tasks", nargs="+", default=["ioi"])
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_features", type=int, default=200)
    ap.add_argument("--n_splits", type=int, default=3)
    ap.add_argument("--n_random_sets", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    config = load_config(base / "config.json")
    for m in args.models:
        _, mc = resolve_model_cfg(config, m)
        layers = args.layers or [mc["target_layers"][1]]
        for layer in layers:
            for task in args.tasks:
                try:
                    run_one(args, config, base, m, layer, task)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"[失败] {m} L{layer} {task}: {e}")

    out_dir = base / "output" / "phaseStep2" / f"cfse_heldout{args.tag}"
    summary = summarize(base, out_dir)
    if summary:
        render_md(summary, out_dir)
        print(f"\n[汇总] {out_dir / 'summary_49.md'}")


if __name__ == "__main__":
    main()
