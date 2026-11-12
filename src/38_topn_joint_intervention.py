# -*- coding: utf-8 -*-
"""Step 2-4: top-N 联合干预 —— 把「分布级」证据升级为「行为级」证据

背景 (36 的结论与其口径缺陷):
  - 36 用单特征 ablation/steering 度量因果效应量 C, 但发现**翻转率不可跨组合比较**:
    翻转率几乎完全由无干预行为 B0 的边际决定 —— B0 acc=1.000 的 IOI 组合里
    单特征 logit 边际 3.85~6.89, 任何单特征都不可能翻盘 (指标饱和);
    而 B0 acc=0.635 的 tinyllama|L7|gt 单特征即可翻转 0.18。
  - 因此「翻转率=0」既可能是「特征无用」也可能是「任务太容易」, 无法区分。

本脚本的对策 (两条):
  1. **top-N 联合干预**: 把 36 里按 M2_mean_act_task 排出的 top-N 特征**同时** ablate/steer
     (N ∈ {1,5,10,20,50}), 度量联合效应 B_joint 与翻转率 flip_joint ——
     这回答的是「指标选出的特征集合整体是否真的因果有用」(行为级证据)。
  2. **边际归一化效应量**: C_norm = |B_joint − B0| / std(B0 per-prompt), 以及
     C_rel = |B_joint − B0| / |B0| —— 消掉任务难度(B0 边际)这一混杂因素,
     使不同 (model,layer,task) 之间可比。

对照:
  - 每个 N 做 R=20 个**同规模随机特征集合**的联合 ablation -> C/flip 的 mean/std/max;
  - mean-ablate (整段残差替换为均值) 与 all-feature ablation (去掉整个 SAE 重建) 作上界;
  - N=1 即 36 的 top-1, 用于交叉验证两条流水线一致。

特征来源: 直接读 36 已落盘的 run JSON 的 features 列表 (含 M2_mean_act_task),
          保证与 36 的选择口径完全一致, 不重算。

资源: CPU 严格节流 (OMP/MKL=1, torch threads=1), 单进程串行, 独显 cuda:0, 不打扰同机其他实验。
输出: output/phaseStep2/topn_joint/{model}/layer{L}/{task}.json
      output/phaseStep2/topn_joint/summary_38.json
      output/phaseStep2/topn_joint/topn_joint_summary.md
      回填 output/phaseF/phaseF_report.md §18
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

from model_utils import load_config, resolve_model_cfg, model_needs_fp8

# 复用 36 的全部基建 (文件名以数字开头, 只能用 importlib 动态加载)
_m36 = import_module("36_causal_validity")
TASK_BUILDERS = _m36.TASK_BUILDERS
ResumeManager = _m36.ResumeManager
batch_behavior = _m36.batch_behavior
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae
find_decoder_layers = _m36.find_decoder_layers
load_model_for_generation = _m36.load_model_for_generation
make_mean_ablate_hook = _m36.make_mean_ablate_hook
make_all_feature_ablate_hook = _m36.make_all_feature_ablate_hook
_save = _m36._save


# ----------------------------------------------------------------------------
# 集合级 hook (与 36 的单特征 hook 同构, 系数按集合求和)
# ----------------------------------------------------------------------------
def make_feature_set_hook(sae, mean_t, std_t, feat_list, mode, alpha):
    """对特征集合 {f} 同时做 ablation / steering.

    delta = Σ_f coef_f * W[:, f] * std ,  coef_f = -s_f (ablate) / (alpha-1)*s_f (steer)
    实现上等价于 (B,seq,F) 的系数矩阵左乘 (F, input_dim) 的 decoder 列子矩阵。
    """
    idx = torch.tensor([int(f) for f in feat_list], dtype=torch.long,
                       device=sae.decoder.weight.device)
    Wsel = sae.decoder.weight[:, idx].detach().float().transpose(0, 1)  # (F, input_dim)

    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        orig_dtype = hs.dtype
        h = hs.float()
        with torch.no_grad():
            normed = (h - mean_t) / std_t
            sparse, _ = sae.encode(normed)
            s = sparse[..., idx]                             # (B, seq, F)
        coef = -s if mode == "ablate" else (alpha - 1.0) * s
        delta = torch.matmul(coef, Wsel) * std_t.view(1, 1, -1)
        h_new = (h + delta).to(orig_dtype)
        if isinstance(output, (tuple, list)):
            return (h_new,) + tuple(output[1:])
        return h_new

    return hook


def _flip(vals, b0_list):
    return float(np.mean([1.0 if (v > 0) != (v0 > 0) else 0.0
                          for v, v0 in zip(vals, b0_list)]))


# ----------------------------------------------------------------------------
# 单个 (model, layer, task)
# ----------------------------------------------------------------------------
def run_one(args, config, base_dir, m, layer, task, ns):
    _, model_cfg = resolve_model_cfg(config, m)
    cv_dir = base_dir / "output" / "phaseStep2" / "causal_validity"
    src = cv_dir / m / f"layer{layer}" / f"{task}.json"
    out_root = base_dir / "output" / "phaseStep2" / "topn_joint"
    out_dir = out_root / m / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"
    resume = ResumeManager(out_dir, f"38_{task}_layer{layer}")
    if resume.is_completed() and out_file.exists() and not args.force:
        print(f"  [跳过] {m} layer{layer} {task} 已完成")
        return json.load(open(out_file, encoding="utf-8"))

    result = {"model": m, "layer": layer, "task": task,
              "timestamp": datetime.now().isoformat()}

    # ---- 1. 复用 36 的特征清单 ----
    if not src.exists():
        result["error"] = "缺少 36 的因果效度结果 (先跑 36_causal_validity.py)"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        return result
    prev = json.load(open(src, encoding="utf-8"))
    if prev.get("error") or prev.get("at_chance"):
        result["error"] = "36 结果无效 (error/at_chance), 跳过"
        result["b0_acc"] = prev.get("b0_acc")
        _save(out_file, result)
        resume.mark_completed({"skip": result["error"]})
        return result
    feats_sorted = sorted([f for f in (prev.get("features") or [])
                           if f.get("M2_mean_act_task") is not None],
                          key=lambda r: -r["M2_mean_act_task"])
    if not feats_sorted:
        result["error"] = "36 结果无 features"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        return result
    result["n_features_available"] = len(feats_sorted)
    result["ref_top_feature"] = {
        "feature_idx": feats_sorted[0]["feature_idx"],
        "M2_mean_act_task": feats_sorted[0]["M2_mean_act_task"],
        "C_ablate_36": feats_sorted[0].get("C_ablate"),
        "C_steer_36": feats_sorted[0].get("C_steer"),
        "flip_ablate_36": feats_sorted[0].get("flip_ablate"),
    }
    ns = [n for n in ns if n <= len(feats_sorted)]

    # ---- 2. 加载 per-layer SAE ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded = load_per_layer_sae(base_dir, m, layer, device)
    if loaded is None:
        result["error"] = "缺少 per-layer SAE (先跑 35_train_per_layer_sae.py)"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        return result
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)
    n_latents = int(sae.decoder.weight.shape[1])

    # ---- 3. 加载基座模型 (fp8 三级绕过, 口径与 36 一致) ----
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
            print("  [fp8] 已置 HF_HUB_OFFLINE=True 以绕过 kernel 信任联网检查")
        except Exception as _e:
            print(f"  [fp8] 设置离线模式失败: {_e}")
        try:
            os.environ.setdefault(
                "TRITON_CACHE_DIR",
                os.path.join(os.path.expanduser("~"), ".cache", "triton"))
            print(f"  [fp8] TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")
        except Exception as _e:
            print(f"  [fp8] 设置 TRITON_CACHE_DIR 失败: {_e}")
        try:
            import huggingface_hub._snapshot_download as _sd
            _sd.read_tree_cache = lambda *a, **k: None
            print("  [fp8] 已跳过 snapshot 完整性检查 (离线仅用 torch-cuda 变体)")
        except Exception as _e:
            print(f"  [fp8] patch read_tree_cache 失败: {_e}")
    model, tokenizer = load_model_for_generation(base_dir, model_cfg, model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers_mod = find_decoder_layers(model)
    probe = "When Mary and John went to the store, John gave a drink to"
    lmap = calibrate_layer_map(model, tokenizer, layers_mod, device, probe)
    if str(layer) not in lmap:
        result["error"] = f"层对齐失败: hidden_states[{layer}] 未匹配到任何 block"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        del model
        return result
    block_idx = lmap[str(layer)]["block_idx"]
    layer_mod = layers_mod[block_idx]
    result["block_idx"] = block_idx
    result["align_max_abs_diff"] = lmap[str(layer)]["max_abs_diff"]
    print(f"  [对齐] hidden_states[{layer}] <-> layers_mod[{block_idx}] "
          f"(max|diff|={lmap[str(layer)]['max_abs_diff']:.2e})")

    # ---- 4. 任务构造 + B0 ----
    builder = TASK_BUILDERS[task]
    items, task_meta = builder(tokenizer, args.n_prompts, seed=args.seed)
    if not items:
        result["error"] = f"任务构造失败: {task_meta}"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        del model
        return result
    b0_list, b0 = batch_behavior(model, tokenizer, items, layer_mod, None, device,
                                 args.batch_size)
    b0_arr = np.asarray(b0_list, dtype=float)
    b0_std = float(b0_arr.std())
    acc0 = float(np.mean([1.0 if v > 0 else 0.0 for v in b0_list]))
    result.update({"n_prompts": len(items), "b0": float(b0), "b0_acc": acc0,
                   "b0_logit_std": b0_std, "at_chance_36": False})
    print(f"  [B0] B0={b0:.3f} acc={acc0:.3f} std(per-prompt logit)={b0_std:.3f}")

    # ---- 5. hook 恒等性检查 (集合 hook, N=1, alpha=1) ----
    hk_id = make_feature_set_hook(sae, mean_t, std_t, [feats_sorted[0]["feature_idx"]],
                                  "steer", 1.0)
    _, b_hk = batch_behavior(model, tokenizer, items[:4], layer_mod, hk_id, device, 4)
    _, b_no = batch_behavior(model, tokenizer, items[:4], layer_mod, None, device, 4)
    result["hook_identity_ok"] = bool(abs(b_no - b_hk) < 1e-3)
    print(f"  [hook] 恒等性(alpha=1)检查: {result['hook_identity_ok']}")

    denom = abs(b0) if abs(b0) > 1e-9 else None

    def measure_hook(hk):
        vals, bm = batch_behavior(model, tokenizer, items, layer_mod, hk, device,
                                  args.batch_size)
        C = float(abs(bm - b0))
        return {"B": float(bm), "C": C,
                "C_norm": (C / b0_std if b0_std > 0 else None),
                "C_rel": (C / denom if denom else None),
                "flip": _flip(vals, b0_list),
                "B_std_per_prompt": float(np.asarray(vals, dtype=float).std())}

    def measure_set(feat_list, mode, alpha):
        return measure_hook(make_feature_set_hook(sae, mean_t, std_t, feat_list,
                                                  mode, alpha))

    # ---- 6. 上界参照: mean-ablate / all-feature ----
    refs = {
        "mean_ablate": measure_hook(make_mean_ablate_hook(mean_t)),
        "all_feature_ablate": measure_hook(make_all_feature_ablate_hook(sae, mean_t, std_t)),
    }
    print(f"  [参照] mean-ablate C={refs['mean_ablate']['C']:.3f} "
          f"flip={refs['mean_ablate']['flip']:.3f} | "
          f"all-feature C={refs['all_feature_ablate']['C']:.3f} "
          f"flip={refs['all_feature_ablate']['flip']:.3f}")

    # ---- 7. 逐 N: top 集合 vs 随机集合 ----
    rows = []
    for N in ns:
        fl = [int(f["feature_idx"]) for f in feats_sorted[:N]]
        ab = measure_set(fl, "ablate", 1.0)
        st = measure_set(fl, "steer", args.alpha)
        rng = np.random.RandomState(args.seed + N)
        rc, rf, rcn = [], [], []
        for _ in range(args.n_random_sets):
            rf_feats = rng.choice(n_latents, size=N, replace=False).tolist()
            mm = measure_set(rf_feats, "ablate", 1.0)
            rc.append(mm["C"]); rf.append(mm["flip"])
            if mm["C_norm"] is not None:
                rcn.append(mm["C_norm"])
        rows.append({
            "N": N, "feature_idx": fl,
            "ablate": ab, "steer": st,
            "random_ablate": {
                "n": len(rc),
                "mean_C": float(np.mean(rc)), "std_C": float(np.std(rc)),
                "max_C": float(np.max(rc)),
                "mean_C_norm": (float(np.mean(rcn)) if rcn else None),
                "std_C_norm": (float(np.std(rcn)) if rcn else None),
                "mean_flip": float(np.mean(rf)), "std_flip": float(np.std(rf)),
                "max_flip": float(np.max(rf)),
            },
        })
        z = (ab["C_norm"] / (np.mean(rcn) + 1e-9)) if rcn else None
        print(f"  [N={N:2d}] ablate C={ab['C']:.3f} C_norm={ab['C_norm']:.3f} "
              f"flip={ab['flip']:.3f} | steer C={st['C']:.3f} flip={st['flip']:.3f} "
              f"| rand C={np.mean(rc):.4f} flip={np.mean(rf):.4f} "
              f"| C_norm/rand={z if z is None else round(z, 1)}")

    result["refs"] = refs
    result["results"] = rows
    result["top_features_by"] = "M2_mean_act_task (desc)"
    result["n_random_sets"] = args.n_random_sets
    _save(out_file, result)
    resume.mark_completed({"n_available": len(feats_sorted), "ns": ns,
                           "b0_acc": acc0})
    print(f"  [完成] {out_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ----------------------------------------------------------------------------
# 汇总 + 报告回填
# ----------------------------------------------------------------------------
def collect(root):
    runs = []
    for p in sorted(root.glob("*/layer*/*.json")):
        if p.name.startswith("resume_") or p.name.startswith("summary_"):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "results" not in d or "model" not in d:
            continue
        runs.append(d)
    return runs


def _f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def _agg_table(runs, by_task=True):
    """返回 {group: {N: {...}}} 聚合."""
    out = {}
    for r in runs:
        g = r["task"] if by_task else "all"
        out.setdefault(g, {}).setdefault("runs", 0)
        out[g]["runs"] += 1
        for row in r["results"]:
            N = row["N"]
            d = out[g].setdefault(N, {"c_joint": [], "c_norm": [], "c_rel": [],
                                      "flip": [], "rand_c": [], "rand_flip": [],
                                      "rand_cnorm": [], "steer_c": [], "steer_flip": [],
                                      "n": 0})
            d["c_joint"].append(row["ablate"]["C"])
            if row["ablate"]["C_norm"] is not None:
                d["c_norm"].append(row["ablate"]["C_norm"])
            if row["ablate"]["C_rel"] is not None:
                d["c_rel"].append(row["ablate"]["C_rel"])
            d["flip"].append(row["ablate"]["flip"])
            d["rand_c"].append(row["random_ablate"]["mean_C"])
            d["rand_flip"].append(row["random_ablate"]["mean_flip"])
            if row["random_ablate"]["mean_C_norm"] is not None:
                d["rand_cnorm"].append(row["random_ablate"]["mean_C_norm"])
            d["steer_c"].append(row["steer"]["C"])
            d["steer_flip"].append(row["steer"]["flip"])
            d["n"] += 1
    return out


def render_md(runs, ts):
    agg = _agg_table(runs, by_task=True)
    lines = ["# top-N 联合干预 (行为级证据) —— 汇总", "",
             f"- 生成时间: {ts}",
             f"- 组合数: {len(runs)} (有效, 已排除 36 中 at_chance/error 的 (model,layer,task))",
             "- 特征来源: 36 的 features 列表按 M2_mean_act_task 降序取 top-N",
             "- 口径: C_joint=|B_joint−B0|; C_norm=C_joint/std(B0 per-prompt); "
             "C_rel=C_joint/|B0|; flip_joint 相对 B0 的逐 prompt 符号翻转率",
             "- 随机基线: 每个 N 的 R 个同规模随机特征集合 (同 seed) 联合 ablation",
             "", "## 按任务聚合", ""]
    for g in sorted(agg.keys()):
        lines.append(f"### 任务 {g}  (n_combos={agg[g]['runs']})")
        lines.append("")
        lines.append("| N | top C_joint | top C_norm | top C_rel | top flip | "
                     "steer C | steer flip | rand C | rand C_norm | rand flip | "
                     "top/rand(C) | top/rand(flip) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for N in sorted(k for k in agg[g].keys() if isinstance(k, int)):
            d = agg[g][N]
            mc, mcn = float(np.mean(d["c_joint"])), float(np.mean(d["c_norm"])) if d["c_norm"] else None
            mcr = float(np.mean(d["c_rel"])) if d["c_rel"] else None
            mf = float(np.mean(d["flip"]))
            sc, sf = float(np.mean(d["steer_c"])), float(np.mean(d["steer_flip"]))
            rcn = float(np.mean(d["rand_c"]))
            rcnorm = float(np.mean(d["rand_cnorm"])) if d["rand_cnorm"] else None
            rf = float(np.mean(d["rand_flip"]))
            lines.append(f"| {N} | {_f(mc,3)} | {_f(mcn,3)} | {_f(mcr,3)} | {_f(mf,3)} | "
                         f"{_f(sc,3)} | {_f(sf,3)} | {_f(rcn)} | {_f(rcnorm,3)} | {_f(rf)} | "
                         f"{_f(mc/(rcn+1e-9),1)} | {_f(mf/(rf+1e-9),1)} |")
        lines.append("")
    # 逐组合 (最大 N): z = (C_top − mean C_rand) / std C_rand, 即 top 集合相对
    # 同规模随机集合的显著性 (边际归一化的零假设检验)。
    per = []
    for r in runs:
        if not r["results"]:
            continue
        row = r["results"][-1]
        rnd = row["random_ablate"]
        sd = rnd["std_C"] + 1e-12
        per.append({
            "key": f"{r['model']}|L{r['layer']}|{r['task']}", "r": r, "row": row,
            "z_ablate": (row["ablate"]["C"] - rnd["mean_C"]) / sd,
            "z_steer": (row["steer"]["C"] - rnd["mean_C"]) / sd,
        })
    per.sort(key=lambda x: -x["z_ablate"])
    zs = [x["z_ablate"] for x in per]
    n_z2 = sum(1 for z in zs if z < 2)
    n_z5 = sum(1 for z in zs if z < 5)
    flips = [x["key"] for x in per if x["row"]["ablate"]["flip"] > 0.05]
    lines += ["## 显著性: top 集合 vs 同规模随机集合 (最大 N)", "",
              f"- 中位 z = {_f(float(np.median(zs)),1)}; "
              f"z<2 (与随机集合不可区分) 的组合: {n_z2}/{len(per)}; "
              f"z<5: {n_z5}/{len(per)}",
              f"- 行为级翻转率 >0.05 的组合: {len(flips)}/{len(per)} "
              f"({', '.join(flips) if flips else '无'})",
              "- 判读: 翻转率只在**未被任务饱和度掩盖**的组合上可见 (B0 acc<1.0), "
              "因此必须配合 C_norm 才能跨组合比较。",
              "", "## 逐组合 (最大 N, 按 z 降序)", "",
              "| 组合 | B0 | B0 acc | std(B0) | N | C_joint | C_norm | flip | "
              "rand C | rand flip | z(abl) | z(steer) | mean-ablate C | all-feature C |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in per:
        r, row = x["r"], x["row"]
        lines.append(
            f"| {x['key']} | {_f(r.get('b0'),3)} | {_f(r.get('b0_acc'),3)} | "
            f"{_f(r.get('b0_logit_std'),3)} | {row['N']} | {_f(row['ablate']['C'],3)} | "
            f"{_f(row['ablate']['C_norm'],3)} | {_f(row['ablate']['flip'],3)} | "
            f"{_f(row['random_ablate']['mean_C'])} | "
            f"{_f(row['random_ablate']['mean_flip'])} | "
            f"{_f(x['z_ablate'],1)} | {_f(x['z_steer'],1)} | "
            f"{_f((r.get('refs') or {}).get('mean_ablate', {}).get('C'),3)} | "
            f"{_f((r.get('refs') or {}).get('all_feature_ablate', {}).get('C'),3)} |")
    lines.append("")
    return "\n".join(lines)


def summarize(base_dir, args):
    root = base_dir / "output" / "phaseStep2" / "topn_joint"
    if not root.exists():
        print("[跳过] 尚无 topn_joint 结果")
        return
    runs = collect(root)
    if not runs:
        print("[跳过] 未找到有效结果")
        return
    ts = datetime.now().isoformat()
    rows = []
    for r in runs:
        rows.append({
            "model": r["model"], "layer": r["layer"], "task": r["task"],
            "b0": r.get("b0"), "b0_acc": r.get("b0_acc"),
            "b0_logit_std": r.get("b0_logit_std"),
            "hook_identity_ok": r.get("hook_identity_ok"),
            "align_max_abs_diff": r.get("align_max_abs_diff"),
            "n_features_available": r.get("n_features_available"),
            "ref_top_feature": r.get("ref_top_feature"),
            "refs": r.get("refs"),
            "results": r.get("results"),
        })
    summary = {"timestamp": ts, "n_runs": len(runs), "runs": rows}
    _save(root / "summary_38.json", summary)
    md = render_md(runs, ts)
    (root / "topn_joint_summary.md").write_text(md, encoding="utf-8")
    print(f"[汇总] {root / 'topn_joint_summary.md'} ({len(runs)} 组合)")

    rp = base_dir / args.report
    if rp.exists() and not args.no_report:
        _m37 = import_module("37_summarize_causal_validity")
        _m37._upsert_section(
            rp, "## 18. Phase Step2 续: top-N 联合干预 (行为级证据)", md)
        print(f"[回填] {rp} §18")


def main():
    ap = argparse.ArgumentParser(description="Step2-4: top-N 联合干预")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--layers", type=str, default=None, help="逗号分隔, 默认 target_layers 三层")
    ap.add_argument("--tasks", type=str, default="ioi,gt")
    ap.add_argument("--ns", type=str, default="1,5,10,20,50")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_random_sets", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", type=str, default="output/phaseF/phaseF_report.md")
    ap.add_argument("--no_report", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    base_dir = Path(args.config).parent
    if args.summarize_only:
        summarize(base_dir, args)
        return

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [args.model] if args.model else list(config["models"].keys())

    out_root = base_dir / "output" / "phaseStep2" / "topn_joint"
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = _m36._mod05.LogSaver(str(log_dir / f"38_topn_joint_{ts}.log"))
    sys.stdout = log_saver

    summary_path = out_root / "summary_38.json"
    summary = {"timestamp": ts, "runs": {}}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev.get("runs"), dict):
                summary["runs"] = prev["runs"]
        except Exception:
            pass
    try:
        for m in models:
            _, model_cfg = resolve_model_cfg(config, m)
            if args.layers:
                layers = [int(x) for x in args.layers.split(",")]
            else:
                layers = list(model_cfg["target_layers"])
            for layer in layers:
                for task in tasks:
                    key = f"{m}|L{layer}|{task}"
                    try:
                        res = run_one(args, config, base_dir, m, layer, task, ns)
                        summary["runs"][key] = {
                            "error": res.get("error"),
                            "b0": res.get("b0"), "b0_acc": res.get("b0_acc"),
                            "b0_logit_std": res.get("b0_logit_std"),
                            "hook_identity_ok": res.get("hook_identity_ok"),
                            "refs": res.get("refs"), "results": res.get("results"),
                        }
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[失败] {key}: {e}")
                        summary["runs"][key] = {"error": str(e)[:400]}
                    _save(summary_path, summary)
    finally:
        log_saver.close()
        sys.stdout = log_saver.terminal

    print(f"[完成] 汇总: {summary_path}")
    summarize(base_dir, args)


if __name__ == "__main__":
    main()
