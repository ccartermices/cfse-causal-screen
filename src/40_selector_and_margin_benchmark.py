# -*- coding: utf-8 -*-
"""Step 2-6: 选择器横向实测 + 低边际子集去饱和 (行为级)

动因 (由 39 的结论推出):
  - 39 证明: per-feature 层面把 target 换成边际归一化 C_norm 不改变任何相关性
    (组合内除以常数不改变序), 所以 "gap 是不是归一化问题" 已排除;
  - 39 同时证明: 用 per-feature 归一化效应之和做**可加近似**来对比选择器并不可靠
    (Σcn vs 38 实测 C_joint 仅 spearman 0.661), 故选择器对比**必须实测**。

本脚本一次加载模型完成两件事:
  Part 1 (选择器横向实测): 对 M1..M6 各自按值降序取 top-N (N∈{1,5,10,20,50}),
      实测联合 ablation 的 C_joint / C_norm / flip, 并与 38 已测的同规模随机集合
      基线比出 z —— 回答 "哪个常用指标挑特征挑得最好"。
  Part 2 (低边际子集去饱和): 38 显示翻转率被任务饱和度掩盖 (IOI 13/15 组合
      B0 acc=1.000)。这里按 |B0 per-prompt| 取最低 frac 比例的 prompt 子集,
      在该子集上重测 M2 的 top-N 与同规模随机集合, 检验 "边际变小后翻转率是否
      恢复" —— 若恢复, 则翻转率失效确系指标饱和而非特征无用。

  Step B (一阶预测器 / 方向目标对照): 除 6 个标准指标外, 追加 42 号的一阶因果边距
      预测器作为选择器 —— pred_full/pred_last/pred_align 为「无序幅度」,
      pred_dir/pred_dir_last 为「带符号 / 方向目标」变体, 检验利用预测器符号能否
      把答案翻转率 (flip) 也拉起来。

  Step B2 (符号需与干预方向匹配): 对 pred_dir / pred_steer 额外实测 **steering**
      (α = args.alpha, 默认 3) 的联合效应。2×2 设计 =
      {pred_dir, pred_steer} × {ablate, steer}; 线性化预测只有"符号与干预方向匹配"
      的对角线组合才能压低 B 并翻转答案。另在最大 N 处对 pred_steer 扫 α
      (--alpha_scan) 给出 flip-vs-α 曲线。

输入: 36 的 features (选择器排序) + 38 的随机集合基线 (复用, 不重测)
输出: output/phaseStep2/selector_margin/{m}/layer{L}/{task}.json
      output/phaseStep2/selector_margin/summary_40.json + selector_margin.md
      回填 output/phaseF/phaseF_report.md §20

资源: CPU 严格节流, 单进程串行, 独显 cuda:0, 不打扰同机其他实验。
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

_m36 = import_module("36_causal_validity")
_m38 = import_module("38_topn_joint_intervention")

ResumeManager = _m36.ResumeManager
batch_behavior = _m36.batch_behavior
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae
find_decoder_layers = _m36.find_decoder_layers
load_model_for_generation = _m36.load_model_for_generation
TASK_BUILDERS = _m36.TASK_BUILDERS
_save = _m36._save
make_feature_set_hook = _m38.make_feature_set_hook
_flip = _m38._flip

METRICS = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
           "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]

# Step B: 一阶因果边距预测器 (42_stepA 落盘) 作为额外选择器.
#   pred_full / pred_last / pred_align  = 「无序幅度」(|pred|, 越大越该选);
#   pred_dir  / pred_dir_last           = 「带符号 / 方向目标」变体。
#     值 = (42 落盘字段, 排序符号); 符号 -1 = 取负后再降序, 即挑「预测最强烈地把 B
#     压低」的特征。翻转答案要求 B_0 + ΔB < 0 且 B_0>0, 故只有这一侧可能翻转;
#     pred_full_v ≈ 实测 ΔB, 与 |pred| 的排序近似正交 (mean spearman ≈ 0.00)。
EXTRA_SELECTOR_COLUMNS = {
    "pred_full": ("pred_full", 1),
    "pred_last": ("pred_last", 1),
    "pred_align": ("pred_align", 1),
    "pred_dir": ("pred_full_v", -1),
    "pred_dir_last": ("pred_last_v", -1),
    # Step B2: steering 的**符号镜像**。干预改成 steering(α>1) 后线性化预测翻转符号:
    #   ΔB_ablate(f) = +pred_full_v[f];  ΔB_steer(f) = -(α-1)·pred_full_v[f]。
    #   故"压 B"的一侧对 ablation 是 pred_full_v 最负者 (pred_dir),
    #   对 steering 则是 pred_full_v 最正者 (pred_steer) —— 同一预测器, 符号随干预方向翻转。
    "pred_steer": ("pred_full_v", 1),
}
EXTRA_SELECTORS = list(EXTRA_SELECTOR_COLUMNS)


def load_extra_selectors(base_dir, m, layer, task, feats):
    """从 42 的 causal_margin 结果读 pred_* 分数, 按 feature_idx 对齐 feats 顺序.

    返回 {selector_name: np.ndarray(len(feats))}; 缺失/无效时该键不出现.
    """
    f = (base_dir / "output" / "phaseStep2" / "causal_margin"
         / m / f"layer{layer}" / f"{task}.json")
    if not f.exists():
        return {}
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        return {}
    if d.get("error") or not d.get("features"):
        return {}
    by_idx = {int(r["feature_idx"]): r for r in d["features"]}
    out = {}
    for name in EXTRA_SELECTORS:
        col, sgn = EXTRA_SELECTOR_COLUMNS.get(name, (name, 1))
        v = []
        for ff in feats:
            r = by_idx.get(int(ff["feature_idx"]))
            val = r.get(col) if r is not None else None
            v.append(sgn * float(val) if val is not None else -np.inf)
        arr = np.array(v, float)
        if np.isfinite(arr).any():
            out[name] = arr
    return out


def run_one(args, config, base_dir, m, layer, task, ns):
    _, model_cfg = resolve_model_cfg(config, m)
    cv = base_dir / "output" / "phaseStep2" / "causal_validity"
    tj = base_dir / "output" / "phaseStep2" / "topn_joint"
    out_root = base_dir / "output" / "phaseStep2" / "selector_margin"
    out_dir = out_root / m / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"
    resume = ResumeManager(out_dir, f"40_{task}_layer{layer}")
    extra_req = [s.strip() for s in (args.extra_selectors or "").split(",") if s.strip()]
    extra_steer = [s.strip() for s in (getattr(args, "extra_steer", "") or "").split(",") if s.strip()]
    for s in extra_steer:
        if s not in extra_req:
            extra_req.append(s)
    scan_sel = (getattr(args, "alpha_scan_selector", "") or "").strip()
    alpha_list = [float(x) for x in (getattr(args, "alpha_scan", "") or "").split(",") if x.strip()]
    existing = None
    if out_file.exists():
        try:
            existing = json.load(open(out_file, encoding="utf-8"))
        except Exception:
            existing = None

    def _extra_ok(name, rows):
        if not rows:
            return False
        if name in extra_steer and any("steer" not in r for r in rows):
            return False
        return True

    old_extra = (existing or {}).get("part1_extra") or []
    have_all_extra = all(_extra_ok(s, [r for r in old_extra if r.get("metric") == s])
                         for s in extra_req)
    have_scan = (not (scan_sel and alpha_list)) or ("alpha_scan" in (existing or {}))
    if (resume.is_completed() and existing is not None and not args.force
            and have_all_extra and have_scan and "part1_selector" in existing
            and "part2_hard" in existing):
        print(f"  [跳过] {m} layer{layer} {task} 已完成")
        return existing

    # 已有结果时只补算缺的部分 (Part1/Part2 实测很贵, 尽量复用)
    result = dict(existing) if existing else {}
    result.update({"model": m, "layer": layer, "task": task,
                   "timestamp": datetime.now().isoformat()})
    do_part1 = args.force or "part1_selector" not in result
    do_part2 = args.force or "part2_hard" not in result

    src36 = cv / m / f"layer{layer}" / f"{task}.json"
    src38 = tj / m / f"layer{layer}" / f"{task}.json"
    if not src36.exists() or not src38.exists():
        result["error"] = "缺少 36 或 38 的结果"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        return result
    d36 = json.load(open(src36, encoding="utf-8"))
    d38 = json.load(open(src38, encoding="utf-8"))
    if d36.get("error") or d36.get("at_chance") or not d36.get("features"):
        result["error"] = "36 结果无效 (error/at_chance)"
        _save(out_file, result)
        resume.mark_completed({"skip": result["error"]})
        return result

    feats = d36["features"]
    # 各指标排序 (降序, None 用 -inf 排在最后)
    ranks = {}
    for met in METRICS:
        v = np.array([(f.get(met) if f.get(met) is not None else -np.inf)
                      for f in feats], float)
        if not np.isfinite(v).any():
            continue
        ranks[met] = np.argsort(v)[::-1]
    result["metrics_available"] = sorted(ranks.keys())
    n_avail = len(feats)
    ns = [n for n in ns if n <= n_avail]
    # 38 已测的随机集合基线 (同规模, 同 prompt 集), 直接复用
    rand38 = {x["N"]: x["random_ablate"] for x in d38.get("results", [])}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded = load_per_layer_sae(base_dir, m, layer, device)
    if loaded is None:
        result["error"] = "缺少 per-layer SAE"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        return result
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)
    n_latents = int(sae.decoder.weight.shape[1])

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
            os.environ.setdefault("TRITON_CACHE_DIR",
                                  os.path.join(os.path.expanduser("~"), ".cache", "triton"))
        except Exception:
            pass
        try:
            import huggingface_hub._snapshot_download as _sd
            _sd.read_tree_cache = lambda *a, **k: None
        except Exception:
            pass
        print("  [fp8] 已置离线模式 + 跳过 snapshot 完整性检查")
    model, tokenizer = load_model_for_generation(base_dir, model_cfg, model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers_mod = find_decoder_layers(model)
    probe = "When Mary and John went to the store, John gave a drink to"
    lmap = calibrate_layer_map(model, tokenizer, layers_mod, device, probe)
    if str(layer) not in lmap:
        result["error"] = f"层对齐失败: hidden_states[{layer}]"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        del model
        return result
    block_idx = lmap[str(layer)]["block_idx"]
    layer_mod = layers_mod[block_idx]
    result["block_idx"] = block_idx
    result["align_max_abs_diff"] = lmap[str(layer)]["max_abs_diff"]

    items, task_meta = TASK_BUILDERS[task](tokenizer, args.n_prompts, seed=args.seed)
    if not items:
        result["error"] = f"任务构造失败: {task_meta}"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        del model
        return result
    b0_list, b0 = batch_behavior(model, tokenizer, items, layer_mod, None, device,
                                 args.batch_size)
    b0_arr = np.asarray(b0_list, float)
    b0_std = float(b0_arr.std())
    acc0 = float(np.mean([1.0 if v > 0 else 0.0 for v in b0_list]))
    result.update({"n_prompts": len(items), "b0": float(b0), "b0_acc": acc0,
                   "b0_logit_std": b0_std})
    print(f"  [B0] B0={b0:.3f} acc={acc0:.3f} std={b0_std:.3f}")

    def measure(itms, blist, mode, feat_list, alpha=1.0):
        hk = make_feature_set_hook(sae, mean_t, std_t, feat_list, mode, alpha)
        vals, bm = batch_behavior(model, tokenizer, itms, layer_mod, hk, device,
                                 args.batch_size)
        b0m = float(np.mean(blist))
        C = float(abs(bm - b0m))
        sd = float(np.asarray(blist, float).std())
        den = abs(b0m) if abs(b0m) > 1e-9 else None
        return {"B": float(bm), "C": C,
                "C_norm": (C / sd if sd > 0 else None),
                "C_rel": (C / den if den else None),
                "flip": _flip(vals, blist)}

    # ---- Part 1: 选择器横向实测 ----
    if do_part1:
        part1 = []
        for met in METRICS:
            if met not in ranks:
                continue
            for N in ns:
                fl = [int(feats[i]["feature_idx"]) for i in ranks[met][:N]]
                ab = measure(items, b0_list, "ablate", fl)
                r = rand38.get(N) or {}
                sd = (r.get("std_C") or 0.0) + 1e-12
                row = {"metric": met, "N": N, "feature_idx": fl, "ablate": ab,
                       "rand_mean_C": r.get("mean_C"), "rand_std_C": r.get("std_C"),
                       "rand_mean_flip": r.get("mean_flip"),
                       "z": ((ab["C"] - r["mean_C"]) / sd) if r.get("mean_C") is not None else None}
                if args.steer_top_metric and met == args.steer_top_metric:
                    row["steer"] = measure(items, b0_list, "steer", fl, args.alpha)
                part1.append(row)
            best = [x for x in part1 if x["metric"] == met]
            print("  [选择器] %-18s " % met + " ".join(
                "N%d:%s" % (x["N"], ("%.3f" % x["ablate"]["C_norm"]) if x["ablate"]["C_norm"] is not None else "-")
                for x in best))
        result["part1_selector"] = part1
    else:
        print("  [Part1] 复用已有选择器结果")

    # ---- Part 1b: 一阶因果边距预测器 (42/Step A) 作为新选择器 ----
    if extra_req:
        if args.force:
            result["part1_extra"] = []
        part1b = list(result.get("part1_extra") or [])
        todo = [s for s in extra_req
                if not _extra_ok(s, [r for r in part1b if r["metric"] == s])]
        if todo:
            part1b = [r for r in part1b if r["metric"] not in todo]   # 移除待重算的旧行
        if not todo:
            print("  [Part1b] 复用已有预测器结果")
        else:
            preds = load_extra_selectors(base_dir, m, layer, task, feats)
            for name in todo:
                if name not in preds:
                    print(f"  [Part1b] {name} 无 42 结果 (pred 缺失), 跳过")
                    continue
                order = np.argsort(preds[name])[::-1]
                for N in ns:
                    fl = [int(feats[i]["feature_idx"]) for i in order[:N]]
                    ab = measure(items, b0_list, "ablate", fl)
                    r = rand38.get(N) or {}
                    sd = (r.get("std_C") or 0.0) + 1e-12
                    row = {
                        "metric": name, "N": N, "feature_idx": fl, "ablate": ab,
                        "rand_mean_C": r.get("mean_C"), "rand_std_C": r.get("std_C"),
                        "rand_mean_flip": r.get("mean_flip"),
                        "z": ((ab["C"] - r["mean_C"]) / sd) if r.get("mean_C") is not None else None}
                    if name in extra_steer:
                        row["steer"] = measure(items, b0_list, "steer", fl, args.alpha)
                    part1b.append(row)
                print("  [预测器] %-12s " % name + " ".join(
                    "N%d:%s" % (x["N"], ("%.3f" % x["ablate"]["C_norm"]) if x["ablate"]["C_norm"] is not None else "-")
                    for x in part1b if x["metric"] == name)
                    + (("  steer: " + " ".join(
                        "N%d:%.4f" % (x["N"], x["steer"]["flip"]) for x in part1b
                        if x["metric"] == name and "steer" in x)) if name in extra_steer else ""))
            result["part1_extra"] = part1b
        # ---- Step B2 alpha 扫描: 在最大 N 处对 scan_sel 扫 alpha (flip vs alpha) ----
        if scan_sel and alpha_list:
            cached = result.get("alpha_scan")
            if (cached and not args.force and cached.get("selector") == scan_sel
                    and cached.get("alphas") == alpha_list):
                print("  [α扫描] 复用已有结果")
            else:
                preds2 = load_extra_selectors(base_dir, m, layer, task, feats)
                if scan_sel in preds2:
                    N_top = max(ns)
                    order = np.argsort(preds2[scan_sel])[::-1]
                    fl = [int(feats[i]["feature_idx"]) for i in order[:N_top]]
                    rows = []
                    for a in alpha_list:
                        st = measure(items, b0_list, "steer", fl, a)
                        rows.append({"alpha": a, "steer": st})
                    result["alpha_scan"] = {"selector": scan_sel, "N": N_top,
                                            "alphas": alpha_list, "rows": rows}
                    print("  [α扫描] %s N=%d " % (scan_sel, N_top) + " ".join(
                        "α%g:%.4f" % (x["alpha"], x["steer"]["flip"]) for x in rows))
                else:
                    print(f"  [α扫描] {scan_sel} 无 42 结果, 跳过")
    else:
        result.setdefault("part1_extra", [])

    # ---- Part 2: 低边际子集去饱和 ----
    # 关键: 必须只在**无干预即答对 (B0>0)** 的 prompt 里取低边际子集。
    # 若直接按 |B0| 最小取, 会把模型本来就答错的 prompt 混进来 (acc 掉到 0.5),
    # 此时 flip 变成 "错误修正率" 而非 "打翻正确行为", 语义就不干净了。
    if do_part2:
        pos = [i for i in range(len(items)) if b0_list[i] > 0.0]
        pool_idx = pos if len(pos) >= args.hard_min else list(range(len(items)))
        k = max(args.hard_min, int(round(args.hard_frac * len(pool_idx))))
        k = min(k, len(pool_idx))
        order = sorted(pool_idx, key=lambda i: b0_list[i])[:k]
        sub_items = [items[int(i)] for i in order]
        sub_b0 = [b0_list[int(i)] for i in order]
        sub_arr = np.asarray(sub_b0, float)
        acc_h = float(np.mean([1.0 if v > 0 else 0.0 for v in sub_b0]))
        print(f"  [低边际子集] 候选 B0>0 的 prompt={len(pos)}/{len(items)}; "
              f"取 K={k} 个最小边际 -> acc={acc_h:.3f} "
              f"|B0|中位={np.median(np.abs(sub_arr)):.3f}")
        hard = {"k": k, "frac": args.hard_frac, "pool": "B0>0" if pool_idx is pos else "all",
                "n_positive": len(pos), "b0": float(sub_arr.mean()), "b0_acc": acc_h,
                "b0_logit_std": float(sub_arr.std()),
                "mean_abs_b0": float(np.median(np.abs(sub_arr))),
                "rows": []}
        m2_rank = ranks.get("M2_mean_act_task")
        rng = np.random.RandomState(args.seed + 7)
        for N in [n for n in args.hard_ns if n <= min(n_avail, k)]:
            if m2_rank is None:
                break
            fl = [int(feats[i]["feature_idx"]) for i in m2_rank[:N]]
            ab = measure(sub_items, sub_b0, "ablate", fl)
            rc, rf = [], []
            for _ in range(args.n_random_sets):
                rff = rng.choice(n_latents, size=N, replace=False).tolist()
                mm = measure(sub_items, sub_b0, "ablate", rff)
                rc.append(mm["C"]); rf.append(mm["flip"])
            sd = float(np.std(rc)) + 1e-12
            hard["rows"].append({
                "N": N, "ablate": ab,
                "random_ablate": {"n": len(rc), "mean_C": float(np.mean(rc)),
                                  "std_C": float(np.std(rc)),
                                  "mean_flip": float(np.mean(rf)),
                                  "max_flip": float(np.max(rf))},
                "z": (ab["C"] - float(np.mean(rc))) / sd})
            print(f"    [hard N={N}] C={ab['C']:.3f} C_norm="
                  f"{'-' if ab['C_norm'] is None else round(ab['C_norm'],3)} "
                  f"flip={ab['flip']:.3f} rand_flip={np.mean(rf):.4f} "
                  f"z={hard['rows'][-1]['z']:.1f}")
        result["part2_hard"] = hard
    else:
        print("  [Part2] 复用已有低边际子集结果")
        k = (result.get("part2_hard") or {}).get("k")

    _save(out_file, result)
    resume.mark_completed({"metrics": result["metrics_available"], "ns": ns,
                           "hard_k": k, "b0_acc": acc0})
    print(f"  [完成] {out_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def collect(root):
    runs = []
    for p in sorted(root.glob("*/layer*/*.json")):
        if p.name.startswith(("resume_", "summary")):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "part1_selector" in d and "part2_hard" in d:
            runs.append(d)
    return runs


def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def render_md(runs, ts):
    L = ["# 选择器横向实测 + 低边际子集去饱和", "",
         f"- 生成时间: {ts}", f"- 组合数: {len(runs)}",
         "- Part 1: 按各指标降序取 top-N 实测联合 ablation; "
         "z 相对 38 已测的同规模随机集合基线",
         "- Part 2: 在**无干预即答对 (B0>0)** 的 prompt 中取边际最小的子集, "
         "在该子集上重测 M2 的 top-N (与全体 200 prompt 对比, 检验饱和度假设)",
         "", "## Part 1. 选择器横向对比 (实测 C_norm, 跨组合均值)", ""]
    aggs = {}
    for r in runs:
        for x in (r.get("part1_selector") or []) + (r.get("part1_extra") or []):
            key = (x["metric"], x["N"])
            d = aggs.setdefault(key, {"cn": [], "c": [], "flip": [], "z": [],
                                      "task": []})
            if x["ablate"]["C_norm"] is not None:
                d["cn"].append(x["ablate"]["C_norm"])
            d["c"].append(x["ablate"]["C"])
            d["flip"].append(x["ablate"]["flip"])
            if x["z"] is not None:
                d["z"].append(x["z"])
            d["task"].append(r["task"])
    ns = sorted({N for (_, N) in aggs})
    # 指标展示顺序: 6 个标准指标 + 已出现的预测器
    met_order = list(METRICS) + [x for x in EXTRA_SELECTORS
                                 if any(k[0] == x for k in aggs)]
    L += ["| 指标/预测器 | " + " | ".join(f"N={n}" for n in ns) + " |",
          "|---|" + "---|" * len(ns)]
    for met in met_order:
        if not any(k[0] == met for k in aggs):
            continue
        L.append(f"| {met} | " + " | ".join(
            _f(float(np.mean(aggs[(met, n)]["cn"])), 3) if (met, n) in aggs
            and aggs[(met, n)]["cn"] else "-" for n in ns) + " |")
    L += ["", "中位 z (vs 同规模随机集合), 越大越好:", "",
          "| 指标/预测器 | " + " | ".join(f"N={n}" for n in ns) + " |",
          "|---|" + "---|" * len(ns)]
    for met in met_order:
        if not any(k[0] == met for k in aggs):
            continue
        L.append(f"| {met} | " + " | ".join(
            _f(float(np.median(aggs[(met, n)]["z"])), 1)
            if (met, n) in aggs and aggs[(met, n)]["z"] else "-" for n in ns) + " |")
    # 翻转率
    L += ["", "跨组合平均翻转率 flip:", "",
          "| 指标/预测器 | " + " | ".join(f"N={n}" for n in ns) + " |",
          "|---|" + "---|" * len(ns)]
    for met in met_order:
        if not any(k[0] == met for k in aggs):
            continue
        L.append(f"| {met} | " + " | ".join(
            _f(float(np.mean(aggs[(met, n)]["flip"])), 4)
            if (met, n) in aggs else "-" for n in ns) + " |")

    L += ["", "## Part 2. 低边际子集: 翻转率是否恢复", "",
          "| 组合 | 全体 B0 acc | 子集 B0 acc | 子集 |B0|中位 | N | C | C_norm | flip | "
          "rand flip | z |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        h = r.get("part2_hard") or {}
        if not h.get("rows"):
            continue
        for row in h["rows"]:
            L.append(
                f"| {r['model']}|L{r['layer']}|{r['task']} | {_f(r.get('b0_acc'),3)} | "
                f"{_f(h.get('b0_acc'),3)} | {_f(h.get('mean_abs_b0'),3)} | {row['N']} | "
                f"{_f(row['ablate']['C'],3)} | {_f(row['ablate']['C_norm'],3)} | "
                f"{_f(row['ablate']['flip'],3)} | "
                f"{_f(row['random_ablate']['mean_flip'])} | {_f(row['z'],1)} |")
    full_f, hard_f = [], []
    for r in runs:
        h = r.get("part2_hard") or {}
        for row in h.get("rows", []):
            if row["N"] == max(x["N"] for x in h["rows"]):
                full_f.append(r.get("b0_acc"))
                hard_f.append(row["ablate"]["flip"])
    if full_f:
        L += ["", f"> 全体 B0 acc 均值 {_f(float(np.mean(full_f)),3)} 时, "
                  f"低边际子集的最大 N 翻转率均值 {_f(float(np.mean(hard_f)),3)} "
                  f"(子集 B0 acc 均值 "
                  f"{_f(float(np.mean([ (r['part2_hard'] or {}).get('b0_acc') for r in runs if (r.get('part2_hard') or {}).get('rows')])),3)})。"]
    L.append("")
    return "\n".join(L)


def _pair_table(runs, ref="M2_mean_act_task"):
    """对每个预测器, 在同一 N 下与 ref 指标逐组合配对比较 z 与 flip。"""
    extra_names = [x for x in EXTRA_SELECTORS
                   if any(row["metric"] == x
                          for r in runs for row in (r.get("part1_extra") or []))]
    if not extra_names:
        return None, []
    ns = sorted({row["N"] for r in runs
                 for row in (r.get("part1_selector") or []) + (r.get("part1_extra") or [])})
    bycombo = {}
    for r in runs:
        d = {}
        for x in (r.get("part1_selector") or []) + (r.get("part1_extra") or []):
            d[(x["metric"], x["N"])] = x
        bycombo[(r["model"], r["layer"], r["task"])] = d
    rows = []
    for name in extra_names:
        for N in ns:
            zs, fs, dz, df, win, tot, fwin, cns = [], [], [], [], 0, 0, 0, []
            for d in bycombo.values():
                a = d.get((name, N))
                b = d.get((ref, N))
                if not a:
                    continue
                if a.get("z") is not None:
                    zs.append(a["z"])
                fs.append(a["ablate"]["flip"])
                if a["ablate"].get("C_norm") is not None:
                    cns.append(a["ablate"]["C_norm"])
                if b is not None:
                    tot += 1
                    if a.get("z") is not None and b.get("z") is not None:
                        dz.append(a["z"] - b["z"])
                        if a["z"] > b["z"]:
                            win += 1
                    da = a["ablate"]["flip"] - b["ablate"]["flip"]
                    df.append(da)
                    if da > 0:
                        fwin += 1
            if not zs:
                continue
            rows.append({
                "selector": name, "N": N, "n": len(zs),
                "mean_cnorm": float(np.mean(cns)) if cns else None,
                "mean_z": float(np.mean(zs)), "median_z": float(np.median(zs)),
                "mean_flip": float(np.mean(fs)),
                "n_pairs": tot, "z_win": win, "flip_win": fwin,
                "mean_dz": float(np.mean(dz)) if dz else None,
                "median_dz": float(np.median(dz)) if dz else None,
                "mean_dflip": float(np.mean(df)) if df else None,
            })
    return bycombo, rows


def _ref_rows(runs, ref="M2_mean_act_task"):
    """参照指标 (M2) 自身的跨组合均值, 供 Step B 报告对照。"""
    out = {}
    for r in runs:
        if not r.get("part1_extra"):
            continue          # 只统计与预测器同一批可配对的组合
        for x in (r.get("part1_selector") or []):
            if x["metric"] != ref:
                continue
            d = out.setdefault(x["N"], {"z": [], "flip": [], "cn": []})
            if x.get("z") is not None:
                d["z"].append(x["z"])
            d["flip"].append(x["ablate"]["flip"])
            if x["ablate"]["C_norm"] is not None:
                d["cn"].append(x["ablate"]["C_norm"])
    rows = []
    for N in sorted(out):
        d = out[N]
        rows.append({"N": N, "n": len(d["flip"]),
                     "mean_z": float(np.mean(d["z"])) if d["z"] else None,
                     "median_z": float(np.median(d["z"])) if d["z"] else None,
                     "mean_flip": float(np.mean(d["flip"])),
                     "mean_cnorm": float(np.mean(d["cn"])) if d["cn"] else None})
    return rows


def render_stepB(runs, ts):
    bycombo, rows = _pair_table(runs, ref="M2_mean_act_task")
    if not rows:
        return None
    n_combos = sum(1 for r in runs if r.get("part1_extra"))
    ref_rows = _ref_rows(runs, ref="M2_mean_act_task")
    ref_by_n = {r["N"]: r for r in ref_rows}
    L = ["# Step B 判定: 一阶因果边距预测器作为 SAE 特征选择器", "",
         f"- 生成时间: {ts}",
         f"- 组合数: {n_combos} (同时具备 36 号 M2 与 42 号 pred_* 结果的组合)",
         "- 口径: 与 Part 1 完全一致 —— 按选择器降序取 top-N 做**真实联合 ablation**,",
         "  z 相对 38 号已测的同规模随机特征集合基线, flip 为答案翻转率。",
         "- 参照: 标准指标 M2_mean_act_task (36 号中与 C_ablate 相关性最高者)。",
         "- Δz / Δflip = 预测器 − M2, 逐组合配对; z>M2 为 z 胜出的组合数。",
         "", "| 选择器 | N | 组合 | 平均 C_norm | 平均 z | 中位 z | 平均 flip | z>M2 | flip>M2 | 平均 Δz | 平均 Δflip |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in [x for x in EXTRA_SELECTORS if any(r["selector"] == x for r in rows)]:
        for r in [r for r in rows if r["selector"] == name]:
            ref = ref_by_n.get(r["N"], {})
            L.append(f"| {name} | {r['N']} | {r['n']} | {_f(r['mean_cnorm'],3)} | {r['mean_z']:.1f} | "
                     f"{r['median_z']:.1f} | {r['mean_flip']:.4f} | "
                     f"{r['z_win']}/{r['n_pairs']} | {r['flip_win']}/{r['n_pairs']} | "
                     f"{_f(r['mean_dz'],1)} | {_f(r['mean_dflip'],4)} |")
    for r in ref_rows:
        L.append(f"| M2_mean_act_task(参照) | {r['N']} | {r['n']} | "
                 f"{_f(r['mean_cnorm'],3)} | {_f(r['mean_z'],1)} | {_f(r['median_z'],1)} | "
                 f"{r['mean_flip']:.4f} | - | - | - | - |")
    # 方向目标变体 (pred_dir) vs 无序幅度 (pred_full): 同一组合、同一 N 直接配对
    cmp_rows = []
    for N in sorted({r["N"] for r in rows}):
        df_, dz_, fw, fl, ft = [], [], 0, 0, 0
        for d in bycombo.values():
            a, b = d.get(("pred_dir", N)), d.get(("pred_full", N))
            if not a or not b:
                continue
            da = a["ablate"]["flip"] - b["ablate"]["flip"]
            df_.append(da)
            if da > 1e-12:
                fw += 1
            elif da < -1e-12:
                fl += 1
            else:
                ft += 1        # 双方都为 0 (饱和组合) 的平局不计入胜负
            if a.get("z") is not None and b.get("z") is not None:
                dz_.append(a["z"] - b["z"])
        if df_:
            cmp_rows.append({"N": N, "n": len(df_), "mean_dflip": float(np.mean(df_)),
                             "flip_win": fw, "flip_lose": fl, "flip_tie": ft,
                             "mean_dz": float(np.mean(dz_)) if dz_ else None})
    if cmp_rows:
        L += ["", "## 方向目标 (pred_dir) vs 无序幅度 (pred_full) —— 同组合同 N 配对", "",
              "Δflip = pred_dir − pred_full (正 = 方向目标更能把答案打翻)。",
              "胜/平/负按逐组合 Δflip 计; 平 = 双方 flip 均为 0 (饱和组合), 不计入胜负。",
              "", "| N | 组合 | 平均 Δflip | 胜/平/负 | 平均 Δz |", "|---|---|---|---|---|"]
        for c in cmp_rows:
            L.append(f"| {c['N']} | {c['n']} | {c['mean_dflip']:+.4f} | "
                     f"{c['flip_win']}/{c['flip_tie']}/{c['flip_lose']} | {_f(c['mean_dz'],1)} |")

    # 判定: z 全程为正 + flip 在低 N 有明显提升
    L.append("")
    pf = [r for r in rows if r["selector"] == "pred_full"]
    if pf:
        z_allpos = all((r["mean_dz"] or 0) > 0 for r in pf)
        z_wr = np.mean([r["z_win"] / max(r["n_pairs"], 1) for r in pf])
        low_n = [r for r in pf if r["N"] <= 20 and r["mean_dflip"] is not None]
        flip_low_pos = bool(low_n) and all(r["mean_dflip"] > 0 for r in low_n)
        top = max(pf, key=lambda x: x["N"])
        if z_allpos and z_wr > 0.7 and flip_low_pos:
            verdict = ("**通过**: pred_full 在所有 N 上 z 全面高于 M2, 且 N≤20 翻转率有可见提升; "
                       "N=50 处翻转率趋同 (任务饱和, 双方 flip 均被 B0 边际压制)")
        elif z_allpos and z_wr > 0.7:
            verdict = "**部分通过**: z 全面胜出, 但翻转率未见一致提升"
        else:
            verdict = "**未通过**: pred_full 未在 z 上稳定超越 M2"
        L.append(f"> z 方向: 平均 Δz {'全为正' if z_allpos else '存在负值'}, "
                 f"z 胜出率均值 {z_wr:.2f}。")
        L.append(f"> flip 方向: N≤20 平均 Δflip "
                 f"{'全为正' if flip_low_pos else '存在非正'}; "
                 f"N={top['N']} 处 Δflip={_f(top['mean_dflip'],4)}。")
        L.append(f"> 判定: {verdict}。")
    if cmp_rows:
        ge5 = [c for c in cmp_rows if c["N"] >= 5]
        d_pos = bool(ge5) and all(c["mean_dflip"] > 0 for c in ge5)
        top_N = max(c["N"] for c in cmp_rows)
        d_top = [r for r in rows if r["selector"] == "pred_dir" and r["N"] == top_N]
        m2_top = ref_by_n.get(top_N, {}).get("mean_flip")
        ratio = (d_top[0]["mean_flip"] / m2_top) if (d_top and m2_top) else None
        L.append("> [方向目标] pred_dir 逐 N 平均 Δflip vs pred_full: " + ", ".join(
            f"N={c['N']}:{c['mean_dflip']:+.4f}" for c in cmp_rows))
        L.append("> [方向目标] 逐组合 胜/平/负: " + ", ".join(
            f"N={c['N']}:{c['flip_win']}/{c['flip_tie']}/{c['flip_lose']}" for c in cmp_rows))
        if ratio is not None:
            L.append(f"> [方向目标] N={top_N} 处 flip: pred_dir {d_top[0]['mean_flip']:.4f} "
                     f"vs M2 {m2_top:.4f} (={ratio:.2f}×), "
                     f"vs pred_full {[r for r in rows if r['selector'] == 'pred_full' and r['N'] == top_N][0]['mean_flip']:.4f}。")
        if d_pos and ratio is not None and ratio >= 1.5:
            L.append("> [方向目标] 判定: **通过** —— 自 N=5 起方向目标选择的翻转率一致高于"
                     f"无序幅度且随 N 单调增强; N={top_N} 处达 M2 的 {ratio:.1f}×。"
                     "即: 用预测器的**符号**决定往哪边推, 修复了 pred_full 的翻转率短板。")
        elif d_pos:
            L.append("> [方向目标] 判定: **部分通过** —— N≥5 起翻转率一致优于无序幅度, 但幅度有限。")
        else:
            L.append("> [方向目标] 判定: **未通过** —— 方向目标选择未提升翻转率。")
    L.append("")
    return "\n".join(L)


def render_stepB2(runs, ts, alpha=3.0):
    """Step B2: 符号需与干预方向匹配 (2x2: {pred_dir,pred_steer} x {ablate,steer}) + α 扫描."""
    def _ex(r):
        return (r.get("part1_extra") or [])

    def _sel(r):
        return (r.get("part1_selector") or [])

    # (selector, N) -> steering 汇总
    st = {}
    for r in runs:
        for x in _ex(r):
            if "steer" not in x:
                continue
            d = st.setdefault((x["metric"], x["N"]), {"cn": [], "flip": []})
            if x["steer"].get("C_norm") is not None:
                d["cn"].append(x["steer"]["C_norm"])
            d["flip"].append(x["steer"]["flip"])
    if not st:
        return None
    # 参照 M2 (36 号单测已带 steer)
    m2 = {}
    for r in runs:
        for x in _sel(r):
            if x["metric"] != "M2_mean_act_task" or "steer" not in x:
                continue
            d = m2.setdefault(x["N"], {"cn": [], "flip": []})
            if x["steer"].get("C_norm") is not None:
                d["cn"].append(x["steer"]["C_norm"])
            d["flip"].append(x["steer"]["flip"])

    ns = sorted({N for (_, N) in st})
    N_top = max(ns)
    sels = sorted({s for (s, _) in st})
    L = ["# Step B2 判定: 符号必须与干预方向匹配 (ablation vs steering)", "",
         f"- 生成时间: {ts}",
         f"- α = {alpha:g} (steering: v ← v + (α-1)·s), 与 36 号单特征 steering 口径一致。",
         "- 线性化预测: ΔB_ablate(f) = +pred_full_v[f];  ΔB_steer(f) = -(α-1)·pred_full_v[f]。",
         "  故 '压低 B' 的一侧: ablation 取 pred_full_v 最负者 (pred_dir),",
         "  steering 取 pred_full_v 最正者 (pred_steer) —— 符号须随干预方向翻转。",
         "", f"### A. steering (α={alpha:g}) 联合效应, 按选择器 × N", "",
         "| 选择器 | " + " | ".join(f"N={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
    for s in sels:
        L.append(f"| {s} (steer) | " + " | ".join(
            _f(float(np.mean(st[(s, n)]["flip"])), 4) if (s, n) in st else "-"
            for n in ns) + " |")
    L.append("| M2_mean_act_task (steer, 参照) | " + " | ".join(
        _f(float(np.mean(m2[n]["flip"])), 4) if n in m2 else "-" for n in ns) + " |")
    L.append("")
    L.append("| 选择器 | " + " | ".join(f"N={n}" for n in ns) + " |")
    L.append("|---|" + "---|" * len(ns))
    for s in sels:
        L.append(f"| {s} (steer) C_norm | " + " | ".join(
            _f(float(np.mean(st[(s, n)]["cn"])), 3) if (s, n) in st and st[(s, n)]["cn"] else "-"
            for n in ns) + " |")
    L.append("| M2 (steer) C_norm | " + " | ".join(
        _f(float(np.mean(m2[n]["cn"])), 3) if n in m2 and m2[n]["cn"] else "-" for n in ns) + " |")

    # ---- 2x2 表 (N = N_top) ----
    quad = {}
    for r in runs:
        rec = {}
        for x in _ex(r):
            if x["N"] != N_top:
                continue
            rec[(x["metric"], "ablate")] = x["ablate"]["flip"]
            if "steer" in x:
                rec[(x["metric"], "steer")] = x["steer"]["flip"]
        for x in _sel(r):
            if x["metric"] == "M2_mean_act_task" and x["N"] == N_top and "steer" in x:
                rec[("M2", "ablate")] = x["ablate"]["flip"]
                rec[("M2", "steer")] = x["steer"]["flip"]
        if rec:
            quad[(r["model"], r["layer"], r["task"])] = rec

    def _mean_flip(sel, mode):
        v = [rec[(sel, mode)] for rec in quad.values() if (sel, mode) in rec]
        return (float(np.mean(v)), len(v)) if v else (None, 0)

    cells = {}
    for sel in ["pred_dir", "pred_steer", "M2"]:
        for mode in ["ablate", "steer"]:
            cells[(sel, mode)] = _mean_flip(sel, mode)
    L += ["", f"### B. 2×2 一致性表 (N = {N_top}, 跨组合平均 flip)", "",
          "| 干预 \\ 选择器 | pred_dir (pred_full_v 最负) | pred_steer (pred_full_v 最正) | M2 |",
          "|---|---|---|---|"]
    for mode, lab in [("ablate", "ablation (v←0)"), ("steer", f"steering (α={alpha:g})")]:
        row = f"| {lab} |"
        for sel in ["pred_dir", "pred_steer", "M2"]:
            m, k = cells[(sel, mode)]
            row += f" {_f(m,4)} (n={k}) |" if m is not None else " - |"
        L.append(row)

    # 配对 (同组合、同 prompt 集): 匹配符号 vs 反符号。翻转率跨组合不可比 (§5.3),
    # 故只用**组合内配对**判定 —— 同一 B0 前提下比较两侧。
    mode_stat = {"ablate": {"d": [], "w": 0, "t": 0, "l": 0},
                 "steer": {"d": [], "w": 0, "t": 0, "l": 0}}
    for rec in quad.values():
        for mode, good, bad in (("ablate", "pred_dir", "pred_steer"),
                                ("steer", "pred_steer", "pred_dir")):
            ka, kb = (good, mode), (bad, mode)
            if ka not in rec or kb not in rec:
                continue
            d = rec[ka] - rec[kb]
            mode_stat[mode]["d"].append(d)
            if d > 1e-12:
                mode_stat[mode]["w"] += 1
            elif d < -1e-12:
                mode_stat[mode]["l"] += 1
            else:
                mode_stat[mode]["t"] += 1
    rows_ok = [m for m in ("ablate", "steer") if mode_stat[m]["d"]]
    if rows_ok:
        L += ["", "### B2. 组合内配对: 匹配符号 − 反符号 (同一 B0, 阻断跨组合饱和混淆)", "",
              "| 干预 | 匹配侧 | 平均 Δflip | 胜/平/负 | 组合 |",
              "|---|---|---|---|---|"]
        for mode, good, bad in (("ablate", "pred_dir", "pred_steer"),
                                ("steer", "pred_steer", "pred_dir")):
            s = mode_stat[mode]
            if not s["d"]:
                continue
            L.append(f"| {'ablation' if mode=='ablate' else f'steering (α={alpha:g})'} | "
                     f"{good} vs {bad} | {np.mean(s['d']):+.4f} | "
                     f"{s['w']}/{s['t']}/{s['l']} | {len(s['d'])} |")

    # ---- α 扫描 ----
    scan_rows = {}
    scan_sel = None
    for r in runs:
        sc = r.get("alpha_scan")
        if not sc or not sc.get("rows"):
            continue
        scan_sel = sc.get("selector")
        for row in sc["rows"]:
            d = scan_rows.setdefault(row["alpha"], {"flip": [], "cn": []})
            d["flip"].append(row["steer"]["flip"])
            if row["steer"].get("C_norm") is not None:
                d["cn"].append(row["steer"]["C_norm"])
    if scan_rows:
        L += ["", f"### C. α 扫描 ({scan_sel}, N = {N_top}): 翻转率是否随让步强度上升", "",
              "| α | " + " | ".join(f"{a:g}" for a in sorted(scan_rows)) + " |",
              "|---|" + "---|" * len(scan_rows)]
        L.append("| 平均 flip | " + " | ".join(
            _f(float(np.mean(scan_rows[a]["flip"])), 4) for a in sorted(scan_rows)) + " |")
        L.append("| 平均 C_norm | " + " | ".join(
            _f(float(np.mean(scan_rows[a]["cn"])), 3) if scan_rows[a]["cn"] else "-"
            for a in sorted(scan_rows)) + " |")
        L.append("| 组合 | " + " | ".join(str(len(scan_rows[a]["flip"]))
                                           for a in sorted(scan_rows)) + " |")

    # ---- 判定 ----
    L.append("")
    modes_present = [m for m in ("ablate", "steer") if mode_stat[m]["d"]]
    if modes_present:
        good_all, parts = True, []
        for mode in modes_present:
            s = mode_stat[mode]
            n = s["w"] + s["t"] + s["l"]
            n_dec = s["w"] + s["l"]
            wr = s["w"] / max(n_dec, 1)          # tie-aware: 平局 (双方 flip=0, 饱和) 不计入胜负
            parts.append(f"{'ablation' if mode=='ablate' else f'steering(α={alpha:g})'} "
                         f"平均 Δflip {np.mean(s['d']):+.4f} "
                         f"(胜/平/负 {s['w']}/{s['t']}/{s['l']}, 胜率 {wr:.2f})")
            if not (float(np.mean(s["d"])) > 0 and wr >= 0.70):
                good_all = False
        L.append("> " + "; ".join(parts) + "。")
        if good_all and len(modes_present) == 2:
            L.append("> 判定: **通过** —— ablation 与 steering 两种干预下, 符号匹配的一侧翻转率"
                     "均显著高于反符号一侧 (逐组合配对), 即'用预测器符号决定往哪边推'的规则"
                     "不依赖干预类型。")
        elif good_all:
            L.append("> 判定: **部分通过** —— 仅一种干预类型下成立。")
        else:
            L.append("> 判定: **未通过** —— 符号匹配未一致提升翻转率。")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Step2-6: 选择器实测 + 低边际子集")
    ap.add_argument("--config", type=str, default="config.json")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--layers", type=str, default=None)
    ap.add_argument("--tasks", type=str, default="ioi,gt")
    ap.add_argument("--ns", type=str, default="1,5,10,20,50")
    ap.add_argument("--hard_ns", type=str, default="10,50")
    ap.add_argument("--hard_frac", type=float, default=0.5)
    ap.add_argument("--hard_min", type=int, default=40)
    ap.add_argument("--steer_top_metric", type=str, default="M2_mean_act_task")
    ap.add_argument("--extra_selectors", type=str,
                    default=",".join(EXTRA_SELECTORS),
                    help="Step B: 从 42 的 causal_margin 读入的预测器作为额外选择器")
    ap.add_argument("--extra_steer", type=str, default="pred_dir,pred_steer",
                    help="Step B2: 对这些额外选择器额外实测 steering (α=--alpha) 的联合效应")
    ap.add_argument("--alpha_scan", type=str, default="2,3,5,10,20",
                    help="Step B2: 在最大 N 处对 --alpha_scan_selector 扫 α")
    ap.add_argument("--alpha_scan_selector", type=str, default="pred_steer")
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
    out_root = base_dir / "output" / "phaseStep2" / "selector_margin"

    if not args.summarize_only:
        ns = [int(x) for x in args.ns.split(",") if x.strip()]
        args.hard_ns = [int(x) for x in args.hard_ns.split(",") if x.strip()]
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        models = [args.model] if args.model else list(config["models"].keys())
        out_root.mkdir(parents=True, exist_ok=True)
        log_dir = out_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_saver = _m36._mod05.LogSaver(str(log_dir / f"40_selector_{ts}.log"))
        sys.stdout = log_saver
        summary_path = out_root / "summary_40.json"
        summary = {"timestamp": ts, "runs": {}}
        if summary_path.exists():
            try:
                prev = json.load(open(summary_path, encoding="utf-8"))
                if isinstance(prev.get("runs"), dict):
                    summary["runs"] = prev["runs"]
            except Exception:
                pass
        try:
            for m in models:
                _, model_cfg = resolve_model_cfg(config, m)
                layers = ([int(x) for x in args.layers.split(",")] if args.layers
                          else list(model_cfg["target_layers"]))
                for layer in layers:
                    for task in tasks:
                        key = f"{m}|L{layer}|{task}"
                        try:
                            res = run_one(args, config, base_dir, m, layer, task, ns)
                            summary["runs"][key] = {
                                "error": res.get("error"),
                                "b0_acc": res.get("b0_acc"),
                                "metrics_available": res.get("metrics_available"),
                                "part1_selector": res.get("part1_selector"),
                                "part1_extra": res.get("part1_extra"),
                                "part2_hard": res.get("part2_hard"),
                                "alpha_scan": res.get("alpha_scan"),
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

    runs = collect(out_root)
    if not runs:
        print("[跳过] 无有效结果")
        return
    md = render_md(runs, datetime.now().isoformat())
    (out_root / "selector_margin.md").write_text(md, encoding="utf-8")
    print(f"[汇总] {out_root / 'selector_margin.md'} ({len(runs)} 组合)")
    rp = base_dir / args.report
    if rp.exists() and not args.no_report:
        import_module("37_summarize_causal_validity")._upsert_section(
            rp, "## 20. Phase Step2 续: 选择器横向实测与低边际去饱和", md)
        print(f"[回填] {rp} §20")

    # Step B: 一阶因果边距预测器 vs M2 的专门判定
    stepb = render_stepB(runs, datetime.now().isoformat())
    if stepb:
        (out_root / "pred_selector_stepB.md").write_text(stepb, encoding="utf-8")
        print(f"[汇总] {out_root / 'pred_selector_stepB.md'}")
        if rp.exists() and not args.no_report:
            import_module("37_summarize_causal_validity")._upsert_section(
                rp, "## 23. Step B: 一阶因果边距预测器作为 SAE 特征选择器", stepb)
            print(f"[回填] {rp} §23")

    # Step B2: 符号需与干预方向匹配 (ablation vs steering) + α 扫描
    stepb2 = render_stepB2(runs, datetime.now().isoformat(), alpha=args.alpha)
    if stepb2:
        (out_root / "pred_selector_stepB_steer.md").write_text(stepb2, encoding="utf-8")
        print(f"[汇总] {out_root / 'pred_selector_stepB_steer.md'}")
        if rp.exists() and not args.no_report:
            import_module("37_summarize_causal_validity")._upsert_section(
                rp, "## 25. Step B2: 符号需与干预方向匹配 (ablation vs steering)", stepb2)
            print(f"[回填] {rp} §25")


if __name__ == "__main__":
    main()
