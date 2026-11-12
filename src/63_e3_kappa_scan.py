# -*- coding: utf-8 -*-
"""63_e3_kappa_scan.py — E3：跨模型/层的可测性比值 κ 扫描（补 G4 阈值区间）

动机
----
E10–E12 只在 6 个 (model, layer, task) 组合上量了 κ = sd_between / sd_within，
结果两极化：κ≥0.30 的 2 组全部可用（rel 0.66–0.76），κ<0.10 的 4 组全部不可用
（rel 0.05–0.17）。**0.10–0.30 之间一个样本都没有**，所以 G4 的阈值 0.10 只是
"当前无假阳性的保守线"，不是实测边界。

本脚本在层/模型维度上扩展采样点，专门去填这个空档：
  - 同模型换层（qwen3.5-2b L8 / L22，与已测 L16 对照 → 层的贡献）
  - 同族换规模（qwen3.5-0.8b、pythia-2.8b）
  - 异族（qwen2.5-7b-instruct、ministral-3-3b、minicpm5-1b、tinyllama-1.1b）
  - phi-3.5-mini 只跑 gt（其 ioi 在 36 里 b0_acc=0.59，任务本身 at_chance）

只用 f32 重算的效应矩阵（bf16 会把 qwen 类的 κ 压成伪值），因此复用 57 的
`batch_behavior_f32`（hook lm_head 输入，fp32 重算 correct/distractor 两个 logit）。

口径（与 E10 主表对齐）
----------------------
- E[feature, prompt]：消融单特征后 (logit_c - logit_d) 相对 b0 的差
- κ = sd_f(mean_p E) / mean_f(sd_p E)
- rel：|mean_p E| 在 100/100 分割上的 Spearman，20 次随机重采样取均值
  （= 60 的 rel，但在 100-prompt 尺度上）
- rel@200（与主表同口径）：Spearman-Brown 上推 4r/(1+3r)
- ICC(1)：F(1,P-1) 单因素随机效应

输出
----
output/phaseStep2/cfse_kappa/{model}/layer{L}/{task}_E_f32.npz
output/phaseStep2/cfse_kappa/kappa_scan.json / kappa_scan.md（增量合并）

用法
----
python 63_e3_kappa_scan.py                     # 默认 9 个 spec
python 63_e3_kappa_scan.py --spec qwen3.5-2b:8:ioi --n_prompts 200
"""
import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

_m57 = importlib.import_module("57_resolution_fix")
_m53 = importlib.import_module("53_reliability_curve")
_m36 = importlib.import_module("36_causal_validity")
_m48 = importlib.import_module("48_cfse_offline")

batch_behavior_f32 = _m57.batch_behavior_f32
TASK_BUILDERS = _m36.TASK_BUILDERS
find_decoder_layers = _m36.find_decoder_layers
calibrate_layer_map = _m36.calibrate_layer_map
load_per_layer_sae = _m36.load_per_layer_sae
make_feature_hook = _m36.make_feature_hook
load_model_for_generation = _m36.load_model_for_generation
from model_utils import (load_config, resolve_model_cfg, model_needs_fp8,
                         dequantize_fp8_model)

OUT = BASE / "output" / "phaseStep2" / "cfse_kappa"

# (model, layer, task)：覆盖"换层 / 换规模 / 换族"三类变异
DEFAULT_SPECS = [
    ("qwen3.5-2b", 8, "ioi"),
    ("qwen3.5-2b", 22, "ioi"),
    ("qwen3.5-0.8b", 16, "ioi"),
    ("qwen2.5-7b-instruct", 18, "ioi"),
    ("ministral-3-3b", 17, "ioi"),
    ("minicpm5-1b", 16, "ioi"),
    ("pythia-2.8b", 21, "ioi"),
    ("tinyllama-1.1b", 14, "ioi"),
    ("phi-3.5-mini", 21, "gt"),
]


def sb_up(r, factor):
    """把 r（half 长度信度）上推到 factor 倍长度：k*r/(1+(k-1)*r)。"""
    if r is None or not np.isfinite(r):
        return None
    k = float(factor)
    den = 1.0 + (k - 1.0) * r
    if den <= 0:
        return None
    return float(k * r / den)


def kappa_of(E):
    """sd_between / sd_within（与 §16.4 同定义）。"""
    sb = float(E.mean(axis=1).std(ddof=1))
    sw = float(E.std(axis=1, ddof=1).mean())
    return (sb / sw if sw > 0 else float("nan")), sb, sw


def load_model(name):
    cfg = load_config(BASE / "config.json")
    _, mc = resolve_model_cfg(cfg, name)
    model_path = BASE / mc["local_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"{name}: {model_path}")
    if model_needs_fp8(mc, model_path):
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
    model, tok = load_model_for_generation(BASE, mc, model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model_needs_fp8(mc, model_path):
        # 注意: dequantize_fp8_model 是**就地**反量化, 返回的是被替换的模块数(int)。
        # 早先写成 model = dequantize_fp8_model(model) 会把 model 覆盖成 int,
        # 下一个 model.eval() 就报 "'int' object has no attribute 'eval'"。
        # ministral-3-3b 是唯一带 quantization_config 的组合, 因此只有它炸。
        n_deq = dequantize_fp8_model(model)
        print(f"  [fp8] 就地反量化模块数 = {n_deq}", flush=True)
    model.eval()
    if not hasattr(model, "eval"):
        raise RuntimeError(f"{name}: 反量化后 model 类型异常 ({type(model)})")
    return model, tok


def run_spec(model, tok, model_name, layer, task, args, device):
    out_dir = OUT / model_name / f"layer{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{task}_E_f32.npz"
    if npz.exists() and not args.force:
        print(f"[跳过] 已有 {npz}", flush=True)
        d = np.load(npz)
        E = d["E"].astype(float)
        cand = [int(x) for x in d["cand"].tolist()]
        acc = None
    else:
        layers_mod = find_decoder_layers(model)
        lmap = calibrate_layer_map(model, tok, layers_mod, device,
                                   "When Mary and John went to the store, John gave a drink to")
        block_idx = lmap[str(layer)]["block_idx"]
        layer_mod = layers_mod[block_idx]
        loaded = load_per_layer_sae(BASE, model_name, layer, device)
        if loaded is None:
            print(f"[缺 SAE] {model_name} L{layer}", flush=True)
            return None
        sae, mean_np, std_np = loaded
        mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
        std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

        built = TASK_BUILDERS[task](tok, args.n_prompts, seed=42)
        items = built[0] if isinstance(built, tuple) else built
        if not items:
            print(f"[任务不可构造] {model_name} {task}（tokenizer 无法切分）", flush=True)
            return None
        cand = [int(b["feature_idx"]) for b in loaded_features(BASE, model_name, layer, task)]
        if not cand:
            # 新任务还没有 36 的候选文件。**不能**回退到「SAE 索引 0..N」——
            # 那会更换特征集合，使跨任务的差异同时混入「特征选择」这一因素。
            # 正确做法：借用同模型同层既有任务（优先 ioi）的同一批候选，
            # 这样 E 矩阵的行是同一组特征，跨任务只差在 prompt，可配对比较。
            for alt in ("ioi", "gt", "natural"):
                cand = [int(b["feature_idx"])
                        for b in loaded_features(BASE, model_name, layer, alt)]
                if cand:
                    print(f"[借用 {alt} 的候选特征 {len(cand)}] "
                          f"{model_name} L{layer} {task}", flush=True)
                    break
        if not cand:
            n_total = int(np.asarray(mean_np).shape[0])
            cand = list(range(n_total))
            print(f"[无任何候选可借，回退全集 {n_total}] "
                  f"{model_name} L{layer} {task}", flush=True)
        cand = cand[:args.n_features]

        print(f"[{model_name} L{layer} {task}] prompts={len(items)} features={len(cand)}",
              flush=True)
        b0_vals, b0 = batch_behavior_f32(model, tok, items, layer_mod, None, device,
                                         args.batch_size)
        b0_arr = np.asarray(b0_vals, dtype=float)
        acc = float(np.mean(b0_arr > 0))
        print(f"  b0(f32)={b0:.5f} acc={acc:.3f}", flush=True)

        E = np.zeros((len(cand), len(items)), dtype=float)
        for i, f in enumerate(cand):
            hk = make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
            vals, _ = batch_behavior_f32(model, tok, items, layer_mod, hk, device,
                                         args.batch_size)
            E[i] = np.asarray(vals, dtype=float) - b0_arr
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(cand)}", flush=True)
        np.savez_compressed(npz, E=E.astype(np.float32),
                            cand=np.asarray(cand, dtype=np.int32),
                            b0=float(b0), b0_acc=float(acc))

    k, sb, sw = kappa_of(E)
    vc = _m53.variance_components(E)
    n = E.shape[1]
    half = n // 2
    rc = _m53.rel_curve(E, [half] if 2 * half <= n else [], n_draws=20)
    rel_half = rc.get(half, {}).get("rel")
    rel_full = sb_up(rel_half, 2.0)  # 100 -> 200 prompts，与主表 200/200 同尺度
    return {"model": model_name, "layer": layer, "task": task,
            "n_prompts": int(n), "n_features": int(E.shape[0]),
            "kappa": k, "sd_between": sb, "sd_within": sw,
            "icc": vc["icc"], "median_abs_E": float(np.median(np.abs(E))),
            "mean_abs_effect": vc["mean_abs_effect"],
            "rel_half": rel_half, "rel_at200_sb": rel_full,
            "b0_acc": acc, "npz": str(npz.relative_to(BASE)),
            "timestamp": datetime.now().isoformat()}


def loaded_features(base, model, layer, task):
    """候选特征沿用 36 的 features 列表（与 E10 口径一致：M2 top-200）。"""
    p = base / "output" / "phaseStep2" / "causal_validity" / model / f"layer{layer}" / f"{task}.json"
    if not p.exists():
        return []
    d = json.load(open(p, encoding="utf-8"))
    feats = sorted(d.get("features", []), key=lambda r: -abs(r.get("M2_mean_act_task", 0.0)))
    return feats


def render(recs):
    recs = sorted(recs, key=lambda r: -(r["kappa"] if np.isfinite(r["kappa"]) else -1))
    L = ["# E3：可测性比值 κ 扫描（f32 真值）", "",
         f"> 生成 {datetime.now():%Y-%m-%d %H:%M}｜{len(recs)} 个组合｜"
         f"prompts={recs[0]['n_prompts'] if recs else '-'}", "",
         "| model | layer | task | κ | sd_between | sd_within | ICC | median\\|E\\| | rel@half | rel@200(SB) | b0 acc | 判定 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        k = r["kappa"]
        verdict = ("可用" if k >= 0.30 else ("不可测" if k < 0.10 else "**临界**"))
        L.append(
            f"| {r['model']} | {r['layer']} | {r['task']} | **{k:.3f}** | {r['sd_between']:.4f} | "
            f"{r['sd_within']:.4f} | {r['icc']:.4f} | {r['median_abs_E']:.3f} | "
            f"{r['rel_half']:.3f} | {r['rel_at200_sb'] if r['rel_at200_sb'] is None else round(r['rel_at200_sb'], 3)} | "
            f"{'-' if r['b0_acc'] is None else round(r['b0_acc'], 3)} | {verdict} |")
    L += ["", "κ = sd_between / sd_within（与 n 无关）。",
          "rel@half = 100/100 分割的 Spearman（20 次重采样均值）——**跨模型比较一律用这一列**，",
          "因为 E10 的 6 个点也能用同一 half=100 重算（见 64_e3_figure.py），口径才一致。",
          "rel@200(SB) = 用 Spearman-Brown 从 rel@half 外推到 200-prompt 长度，仅作参考（平行测量假设）。"]
    md = OUT / "kappa_scan.md"
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="*", default=None,
                    help="覆盖默认 spec，格式 model:layer:task")
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_features", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    specs = DEFAULT_SPECS
    if args.spec:
        specs = []
        for s in args.spec:
            m, l, t = s.split(":")
            specs.append((m, int(l), t))

    OUT.mkdir(parents=True, exist_ok=True)
    json_p = OUT / "kappa_scan.json"
    recs = []
    if json_p.exists():
        recs = json.load(open(json_p, encoding="utf-8")).get("runs", [])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    by_model = {}
    for m, l, t in specs:
        by_model.setdefault(m, []).append((l, t))

    for m, jobs in by_model.items():
        print(f"\n===== {m}（{len(jobs)} 个 spec）=====", flush=True)
        try:
            model, tok = load_model(m)
        except Exception as e:
            print(f"[加载失败] {m}: {e}", flush=True)
            continue
        for layer, task in jobs:
            try:
                r = run_spec(model, tok, m, layer, task, args, device)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[失败] {m} L{layer} {task}: {e}", flush=True)
                continue
            if r:
                recs = [x for x in recs if not (x["model"] == r["model"] and
                                                x["layer"] == r["layer"] and
                                                x["task"] == r["task"])]
                recs.append(r)
                json_p.write_text(json.dumps({"runs": recs}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
                print(f"  κ={r['kappa']:.3f} ICC={r['icc']:.4f} rel_half={r['rel_half']:.3f} "
                      f"rel@200={r['rel_at200_sb']}", flush=True)
        del model
        torch.cuda.empty_cache()

    render(recs)
    print(f"\n[写出] {json_p}")


if __name__ == "__main__":
    main()
