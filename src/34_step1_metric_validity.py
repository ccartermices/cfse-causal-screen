# -*- coding: utf-8 -*-
"""Step 1: SAE 注入效应度量有效性 —— 分布级损伤 vs 表面词汇指标

关键勘误 (动机):
  H6/H8/H9 中名为 `drift_ngram*` 的字段, 实际存的是 **Jaccard 相似度**
  (|A∩B|/|A∪B|, 见 27_h6_semantic_injection.py 的 jaccard())。相似度越高
  = 输出越接近 baseline = 改变越小。因此:
    - H9 corr(drift_ngram1, ΔlogPPL) = -0.486 应读作 corr(相似度, 损伤) = -0.486
      ⇒ corr(1-相似度, 损伤) = +0.486: **词重叠距离其实与损伤正相关**(方向正确)。
    - H8/H8b "峰值层 / 低 scale 漂移最大" 同理被符号反转。
  本脚本用**显式命名**重做, 并以分布级 KL(p_base || p_inj) 作为损伤基准,
  输出度量 x 度量 相关矩阵, 一次性厘清"哪些指标有效、方向如何"。

度量 (每模型每层每 scale):
  生成侧 (greedy vs baseline 生成续写):
    jaccard_sim1/2   = Jaccard 相似度 (复现 H9 的 drift_ngram1/2)
    jaccard_dist1/2  = 1 - jaccard_sim
    distinct1/2      = 生成续写的 distinct-n 比例 (多样性)
    rep_rate         = 生成续写中相邻重复 token 比例 (重复坍塌)
    out_len          = 平均生成续写长度
  分布侧 (teacher-forced, held-out 语料):
    kl_base_inj      = 平均 KL( p_base || p_inj )   <-- 分布级损伤 (基准)
    ppl_inj, ppl_ratio, delta_log_ppl

判据:
  jaccard_dist1 与 kl_base_inj 正相关且强度可观 ⇒ 词重叠指标是有效损伤代理;
  弱相关 ⇒ 只能反映表面词汇变化, 不能代表真实能力损失。

资源: CPU 严格节流 (OMP/MKL=1, torch threads=1, num_workers=0), 单进程串行, 落独显。

输出: output/phaseStep1/metric_validity/{model}/step1_metric_validity.json
      output/phaseStep1/metric_validity/step1_metric_validity.json (汇总+相关矩阵)
      output/phaseStep1/step1_summary.md (人读摘要)
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

from model_utils import (
    load_config, resolve_model_cfg, apply_path_template, bootstrap_runtime_env,
)

_mod05 = import_module("05_train_text_sae")
LogSaver = _mod05.LogSaver

_mod27 = import_module("27_h6_semantic_injection")
load_model_for_generation = _mod27.load_model_for_generation
find_decoder_layers = _mod27.find_decoder_layers
build_inject_hook = _mod27.build_inject_hook
load_sae = _mod27.load_sae
tokenize_ngrams = _mod27.tokenize_ngrams
jaccard = _mod27.jaccard


class ResumeManager:
    def __init__(self, output_dir, script_name):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / f"resume_{script_name}.json"

    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_state(self, state):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def is_completed(self):
        return self.load_state().get("status") == "completed"

    def mark_completed(self, result_info=None):
        state = self.load_state()
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        if result_info:
            state["result"] = result_info
        self.save_state(state)


def _rank(a):
    order = sorted(range(len(a)), key=lambda i: a[i])
    r = [0.0] * len(a)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(_rank(list(x)), _rank(list(y)))[0, 1])


def pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(list(x), list(y))[0, 1])


def gen_ids(model, inputs, layer_mod, hook_fn, max_new_tokens):
    handle = None
    if hook_fn is not None and layer_mod is not None:
        handle = layer_mod.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, num_beams=1)
    finally:
        if handle:
            handle.remove()
    return out


def gen_surface_metrics(gen_list, special_ids):
    """生成侧表面指标 (作用于续写 token 序列列表)."""
    sim1, sim2, d1, d2, reps, lens = [], [], [], [], [], []
    for ids in gen_list:
        clean = [i for i in ids if i not in special_ids]
        lens.append(len(clean))
        if len(clean) >= 1:
            d1.append(len(set(clean)) / len(clean))
            reps.append(sum(1 for a, b in zip(clean, clean[1:]) if a == b) / max(len(clean) - 1, 1))
        else:
            d1.append(0.0)
            reps.append(0.0)
        if len(clean) >= 2:
            bg = list(zip(clean, clean[1:]))
            d2.append(len(set(bg)) / len(bg))
        else:
            d2.append(0.0)
    return {
        "mean_output_len": float(np.mean(lens)) if lens else 0.0,
        "distinct1": float(np.mean(d1)) if d1 else 0.0,
        "distinct2": float(np.mean(d2)) if d2 else 0.0,
        "rep_rate": float(np.mean(reps)) if reps else 0.0,
    }


def cache_base_logits(model, tokenizer, texts, device, max_length):
    """teacher-forced 缓存 baseline logits (CPU, fp16), 供 KL/PPL 复用."""
    cached = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"]
        am = enc["attention_mask"]
        with torch.no_grad():
            lg = model(input_ids=ids.to(device), attention_mask=am.to(device)).logits[0]
        cached.append((ids, am, lg.half().cpu()))
    return cached


def dist_metrics(model, cached, layer_mod, hook_fn, device):
    """分布级度量: KL(p_base||p_inj), PPL_inj (teacher-forced, 与 base 复用同一批文本)."""
    handle = None
    if hook_fn is not None and layer_mod is not None:
        handle = layer_mod.register_forward_hook(hook_fn)
    kl_sum = 0.0
    nll_b = 0.0
    nll_i = 0.0
    tok = 0
    try:
        for ids, am, lg_b_half in cached:
            ids_d = ids.to(device)
            am_d = am.to(device)
            with torch.no_grad():
                lg_i = model(input_ids=ids_d, attention_mask=am_d).logits[0].float()
            lg_b = lg_b_half.float().to(device)
            logp_b = torch.log_softmax(lg_b, dim=-1)
            logp_i = torch.log_softmax(lg_i, dim=-1)
            # HF logits[t] 预测 token t+1; 用 [:-1] 对齐 labels
            pb = logp_b[:-1]
            pi = logp_i[:-1]
            tgt = ids_d[0, 1:]
            kl = (pb.exp() * (pb - pi)).sum(-1)
            kl_sum += float(kl.sum().item())
            nll_b += float((-pb.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).sum().item())
            nll_i += float((-pi.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).sum().item())
            tok += int(tgt.numel())
    finally:
        if handle:
            handle.remove()
    tok = max(tok, 1)
    ppl_b = float(np.exp(nll_b / tok))
    ppl_i = float(np.exp(nll_i / tok))
    return {
        "kl_base_inj": float(kl_sum / tok),
        "ppl_base": ppl_b,
        "ppl_inj": ppl_i,
        "ppl_ratio": float(ppl_i / max(ppl_b, 1e-12)),
        "delta_log_ppl": float(np.log(max(ppl_i, 1e-12)) - np.log(max(ppl_b, 1e-12))),
        "num_tokens": tok,
    }


def run_model(args, config, base_dir, m, common_texts):
    _, model_cfg = resolve_model_cfg(config, m)
    out_dir = base_dir / "output" / "phaseStep1" / "metric_validity" / m
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = ResumeManager(out_dir, "34_step1")
    if resume.is_completed() and (out_dir / "step1_metric_validity.json").exists() and not args.force:
        print(f"[跳过] {m} 已完成")
        return

    h8f = base_dir / "output" / "phaseH8" / "layer_sweep" / m / "h8_layer_sweep.json"
    h8 = json.load(open(h8f, encoding="utf-8")) if h8f.exists() else None

    model_path = base_dir / model_cfg["local_path"]
    if not model_path.exists():
        for p in (base_dir / "models").glob("*"):
            if p.is_dir() and (p / "config.json").exists():
                model_path = p
                break

    print(f"\n[{m}] 加载基座模型...")
    try:
        model, tokenizer = load_model_for_generation(base_dir, model_cfg, model_path)
    except Exception as e:
        print(f"[失败] {m} 模型加载失败: {e}")
        resume.save_state({"status": "failed", "error": str(e), "at": datetime.now().isoformat()})
        return

    layers_mod = find_decoder_layers(model)
    if layers_mod is None:
        resume.save_state({"status": "failed", "error": "no decoder layers",
                           "at": datetime.now().isoformat()})
        del model
        return

    n = len(layers_mod)
    tl = model_cfg["target_layers"]
    num_layers = model_cfg["num_hidden_layers"]
    canonical_mid = tl[1] if n == num_layers else int(round(tl[1] / num_layers * (n - 1)))
    early = max(1, canonical_mid // 2)
    late = int(round(0.9 * (n - 1)))
    layer_ids = sorted(set([early, canonical_mid, late]))
    device = next(model.parameters()).device

    sae_path = base_dir / "output" / "phase2" / "text_sae" / m / "best_sae.pt"
    if not sae_path.exists():
        sae_path = base_dir / "output" / "phase2" / "text_sae" / "best_sae.pt"
    sae, mean, std, sae_cfg = load_sae(str(sae_path), device)

    prompts = common_texts[:args.num_prompts]
    held_texts = common_texts[args.num_prompts: args.num_prompts + args.num_ppl_texts]
    if not held_texts:
        held_texts = common_texts[:args.num_ppl_texts]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    special_ids = set(tokenizer.all_special_ids)
    print(f"[{m}] 层={layer_ids} (early={early}, mid={canonical_mid}, late={late}), "
          f"prompts={len(prompts)}, held_texts={len(held_texts)}, device={device}")

    tokenized = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=args.max_length).to(device)
        tokenized.append((inputs, inputs["input_ids"].shape[-1]))

    # baseline 生成
    base_ids = []
    for inputs, ilen in tokenized:
        out = gen_ids(model, inputs, None, None, args.max_new_tokens)
        base_ids.append(out[0][ilen:].tolist())

    # baseline 分布 (缓存)
    cached = cache_base_logits(model, tokenizer, held_texts, device, args.max_length)
    dist_b = dist_metrics(model, cached, None, None, device)
    base_ppl = dist_b["ppl_base"]
    print(f"[{m}] baseline ppl={base_ppl:.3f}, tokens={dist_b['num_tokens']}")

    rows = []
    for li in layer_ids:
        layer_mod = layers_mod[li]
        for scale in scales:
            hook_fn = build_inject_hook(sae, mean, std, device, scale)
            gen = []
            for inputs, ilen in tokenized:
                out = gen_ids(model, inputs, layer_mod, hook_fn, args.max_new_tokens)
                gen.append(out[0][ilen:].tolist())
            sim1 = [jaccard(tokenize_ngrams(a, 1, special_ids),
                            tokenize_ngrams(b, 1, special_ids)) for a, b in zip(base_ids, gen)]
            sim2 = [jaccard(tokenize_ngrams(a, 2, special_ids),
                            tokenize_ngrams(b, 2, special_ids)) for a, b in zip(base_ids, gen)]
            surf = gen_surface_metrics(gen, special_ids)
            dist = dist_metrics(model, cached, layer_mod, hook_fn, device)
            row = {
                "layer_idx": li, "scale": scale,
                "jaccard_sim1": float(np.mean(sim1)), "jaccard_sim2": float(np.mean(sim2)),
                "jaccard_dist1": float(1.0 - np.mean(sim1)),
                "jaccard_dist2": float(1.0 - np.mean(sim2)),
                "kl_base_inj": dist["kl_base_inj"],
                "ppl_inj": dist["ppl_inj"], "ppl_base": dist["ppl_base"],
                "ppl_ratio": dist["ppl_ratio"], "delta_log_ppl": dist["delta_log_ppl"],
            }
            row.update(surf)
            rows.append(row)
            print(f"    L{li} s={scale:.2f}: sim1={row['jaccard_sim1']:.3f} "
                  f"dist1={row['jaccard_dist1']:.3f} rep={row['rep_rate']:.3f} "
                  f"kl={row['kl_base_inj']:.4f} pplR={row['ppl_ratio']:.2f}")

    result = {
        "model": m, "decoder_layers": n, "layer_ids": layer_ids,
        "canonical_mid_layer": canonical_mid, "h8_peak_layer": (h8 or {}).get("peak_layer"),
        "sae_k": sae_cfg["k"], "num_prompts": len(prompts),
        "num_held_texts": len(held_texts), "scales": scales, "base_ppl": base_ppl,
        "rows": rows,
    }
    with open(out_dir / "step1_metric_validity.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    resume.mark_completed({"layers": layer_ids})
    print(f"[{m}] 完成: base_ppl={base_ppl:.2f}")

    del model, sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


METRIC_KEYS = ["jaccard_sim1", "jaccard_dist1", "jaccard_sim2", "jaccard_dist2",
               "distinct1", "distinct2", "rep_rate", "mean_output_len",
               "kl_base_inj", "ppl_ratio", "delta_log_ppl"]


def corr_matrix(rows, keys):
    mat = {}
    for a in keys:
        xa = [r[a] for r in rows]
        mat[a] = {}
        for b in keys:
            xb = [r[b] for r in rows]
            mat[a][b] = {"pearson": pearson(xa, xb), "spearman": spearman(xa, xb)}
    return mat


def main():
    parser = argparse.ArgumentParser(description="Step 1: 注入效应度量有效性")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--models", type=str,
                        default="qwen3.5-2b,qwen3.5-0.8b,tinyllama-1.1b,minicpm5-1b,phi-3.5-mini,ministral-3-3b")
    parser.add_argument("--scales", type=str, default="0.25,0.5,0.75,1.0")
    parser.add_argument("--num_prompts", type=int, default=8)
    parser.add_argument("--num_ppl_texts", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = Path(args.config).parent
    bootstrap_runtime_env(base_dir)
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    root_out = base_dir / "output" / "phaseStep1"
    (root_out / "logs").mkdir(parents=True, exist_ok=True)
    log_saver = LogSaver(str(root_out / "logs" / f"34_step1_{datetime.now():%Y%m%d_%H%M%S}.log"))
    sys.stdout = log_saver
    print(f"[资源] torch threads={torch.get_num_threads()}, "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}, cuda={torch.cuda.is_available()}")

    text_sets = {}
    for m in models:
        _, model_cfg = resolve_model_cfg(config, m)
        act_dir = base_dir / apply_path_template(config["paths"]["text_activation_dir"], m)
        json_files = sorted(act_dir.glob("text_activations_*.json"))
        if not json_files:
            continue
        data = json.load(open(json_files[-1], encoding="utf-8"))
        tl = model_cfg["target_layers"]
        text_sets[m] = set(d["text"] for d in data if d["layer"] == tl[1])
    common = None
    for ts in text_sets.values():
        common = set(ts) if common is None else common.intersection(ts)
    common = sorted(common)
    print(f"[共享文本] {len(common)} 条 (前 {args.num_prompts} 作 prompt, "
          f"其后 {args.num_ppl_texts} 作 held-out)")

    for m in models:
        run_model(args, config, base_dir, m, common)

    summary = {"scales": args.scales, "metric_keys": METRIC_KEYS, "models": {}}
    all_rows = []
    for m in models:
        f = base_dir / "output" / "phaseStep1" / "metric_validity" / m / "step1_metric_validity.json"
        if not f.exists():
            continue
        d = json.load(open(f, encoding="utf-8"))
        per_model = corr_matrix(d["rows"], METRIC_KEYS)
        summary["models"][m] = {
            "layer_ids": d["layer_ids"], "canonical_mid_layer": d["canonical_mid_layer"],
            "h8_peak_layer": d["h8_peak_layer"], "sae_k": d["sae_k"],
            "base_ppl": d["base_ppl"],
            "corr_vs_kl": {k: per_model[k]["kl_base_inj"] for k in METRIC_KEYS},
            "rows": d["rows"],
        }
        all_rows.extend(d["rows"])
    summary["global_corr_matrix"] = corr_matrix(all_rows, METRIC_KEYS) if all_rows else {}
    summary["global_n"] = len(all_rows)
    with open(root_out / "metric_validity" / "step1_metric_validity.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 人读摘要
    lines = ["# Step 1: 注入效应度量有效性 (分布级 KL vs 表面词汇指标)", "",
             f"- 模型: {len(summary['models'])} 个; 全局点数 n={summary['global_n']}",
             f"- 损伤基准: kl_base_inj = 平均 KL(p_base || p_inj) (teacher-forced)", "",
             "## 全局: 各指标 vs KL 损伤 (spearman)", "",
             "| metric | pearson(·, KL) | spearman(·, KL) |", "|---|---|---|"]
    gm = summary["global_corr_matrix"]
    for k in METRIC_KEYS:
        c = gm.get(k, {}).get("kl_base_inj", {}) if gm else {}
        lines.append(f"| {k} | {c.get('pearson')} | {c.get('spearman')} |")
    lines += ["", "## 逐模型: corr(指标, KL) 的 spearman", "",
              "| model | sim1 | dist1 | distinct1 | rep_rate | ppl_ratio |", "|---|---|---|---|---|---|"]
    for m, v in summary["models"].items():
        cv = v["corr_vs_kl"]
        def gs(k):
            c = cv.get(k, {})
            s = c.get("spearman") if isinstance(c, dict) else None
            return f"{s:.3f}" if isinstance(s, (int, float)) else "NA"
        lines.append(f"| {m} | {gs('jaccard_sim1')} | {gs('jaccard_dist1')} | "
                     f"{gs('distinct1')} | {gs('rep_rate')} | {gs('ppl_ratio')} |")
    lines += ["", "> 读法: 若 dist1 (词重叠距离) 与 KL 正相关、sim1 与 KL 负相关,",
              "> 则 n-gram 指标方向正确; 强度决定其作为损伤代理的有效性。",
              "> (H9 报告的 drift_ngram1 实为 sim1, 其 -0.486 等价于 dist1 的 +0.486。)"]
    with open(root_out / "step1_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[全局] n={summary['global_n']}")
    if gm:
        for k in ("jaccard_sim1", "jaccard_dist1", "distinct1", "rep_rate", "ppl_ratio"):
            c = gm.get(k, {}).get("kl_base_inj", {})
            print(f"    corr({k}, KL): pearson={c.get('pearson')}, spearman={c.get('spearman')}")
    print(f"汇总: {root_out / 'metric_validity' / 'step1_metric_validity.json'}")
    print(f"摘要: {root_out / 'step1_summary.md'}")

    log_saver.close()
    sys.stdout = log_saver.terminal


if __name__ == "__main__":
    main()
