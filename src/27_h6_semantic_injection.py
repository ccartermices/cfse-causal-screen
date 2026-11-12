# -*- coding: utf-8 -*-
"""Phase H6: 语义级注入评估 (Generation-time SAE-Guided Injection)

把 H4 的「向量空间」注入公式推进到「生成语义」层: 在推理时于中间层残差流上
注入 blended = h + scale*(decode(encode(norm(h))) - h) (与 03 激活收集口径一致:
hook 在 decoder 层 block 输出处, hidden_states[layer] 残差流), 观察 scale∈{0.0,0.5,1.0}
下生成文本的语义漂移。

度量 (greedy, max_new_tokens=64):
  - 输出长度 / 唯一 bigram 数        —— 注入是否改变生成行为
  - 1/2/3-gram Jaccard (scale 两两间) —— 语义/词汇漂移
与 H4 的向量结论对接: 扰动线性、scale=0.5 时余弦保持 0.87~0.94。

输出: output/phaseH6/semantic_injection/{model}/h6_semantic_injection.json
      output/phaseH6/semantic_injection/h6_semantic_injection.json (汇总)
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from importlib import import_module
from model_utils import (
    load_config, resolve_model_cfg, apply_path_template, load_base_model,
    bootstrap_runtime_env, ensure_fp8_deps, model_needs_fp8,
)

_mod05 = import_module("05_train_text_sae")
BatchTopKSAE = _mod05.BatchTopKSAE
LogSaver = _mod05.LogSaver


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

    def reset(self):
        if self.state_file.exists():
            self.state_file.unlink()


def load_sae(sae_path, device):
    ckpt = torch.load(sae_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = BatchTopKSAE(cfg["input_dim"], cfg["hidden_dim"], cfg["k"]).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae, ckpt["mean"], ckpt["std"], cfg


def find_decoder_layers(model):
    """在任意模型结构中定位 decoder 的 layers ModuleList (含 Mistral3 条件生成包装).

    注: nn.Module 的子模块存于 _modules OrderedDict, vars(mod) 扫描不到,
    因此必须直接遍历 named_modules() 中的 ModuleList 实例。
    选层数最多的 ModuleList 作为 decoder (兼容 Qwen3_5 混合架构 self_attn/linear_attn)。
    """
    candidates = []
    for _, child in model.named_modules():
        if isinstance(child, torch.nn.ModuleList) and len(child) > 0:
            candidates.append((len(child), child))
    if not candidates:
        return None
    # 优先选层数最多且至少含若干层 (>8) 的 ModuleList, 排除掉小容量 (如 lm_head 列表等)
    candidates.sort(key=lambda x: -x[0])
    for _, ml in candidates:
        if len(ml) >= 8:
            return ml
    return candidates[0][1]


def load_model_for_generation(base_dir, model_cfg, model_path):
    """加载生成用基座: 统一走 model_utils.load_base_model (与 14/15 同口径).

    - phi 系列: load_base_model 内部改用 transformers 原生 Phi3ForCausalLM
      (trust_remote_code=False), 绕开模型目录内 4.x 旧自定义 modeling
      —— 该自定义类引用了 transformers 5.x 已移除的 Cache.seen_tokens,
      这才是此前 phi 生成报错的真正根因, 无需 use_cache=False 之类的慢速规避。
    - ministral-3-3b: AutoModelForCausalLM 无法识别 Mistral3Config, 回退
      Mistral3ForConditionalGeneration。
    - 其余模型: AutoModelForCausalLM, device_map="auto" 落到独显。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.generation.utils import GenerationMixin as _GM
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

    if not torch.cuda.is_available():
        print("  [警告] CUDA 不可用, 将回退 CPU!")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=model_cfg.get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_needs_fp8(model_cfg, model_path):
        ensure_fp8_deps()

    model = None
    try:
        model = load_base_model(str(model_path), model_cfg, torch.bfloat16, "auto",
                               trust_remote_code=model_cfg.get("trust_remote_code", True))
    except Exception as e:
        print(f"  [警告] load_base_model 失败: {str(e)[:140]}")
        try:
            model = Mistral3ForConditionalGeneration.from_pretrained(
                str(model_path), trust_remote_code=model_cfg.get("trust_remote_code", True),
                torch_dtype=torch.bfloat16, device_map="auto")
        except Exception as e3:
            print(f"  [警告] ConditionalGeneration 失败: {str(e3)[:140]}")
            try:
                config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), config=config, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, device_map="auto")
            except Exception as e2:
                raise RuntimeError(f"模型加载全部失败: {e2}")

    # 兜底: 确认 generation 接口可用 (原生类通常已继承 mixin, 此处为防御性补丁)
    if "GenerationMixin" not in str(model.__class__.__mro__):
        model.__class__ = type(model.__class__.__name__, (model.__class__, _GM), {})

    model.eval()
    dev = next(model.parameters()).device
    if torch.cuda.is_available() and dev.type != "cuda":
        print(f"  [警告] 模型落在 {dev}, 强制迁移至 cuda")
        model = model.to("cuda")
        dev = next(model.parameters()).device
    print(f"  [设备] 模型已加载于 {dev} (cuda_available={torch.cuda.is_available()})")
    return model, tokenizer


def build_inject_hook(sae, mean, std, device, scale):
    """返回 forward hook: 在第 L 层 block 输出处把残差流替换为 blended."""
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    def hook(module, args, output):
        if scale <= 0:
            return output
        if isinstance(output, tuple) or isinstance(output, list):
            hs = output[0]
        else:
            hs = output
        h_flat = hs.float().reshape(-1, hs.shape[-1])  # (seq, hidden)
        normed = (h_flat - mean_t) / std_t
        with torch.no_grad():
            recon_n = sae.decode(sae.encode(normed)[0])  # (seq, hidden)
        recon_raw = recon_n * std_t + mean_t
        blended = h_flat + scale * (recon_raw - h_flat)
        blended = blended.to(hs.dtype).reshape(hs.shape)
        if isinstance(output, tuple) or isinstance(output, list):
            return (blended,) + output[1:]
        return blended

    return hook


def tokenize_ngrams(ids, n, special_ids):
    ids = [i for i in ids if i not in special_ids]
    if len(ids) < n:
        return set()
    return set(zip(*[ids[k:] for k in range(n)]))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_model(args, config, base_dir, m, common_prompts):
    _, model_cfg = resolve_model_cfg(config, m)
    out_dir = base_dir / "output" / "phaseH6" / "semantic_injection" / m
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = ResumeManager(out_dir, "27_h6")
    if resume.is_completed() and (out_dir / "h6_semantic_injection.json").exists():
        print(f"[跳过] {m} 已完成")
        return

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
        print(f"[失败] {m} 未找到 decoder layers ModuleList")
        resume.save_state({"status": "failed", "error": "no decoder layers", "at": datetime.now().isoformat()})
        del model
        return

    tl = model_cfg["target_layers"]
    num_layers = model_cfg["num_hidden_layers"]
    if len(layers_mod) == num_layers:
        mid_idx = tl[1]
    else:
        mid_idx = int(round(tl[1] / num_layers * (len(layers_mod) - 1)))
    mid_layer = layers_mod[mid_idx]
    print(f"[{m}] decoder layers={len(layers_mod)}, 注入层 idx={mid_idx}")

    sae_path = base_dir / "output" / "phase2" / "text_sae" / m / "best_sae.pt"
    if not sae_path.exists():
        sae_path = base_dir / "output" / "phase2" / "text_sae" / "best_sae.pt"
    device = next(model.parameters()).device
    sae, mean, std, sae_cfg = load_sae(str(sae_path), device)

    prompts = common_prompts[:args.num_prompts]
    scales = [0.0, 0.5, 1.0]
    special_ids = set(tokenizer.all_special_ids)

    results = {str(s): [] for s in scales}
    samples = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        input_len = inputs["input_ids"].shape[-1]
        for scale in scales:
            handle = None
            if scale > 0:
                handle = mid_layer.register_forward_hook(
                    build_inject_hook(sae, mean, std, device, scale))
            try:
                with torch.no_grad():
                    gen = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)
            except Exception as e:
                print(f"  [失败] {m} scale={scale} 生成错误: {str(e)[:150]}")
                if handle:
                    handle.remove()
                continue
            finally:
                if handle:
                    handle.remove()
            gen_ids = gen[0][input_len:].tolist()
            out_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results[str(scale)].append({"prompt_idx": prompts.index(text), "ids": gen_ids,
                                        "len": len([i for i in gen_ids if i not in special_ids]),
                                        "text": out_text[:400]})
            samples.append({"prompt": text[:120], "scale": scale, "output": out_text[:400]})

    # 汇总指标
    def mean_len(lst):
        return float(np.mean([d["len"] for d in lst])) if lst else None

    def mean_uniq_bigrams(lst):
        vals = []
        for d in lst:
            ids = [i for i in d["ids"] if i not in special_ids]
            bg = set(zip(ids, ids[1:])) if len(ids) >= 2 else set()
            vals.append(len(bg))
        return float(np.mean(vals)) if vals else None

    drift = {}
    for sa, sb in [("0.0", "0.5"), ("0.0", "1.0"), ("0.5", "1.0")]:
        drift[f"{sa}_vs_{sb}"] = {}
        for n in (1, 2, 3):
            jacs = []
            for a, b in zip(results[sa], results[sb]):
                ja = tokenize_ngrams(a["ids"], n, special_ids)
                jb = tokenize_ngrams(b["ids"], n, special_ids)
                jacs.append(jaccard(ja, jb))
            drift[f"{sa}_vs_{sb}"][f"ngram{n}_jaccard"] = float(np.mean(jacs)) if jacs else None

    result = {
        "model": m, "layer_idx": mid_idx, "decoder_layers": len(layers_mod),
        "sae_k": sae_cfg["k"], "sae_hidden_dim": sae_cfg["hidden_dim"],
        "num_prompts": len(prompts), "scales": scales,
        "mean_output_len": {str(s): mean_len(results[str(s)]) for s in scales},
        "mean_unique_bigrams": {str(s): mean_uniq_bigrams(results[str(s)]) for s in scales},
        "drift": drift,
        "samples": samples,
    }
    with open(out_dir / "h6_semantic_injection.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    resume.mark_completed({"layer_idx": mid_idx, "prompts": len(prompts)})
    print(f"[{m}] 完成: len={result['mean_output_len']} drift(1-gram)="
          f"{drift['0.0_vs_0.5']['ngram1_jaccard']:.3f}->"
          f"{drift['0.0_vs_1.0']['ngram1_jaccard']:.3f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Phase H6: 语义级注入评估")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--models", type=str,
                        default="qwen3.5-2b,qwen3.5-0.8b,tinyllama-1.1b,minicpm5-1b,phi-3.5-mini,ministral-3-3b")
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = Path(args.config).parent
    bootstrap_runtime_env(base_dir)
    models = [x.strip() for x in args.models.split(",") if x.strip()]

    root_out = base_dir / "output" / "phaseH6"
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "logs").mkdir(parents=True, exist_ok=True)
    log_saver = LogSaver(str(root_out / "logs" / f"27_h6_{datetime.now():%Y%m%d_%H%M%S}.log"))
    sys.stdout = log_saver

    # 共享文本 (middle 层, 与 H3/H5 同口径)
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
    print(f"[共享文本] {len(common)} 条, 取前 {args.num_prompts} 作为生成 prompt")

    for m in models:
        if args.force:
            out_dir = base_dir / "output" / "phaseH6" / "semantic_injection" / m
            r = ResumeManager(out_dir, "27_h6")
            r.reset()
        run_model(args, config, base_dir, m, common)

    # 汇总
    summary = {"num_prompts": args.num_prompts, "max_new_tokens": args.max_new_tokens, "models": {}}
    for m in models:
        f = base_dir / "output" / "phaseH6" / "semantic_injection" / m / "h6_semantic_injection.json"
        if f.exists():
            d = json.load(open(f, encoding="utf-8"))
            summary["models"][m] = {
                "layer_idx": d["layer_idx"], "sae_k": d["sae_k"], "sae_hidden_dim": d["sae_hidden_dim"],
                "num_prompts": d["num_prompts"],
                "mean_output_len": d["mean_output_len"],
                "mean_unique_bigrams": d["mean_unique_bigrams"],
                "drift": d["drift"],
            }
    with open(root_out / "h6_semantic_injection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总: {root_out / 'h6_semantic_injection.json'}")

    log_saver.close()
    sys.stdout = log_saver.terminal


if __name__ == "__main__":
    main()
