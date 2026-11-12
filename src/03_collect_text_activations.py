import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from model_utils import (
    load_config, resolve_model_cfg, apply_path_template,
    get_target_layers, auto_dims_from_loaded_model, auto_dims_from_local_config,
    bootstrap_runtime_env, ensure_fp8_deps, model_needs_fp8,
)


class LogSaver:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class ResumeManager:
    def __init__(self, output_dir, script_name):
        self.output_dir = Path(output_dir)
        self.script_name = script_name
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
        state = self.load_state()
        return state.get("status") == "completed"

    def mark_completed(self, result_info=None):
        state = self.load_state()
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        if result_info:
            state["result"] = result_info
        self.save_state(state)

    def mark_step_done(self, step_name, step_data=None):
        state = self.load_state()
        if "steps_done" not in state:
            state["steps_done"] = {}
        state["steps_done"][step_name] = {"done": True, "timestamp": datetime.now().isoformat()}
        if step_data:
            state["steps_done"][step_name].update(step_data)
        self.save_state(state)

    def is_step_done(self, step_name):
        state = self.load_state()
        return state.get("steps_done", {}).get(step_name, {}).get("done", False)

    def reset(self):
        if self.state_file.exists():
            self.state_file.unlink()


def load_texts_from_dir(data_dir, max_samples=None):
    texts = []
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.endswith(".json"):
                continue
            with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                try:
                    item = json.load(fh)
                    text = item.get("text", item.get("instruction", item.get("content", "")))
                    if isinstance(text, list):
                        text = " ".join(str(t) for t in text)
                    if text and len(str(text).strip()) > 10:
                        texts.append(str(text).strip())
                except:
                    continue
            if max_samples and len(texts) >= max_samples:
                return texts
    return texts


def extract_text_activations(model, tokenizer, texts, target_layers, device, max_length=256, last_n_tokens=3):
    model.eval()
    all_activations = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="[文本激活]"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            
            try:
                outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            except Exception as e:
                print(f"  [跳过] 推理错误: {e}")
                continue
            
            hidden_states = outputs.hidden_states
            
            for layer_idx in target_layers:
                if layer_idx < len(hidden_states):
                    layer_act = hidden_states[layer_idx].float()
                    last_tokens = layer_act[0, -last_n_tokens:, :].cpu().numpy()
                    
                    all_activations.append({
                        "text": text[:200],
                        "layer": layer_idx,
                        "activations": last_tokens.tolist(),
                        "source": "text",
                    })
    
    return all_activations


def main():
    parser = argparse.ArgumentParser(description="阶段1: 数据准备 - 收集文本激活")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model", type=str, default=None, help="模型名(见 config.models), 默认用 default_model")
    parser.add_argument("--max_texts", type=int, default=None)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="强制重新运行，忽略断点续运行状态")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    base_dir = Path(args.config).parent
    bootstrap_runtime_env(base_dir)
    model_name, model_cfg = resolve_model_cfg(config, args.model)
    phase1_cfg = config["phase1_data"]["activation_collection"]
    data_dir = base_dir / config["paths"]["data_dir"]
    output_dir = base_dir / apply_path_template(config["paths"]["text_activation_dir"], model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = ResumeManager(output_dir, "03_text_act")
    if args.force:
        resume.reset()
        print("[强制] 重置断点续运行状态")

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(log_dir / f"03_text_act_{timestamp}.log"))
    sys.stdout = log_saver

    if resume.is_completed():
        existing_npz = sorted(output_dir.glob("text_activations_*.npz"))
        existing_json = sorted(output_dir.glob("text_activations_*.json"))
        if existing_npz and existing_json:
            print("[跳过] 文本激活收集已完成")
            print(f"  NPZ: {existing_npz[-1]}")
            print("  如需重新运行，请使用 --force 参数")
            log_saver.close()
            sys.stdout = log_saver.terminal
            return
    
    model_path = base_dir / model_cfg["local_path"]
    if not model_path.exists():
        for p in (base_dir / "models").glob("*"):
            if p.is_dir() and (p / "config.json").exists():
                model_path = p
                break

    # 原生 fp8 权重的模型(如 Ministral 3)需要 finegrained-fp8 内核, 自动补齐依赖
    if model_needs_fp8(model_cfg, model_path):
        # 离线绕过: 不联网校验内核发布者信任状态, 只用本地已缓存快照
        try:
            from huggingface_hub import constants as _hf_const
            _hf_const.HF_HUB_OFFLINE = True
            print("  [fp8] 已置 HF_HUB_OFFLINE=True 以绕过 kernel 信任联网检查")
        except Exception as _e:
            print(f"  [fp8] 设置 HF_HUB_OFFLINE 失败: {_e}")
        try:
            os.environ.setdefault("TRITON_CACHE_DIR",
                                  os.path.join(os.path.expanduser("~"), ".cache", "triton"))
            print(f"  [fp8] TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")
        except Exception as _e:
            print(f"  [fp8] 设置 TRITON_CACHE_DIR 失败: {_e}")
        try:
            import huggingface_hub._snapshot_download as _sd
            _sd.read_tree_cache = lambda *a, **k: None
            print("  [fp8] 已跳过 snapshot 完整性检查 (离线仅用 torch-cuda 变体)")
        except Exception as _e:
            print(f"  [fp8] 跳过 snapshot 检查失败: {_e}")
        ensure_fp8_deps()

    # 解析目标层: 配置有就用，否则读本地模型维度按 1/3、2/3 生成
    cfg_hidden, cfg_layers = auto_dims_from_local_config(model_cfg, model_path)
    target_layers = get_target_layers(model_cfg, cfg_layers)

    max_texts = args.max_texts or phase1_cfg.get("text_samples", 500)
    sample_ratio = args.sample_ratio or phase1_cfg.get("sample_ratio", 0.1)
    max_length = phase1_cfg.get("max_length", 256)
    last_n_tokens = phase1_cfg.get("last_n_tokens", 3)
    
    print(f"{'='*60}")
    print(f"阶段1: 收集文本激活")
    print(f"{'='*60}")
    print(f"模型: {model_name} ({model_path})")
    print(f"目标层: {target_layers}")
    print(f"最大文本数: {max_texts}, 采样率: {sample_ratio}")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
    
    print("[加载] 模型...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=model_cfg.get("trust_remote_code", True))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except (ValueError, KeyError) as e:
        error_msg = str(e)
        if "does not recognize this architecture" in error_msg or "model type" in error_msg.lower():
            print(f"[警告] 模型架构未识别: {error_msg[:200]}")
            # 多模态条件生成模型 (如 Ministral 3) 不是纯 CausalLM, 回退到 ConditionalGeneration
            print("[尝试] 使用 ConditionalGeneration 加载...")
            try:
                model = Mistral3ForConditionalGeneration.from_pretrained(
                    str(model_path),
                    trust_remote_code=model_cfg.get("trust_remote_code", True),
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                model_cfg["_condgen"] = True
            except Exception as e3:
                print(f"[失败] ConditionalGeneration 加载失败: {e3}")
                print("[尝试] 使用 trust_remote_code=True 手动加载配置...")
                try:
                    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        str(model_path),
                        config=config,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                    )
                except Exception as e2:
                    print(f"[失败] 手动加载也失败: {e2}")
                    print(f"\n[诊断] 检查模型目录文件...")
                    model_dir = Path(model_path)
                    py_files = list(model_dir.glob("*.py"))
                    print(f"  模型目录: {model_dir}")
                    print(f"  Python文件: {[f.name for f in py_files] if py_files else '无'}")
                    if not py_files:
                        print("  [原因] 缺少自定义建模代码文件 (configuration_*.py, modeling_*.py)")
                        print("  [原因] ModelScope 下载可能不包含这些文件")
                    print("\n[解决方案] 请尝试以下步骤:")
                    print("  1. 升级 transformers: pip install --upgrade transformers")
                    print("  2. 从 HuggingFace 镜像下载（包含完整代码文件）:")
                    print("     pip install huggingface_hub")
                    print("     export HF_ENDPOINT=https://hf-mirror.com")
                    print("     huggingface-cli download Qwen/Qwen3.5-2B --local-dir models/Qwen/Qwen3___5-2B")
                    print("  3. 或从 ModelScope 重新下载:")
                    print("     python 02_download_model.py --config config.json --force")
                    raise
        else:
            raise
    
    model.eval()
    print(f"[加载] 完成，设备: {device}")
    
    print("[加载] 文本数据...")
    text_dir = data_dir / "text"
    if not text_dir.exists():
        print(f"[错误] 文本数据目录不存在: {text_dir}")
        print("[提示] 请先运行 01_download_data.py")
        return
    
    texts = load_texts_from_dir(str(text_dir), max_samples=max_texts)
    if sample_ratio < 1.0:
        import random
        random.shuffle(texts)
        texts = texts[:int(len(texts) * sample_ratio)]
    
    print(f"[数据] 文本数: {len(texts)}")
    if len(texts) == 0:
        print("[错误] 没有文本数据！")
        return
    
    all_activations = extract_text_activations(
        model, tokenizer, texts, target_layers, device,
        max_length=max_length, last_n_tokens=last_n_tokens
    )
    
    if len(all_activations) == 0:
        print("[错误] 没有收集到任何激活！")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = output_dir / f"text_activations_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_activations, f, ensure_ascii=False)
    
    npz_data = {}
    for i, act in enumerate(all_activations):
        key = f"{i}_{act['layer']}"
        npz_data[key] = np.array(act["activations"])
    
    npz_file = output_dir / f"text_activations_{timestamp}.npz"
    np.savez_compressed(npz_file, **npz_data)
    
    meta = {
        "timestamp": timestamp,
        "model": model_name,
        "model_id": model_cfg["model_id"],
        "num_samples": len(all_activations),
        "target_layers": target_layers,
        "source": "text",
        "max_length": max_length,
        "last_n_tokens": last_n_tokens,
    }
    meta_file = output_dir / f"metadata_{timestamp}.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"文本激活收集完成!")
    print(f"样本数: {len(all_activations)}")
    print(f"JSON: {json_file}")
    print(f"NPZ: {npz_file}")
    print(f"{'='*60}")

    resume.mark_completed({"npz": str(npz_file), "json": str(json_file), "num_samples": len(all_activations)})

    log_saver.close()
    sys.stdout = log_saver.terminal


if __name__ == "__main__":
    main()
