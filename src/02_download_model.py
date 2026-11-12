import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime


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

    def get_step_data(self, step_name):
        state = self.load_state()
        return state.get("steps_done", {}).get(step_name, {})

    def reset(self):
        if self.state_file.exists():
            self.state_file.unlink()


def download_model_from_modelscope(model_id, target_dir, ignore_file_pattern=None):
    """从 ModelScope 下载模型到指定目录"""
    from modelscope import snapshot_download
    import shutil

    print(f"[下载] 模型: {model_id}")
    print(f"[目标] 目录: {target_dir}")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    # ModelScope 下载到缓存
    cache_dir = target.parent / "_ms_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    kw = {}
    if ignore_file_pattern:
        kw["ignore_file_pattern"] = ignore_file_pattern
        print(f"[过滤] 忽略文件模式: {ignore_file_pattern}")
    cached = snapshot_download(model_id, cache_dir=str(cache_dir), **kw)

    # 复制到目标(避开 .lock / __pycache__ 等)
    if Path(cached).resolve() != target.resolve():
        for item in Path(cached).iterdir():
            dst = target / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

    print(f"[完成] 模型已下载到: {target}")
    return str(target)


def download_model_from_hf(model_id, target_dir, hf_endpoint=None):
    """从 HuggingFace 镜像下载模型到指定目录
    策略: 先下到缓存再复制, 避免 local_dir 在某些镜像下失效
    """
    from huggingface_hub import snapshot_download
    import shutil

    print(f"[下载] 模型: {model_id}")
    print(f"[目标] 目录: {target_dir}")
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"[HF_ENDPOINT] {hf_endpoint}")

    # 先下到缓存
    cache_dir = target.parent / "_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        cached = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            local_dir=None,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        # 老版本 huggingface_hub 不支持 cache_dir
        cached = snapshot_download(repo_id=model_id)

    # 复制到目标目录
    cached_path = Path(cached)
    if cached_path.resolve() != target.resolve():
        for item in cached_path.iterdir():
            dst = target / item.name
            if item.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(item, dst)
            else:
                if dst.exists():
                    dst.unlink()
                shutil.copy2(item, dst)

    print(f"[完成] 模型已下载到: {target}")
    return str(target)


def verify_model(local_dir):
    """校验模型文件完整性 + 读出真实 hidden/layers"""
    required = ["config.json"]
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    found = set()
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            if f in required + tokenizer_files:
                found.add(f)

    missing = [f for f in required if f not in found]
    if missing:
        print(f"[警告] 缺少文件: {missing}")
        return False

    # 读 config 抽取真实参数
    cfg = None
    for root, dirs, files in os.walk(local_dir):
        if "config.json" in files:
            with open(os.path.join(root, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            break

    if cfg:
        # Qwen3.5 / 多模态模型的真实字段在 text_config / vision_config 子字典
        sub = cfg.get("text_config", cfg.get("language_model_config", {}))

        def _get(key):
            # 优先顶层, 否则从 text_config 子字典取
            return cfg.get(key, sub.get(key, "未知"))

        print(f"[验证] 模型类型: {cfg.get('model_type', '未知')}")
        if sub and sub != cfg:
            print(f"[验证] 子结构: text_config (含 hidden_size/layers)")
        print(f"[验证] 层数: {_get('num_hidden_layers')}")
        print(f"[验证] 隐藏维度: {_get('hidden_size')}")
        # 一些常见字段
        for k in ("intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim"):
            v = _get(k)
            if v != "未知":
                print(f"[验证] {k}: {v}")
        if "num_experts" in cfg or "num_local_experts" in cfg or "num_experts" in sub:
            ne = cfg.get("num_experts", cfg.get("num_local_experts", sub.get("num_experts")))
            print(f"[验证] MoE 专家数: {ne}")
        # linear/full attention 标记
        layer_types = _get("layer_types")
        if isinstance(layer_types, list):
            n_full = sum(1 for t in layer_types if t == "full_attention")
            n_linear = sum(1 for t in layer_types if t == "linear_attention")
            print(f"[验证] 注意力分布: full={n_full}, linear={n_linear}")
        # 大小估计
        try:
            sz = sum(p.stat().st_size for p in Path(local_dir).rglob("*.safetensors"))
            print(f"[验证] 权重体量: {sz/1e9:.2f} GB")
        except Exception:
            pass

    print("[验证] 模型文件完整")
    return True


def resolve_model_cfg(config, model_name):
    """从 config["models"][name] 解析模型配置, 兼容旧 config["model"]"""
    models_dict = config.get("models")
    if models_dict and model_name in models_dict:
        return models_dict[model_name], model_name

    # 兼容旧版单模型 config
    if "model" in config:
        if model_name in (None, "default", config.get("default_model")):
            return config["model"], model_name
        raise KeyError(f"config 中没有找到模型: {model_name}")

    raise KeyError(f"config 中没有 models 字典, 也没有 model 字段")


def main():
    parser = argparse.ArgumentParser(description="阶段1: 数据准备 - 下载模型(支持多模型)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model_name", type=str, default=None,
                        help="config.models 里的 key, 例如 qwen3.5-2b / llama-3.2-1b")
    parser.add_argument("--model_id", type=str, default=None,
                        help="覆盖 model_id, 直接指定 HuggingFace/ModelScope 仓库")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "modelscope", "huggingface"],
                        help="下载源, 默认 auto 优先 ModelScope 失败则切 HuggingFace")
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com",
                        help="HuggingFace 镜像端点(国内推荐 hf-mirror.com)")
    parser.add_argument("--ignore_file_pattern", type=str, default=None,
                        help="ModelScope 下载时忽略的文件 glob, 例如 '*.bin' 可跳过重复的 pytorch_model.bin")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_dir = Path(args.config).parent
    default_name = config.get("default_model", "qwen3.5-2b")
    model_name = args.model_name or default_name
    model_cfg, resolved_name = resolve_model_cfg(config, model_name)

    log_dir = base_dir / "models" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    resume = ResumeManager(log_dir, f"02_model_{resolved_name}")
    if args.force:
        resume.reset()
        print(f"[强制] 重置 {resolved_name} 的断点续运行状态")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(log_dir / f"02_model_{resolved_name}_{timestamp}.log"))
    sys.stdout = log_saver

    try:
        if resume.is_completed():
            local_path = base_dir / model_cfg["local_path"]
            if local_path.exists() and any(local_path.glob("config.json")):
                print(f"[跳过] 模型 {resolved_name} 下载已完成")
                verify_model(str(local_path))
                log_saver.close()
                sys.stdout = log_saver.terminal
                return

        model_id = args.model_id or model_cfg.get("modelscope_id") or model_cfg["model_id"]
        local_path = base_dir / model_cfg["local_path"]

        if local_path.exists() and any(local_path.glob("config.json")):
            print(f"[跳过] 模型 {resolved_name} 已存在: {local_path}")
            verify_model(str(local_path))
            resume.mark_completed({"model_name": resolved_name, "model_id": model_id, "model_path": str(local_path)})
            log_saver.close()
            sys.stdout = log_saver.terminal
            return

        # 按 source 选择下载源
        downloaded = False
        last_err = None
        sources = []
        if args.source == "auto":
            sources = ["modelscope", "huggingface"]
        else:
            sources = [args.source]

        target_path = base_dir / model_cfg["local_path"]
        for src in sources:
            try:
                if src == "modelscope":
                    model_dir = download_model_from_modelscope(
                        model_id, str(target_path),
                        ignore_file_pattern=args.ignore_file_pattern)
                else:
                    model_dir = download_model_from_hf(model_id, str(target_path),
                                                       hf_endpoint=args.hf_endpoint)
                verify_model(model_dir)
                downloaded = True
                break
            except Exception as e:
                last_err = e
                print(f"[失败] {src} 源下载失败: {e}")
                continue

        if not downloaded:
            raise RuntimeError(f"模型 {model_id} 全部下载源都失败, 最后错误: {last_err}")

        print(f"\n{'='*60}")
        print(f"模型 {resolved_name} 下载完成!")
        print(f"模型路径: {model_dir}")
        print(f"{'='*60}")

        resume.mark_completed({
            "model_name": resolved_name,
            "model_id": model_id,
            "model_path": model_dir,
        })

    finally:
        log_saver.close()
        sys.stdout = log_saver.terminal


if __name__ == "__main__":
    main()
