import json
import os
import subprocess
import sys
from pathlib import Path


def bootstrap_runtime_env(project_dir=None):
    """脚本启动时的运行时环境引导（幂等）。

    - 默认使用 HF 镜像端点(模型走镜像下载更稳)，可用 HF_ENDPOINT 覆盖
    - 把 triton 缓存重定向到项目目录(避免沙箱对 ~/.triton 的限制)
    - 若模型需要 finegrained-fp8 内核(如 Ministral 3)，确保 triton/kernels 可用
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    base = Path(project_dir) if project_dir else Path(__file__).resolve().parent
    triton_cache = base / ".triton_cache"
    triton_home = base / ".triton_home"
    os.environ.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    os.environ.setdefault("TRITON_HOME", str(triton_home))
    triton_cache.mkdir(parents=True, exist_ok=True)
    triton_home.mkdir(parents=True, exist_ok=True)


def ensure_fp8_deps():
    """确保 finegrained-fp8 需要的最小 Python 依赖可用(缺失时自动 pip 安装)。

    只用于原生 fp8 权重的模型(如 Ministral 3)。安装失败仅告警不中断。
    返回 (triton_ok, kernels_ok)。
    """
    triton_ok, kernels_ok = True, True
    try:
        import triton  # noqa: F401
    except ImportError:
        triton_ok = False
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "triton-windows"]
            )
            import triton  # noqa: F401
            triton_ok = True
        except Exception as e:
            print(f"[警告] 自动安装 triton-windows 失败: {e}")
    try:
        import kernels  # noqa: F401
    except ImportError:
        kernels_ok = False
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "kernels==0.16.0"]
            )
            import kernels  # noqa: F401
            kernels_ok = True
        except Exception as e:
            print(f"[警告] 自动安装 kernels 失败: {e}")
    return triton_ok, kernels_ok


def model_needs_fp8(model_cfg, model_path):
    """判断本地模型是否为原生 fp8 权重(需要 finegrained-fp8 内核)。

    读取模型目录 config.json 的 quantization_config.quant_method。
    """
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                c = json.load(f)
            qc = c.get("quantization_config", {})
            if (qc or {}).get("quant_method") == "fp8":
                return True
        except Exception:
            return False
    return False


def _dequant_fp8_weight(weight, scale_inv, dtype):
    """把 (fp8 weight, scale_inv) 反量化为 dtype 的普通浮点权重。

    语义与 transformers.integrations.finegrained_fp8.Fp8Dequantize 一致:
        w = weight.to(fp32) * scale_inv
    标量 scale_inv -> per-tensor; 二维 scale 网格 -> per-block (block 尺寸由网格反推)。
    """
    import torch
    if weight.element_size() > 1:                      # 已是普通浮点, 无需反量化
        return weight.to(dtype)
    w = weight.detach().to(torch.float32)
    s = scale_inv.detach()
    if s.dtype == torch.uint8:                         # MXFP8 的 E8M0 以字节存储: scale = 2^(b-127)
        s = (s.to(torch.float32) - 127.0).exp2()
    else:
        s = s.to(torch.float32)
    if s.numel() == 1:                                 # per-tensor
        return (w * s.reshape(())).to(dtype)
    rows, cols = w.shape[-2], w.shape[-1]
    srows, scols = s.shape[-2], s.shape[-1]
    if rows % srows or cols % scols:
        raise ValueError(f"weight {tuple(w.shape)} 与 scale {tuple(s.shape)} 网格不匹配")
    bm, bn = rows // srows, cols // scols
    q = w.reshape(-1, srows, bm, scols, bn)
    sb = s.reshape(-1, srows, scols, 1, 1)
    return (q * sb).reshape(w.shape).to(dtype)


def dequantize_fp8_model(model, dtype=None):
    """把 finegrained-FP8 权重就地反量化为普通浮点 nn.Linear, 使模型可反向传播。

    fp8 前向走 Triton/DeepGEMM 内核, 未注册 autograd, 无法求 dB/dh。反量化后
    FP8Linear.forward 会走 `weight.element_size() > 1` 分支的 F.linear, 恢复可微。

    返回被替换的模块数。
    """
    import torch
    import torch.nn as nn
    if dtype is None:
        dtype = torch.bfloat16
    try:
        from transformers.integrations.finegrained_fp8 import FP8Linear, FP8Experts
    except Exception:
        return 0

    n_done = 0
    for _, mod in model.named_modules():
        if isinstance(mod, FP8Linear):                 # 含 FP8GroupedLinear 子类
            if mod.weight.element_size() <= 1:
                mod.weight = nn.Parameter(
                    _dequant_fp8_weight(mod.weight.data, mod.weight_scale_inv.data, dtype),
                    requires_grad=False)
                if mod.bias is not None:
                    mod.bias = nn.Parameter(mod.bias.data.to(dtype), requires_grad=False)
                n_done += 1
        elif isinstance(mod, FP8Experts):
            for wname, sname in (("gate_up_proj", "gate_up_proj_scale_inv"),
                                 ("up_proj", "up_proj_scale_inv"),
                                 ("down_proj", "down_proj_scale_inv")):
                w = getattr(mod, wname, None)
                s = getattr(mod, sname, None)
                if w is None or s is None or w.element_size() <= 1:
                    continue
                setattr(mod, wname, nn.Parameter(
                    _dequant_fp8_weight(w.data, s.data, dtype), requires_grad=False))
            n_done += 1
    return n_done


def load_config(config_path="config.json"):
    """读取多模型 config.json。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model_cfg(config, model_name=None):
    """从多模型 config 解析出目标模型配置。

    返回 (model_name, model_cfg)。
    兼容旧的单模型格式（顶层 model 字段）。
    """
    if "models" in config:
        name = model_name or config.get("default_model")
        if name is None:
            raise ValueError("config 未设置 default_model 且未指定 --model")
        if name not in config["models"]:
            raise ValueError(f"模型 '{name}' 不在 config.models 中，可选: {list(config['models'])}")
        cfg = config["models"][name]
        # 合并通用 phase2_sae 训练超参到 cfg["_train"]，供 SAE 训练脚本使用
        cfg.setdefault("_train", {})
        for kind in ("text_sae", "vision_sae", "unified_vl_sae"):
            if kind in config["phase2_sae"]:
                merged = dict(config["phase2_sae"][kind])
                model_kind = cfg.get("sae", {}).get(kind, {})
                # 模型级 sae 覆盖: input_dim/hidden_dim/k 优先; 其余用通用超参
                merged.update(model_kind)
                cfg["_train"][kind] = merged
        return name, cfg
    return model_name or "legacy", config["model"]


def apply_path_template(path_tmpl, model_name):
    """把 paths 里的 {model_name} 模板替换为实际模型名。"""
    return path_tmpl.format(model_name=model_name)


def get_target_layers(model_cfg, num_layers=None):
    """返回目标层列表。

    优先用 config 里配的 target_layers；
    没有则按 1/3 与 2/3 比例生成（最多 3 层）。
    """
    tl = model_cfg.get("target_layers")
    if tl:
        return tl
    n = num_layers or model_cfg.get("num_hidden_layers") or 24
    if n <= 0:
        return [0]
    l1 = int(n / 3)
    l2 = int(2 * n / 3)
    layers = [l for l in (l1, l2) if 0 <= l < n]
    if len(layers) < 1:
        return [0]
    return sorted(set(layers))


def load_base_model(model_path, model_cfg=None, torch_dtype=None, device_map="auto", trust_remote_code=True):
    """加载因果LM基座(带可选 LoRA 适配的通用入口)。

    - phi 系列: 用 transformers 原生 Phi3ForCausalLM(trust_remote_code=False)，
      绕开模型目录内 4.x 旧 trust_remote_code 自定义 modeling
      (自定义类引用了 transformers 5.x 已移除的 Cache API, 彻底规避注意力量匹配问题)。
    - 其余模型: 走 AutoModelForCausalLM。
    """
    from transformers import AutoModelForCausalLM
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    kwargs["device_map"] = device_map
    mtype = (model_cfg or {}).get("model_type", "")
    if str(mtype).lower().startswith("phi"):
        from transformers import Phi3ForCausalLM
        return Phi3ForCausalLM.from_pretrained(str(model_path), trust_remote_code=False, **kwargs)
    return AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=trust_remote_code, **kwargs)


def auto_dims_from_loaded_model(model):
    """从已加载的 model.config 读取 (hidden_size, num_hidden_layers)。

    重点处理 Qwen3.5/多模态模型的 text_config 子字典嵌套。
    读不到返回 (None, None)。
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None, None
    c = getattr(cfg, "text_config", None) or cfg
    hidden = getattr(c, "hidden_size", None) or getattr(c, "d_model", None)
    n_layers = getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", None)
    return hidden, n_layers


def auto_dims_from_local_config(model_cfg, model_path):
    """从本地 models/<name>/config.json 读取 (hidden_size, num_hidden_layers)。

    优先返回 config 里已填写的值；缺失时读本地 config。
    读不到返回 (None, None)。
    """
    hidden = model_cfg.get("hidden_size")
    n_layers = model_cfg.get("num_hidden_layers")
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                c = json.load(f)
            tc = c.get("text_config", {})
            hidden = hidden or c.get("hidden_size") or tc.get("hidden_size") or tc.get("d_model")
            n_layers = n_layers or c.get("num_hidden_layers") or tc.get("num_hidden_layers")
        except Exception:
            pass
    return hidden, n_layers