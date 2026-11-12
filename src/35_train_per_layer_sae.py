# -*- coding: utf-8 -*-
"""Step 2-1: 按层新训 per-layer SAE (因果效度主线的前置)

动机 (已核实的关键勘误):
  03_collect_text_activations.py 把 [target_layers] 三个层的激活写进同一个 npz
  (键 `{i}_{layer}`, 值 shape `(last_n_tokens, hidden)`); 而 05_train_text_sae.py 的
  ActivationDataset **不按 layer 过滤**, 把三个层的行池化在一起算 mean/std 并训练。
  ⇒ 现有 per-model SAE 的单个特征**无法归属到某一层**, 不能做层定位的单特征干预。

本脚本: 按 layer 后缀过滤 npz 行, 每层独立算 mean/std + 训练一个 BatchTopKSAE,
        输出到 output/phaseStep2/layer_sae/{model}/layer{L}/best_sae.pt。

口径:
  - train/val = 80/20 (seed=42, 按 row 打散);
  - mean/std **只用 train 行**估计 (避免 val 泄漏);
  - 120 epoch 上限 + val recon 早停 (patience=15), best 按 val recon 保存;
  - 复用 05 的 BatchTopKSAE (BatchTopK + encoder bias + decoder 列归一化)。

资源: CPU 严格节流 (OMP/MKL=1, torch threads=1), 单进程串行, 独显。
输出: output/phaseStep2/layer_sae/{model}/layer{L}/best_sae.pt
      output/phaseStep2/layer_sae/{model}/layer{L}/per_layer_sae_metrics.json
      output/phaseStep2/layer_sae/per_layer_sae_summary.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from importlib import import_module

torch.set_num_threads(1)

from model_utils import load_config, resolve_model_cfg, apply_path_template

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


def load_layer_rows(npz_path, layer):
    """从 npz 里抽出指定层的全部行: (N, hidden)."""
    data = np.load(npz_path)
    rows = []
    for k in data.files:
        if k.split("_")[-1] != str(layer):
            continue
        arr = np.asarray(data[k], dtype=np.float32)
        if arr.ndim == 2:
            rows.append(arr)
        elif arr.ndim == 1:
            rows.append(arr[None, :])
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * x.sum())) - (n + 1) / n)


def train_one_layer(rows, sae_cfg, device, seed=42, val_frac=0.2,
                    num_epochs=120, patience=15, batch_size=None):
    hidden = rows.shape[1]
    input_dim = sae_cfg["input_dim"]
    hidden_dim = sae_cfg["hidden_dim"]
    k = sae_cfg["k"]
    lr = sae_cfg["learning_rate"]
    weight_decay = sae_cfg.get("weight_decay", 0.01)
    l1_lambda = sae_cfg.get("l1_lambda", 1e-4)
    warmup_epochs = sae_cfg.get("warmup_epochs", 3)
    resample_every = sae_cfg.get("resample_dead_every", 5)

    if hidden != input_dim:
        print(f"  [调整] input_dim: {input_dim} -> {hidden}")
        input_dim = hidden

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(rows))
    n_val = max(1, int(len(rows) * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_raw = rows[train_idx]
    val_raw = rows[val_idx]

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0) + 1e-8
    train = (train_raw - mean) / std
    val = (val_raw - mean) / std

    bs = batch_size or sae_cfg.get("batch_size", 2048)
    bs = min(bs, max(1, len(train)))

    sae = BatchTopKSAE(input_dim, hidden_dim, k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(train) / bs)))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_t = torch.tensor(train, dtype=torch.float32)
    val_t = torch.tensor(val, dtype=torch.float32).to(device)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    history = []
    no_improve = 0
    neuron_activity = torch.zeros(hidden_dim, dtype=torch.long, device=device)
    all_topk = []

    for epoch in range(num_epochs):
        sae.train()
        order = torch.randperm(len(train_t))
        ep_loss = ep_recon = 0.0
        nb = 0
        for s in range(0, len(train_t), bs):
            idx = order[s:s + bs]
            batch = train_t[idx].to(device)
            reconstructed, sparse_hidden, topk_idx = sae(batch)
            total_loss, recon_loss, _ = sae.loss(batch, reconstructed, sparse_hidden, l1_lambda)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            sae._normalize_decoder()
            neuron_activity.scatter_add_(0, topk_idx.flatten(),
                                         torch.ones_like(topk_idx.flatten(), dtype=torch.long))
            ep_loss += total_loss.item()
            ep_recon += recon_loss.item()
            nb += 1

        sae.eval()
        with torch.no_grad():
            v_recon, v_sparse, v_topk = sae(val_t)
            val_recon = float(torch.nn.functional.mse_loss(v_recon, val_t).item())
            cos = float(torch.nn.functional.cosine_similarity(
                v_recon, val_t, dim=-1).mean().item())
            all_topk.append(v_topk.cpu().numpy().ravel())

        n_dead = int((neuron_activity == 0).sum().item())
        history.append({"epoch": epoch + 1, "train_loss": ep_loss / nb,
                        "train_recon": ep_recon / nb, "val_recon": val_recon,
                        "val_cos": cos, "dead": n_dead})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}: train_recon={ep_recon/nb:.5f} "
                  f"val_recon={val_recon:.5f} cos={cos:.4f} dead={n_dead}/{hidden_dim}")

        if val_recon < best_val - 1e-6:
            best_val = val_recon
            best_epoch = epoch + 1
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in sae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [早停] epoch {epoch+1} (best={best_epoch}, val_recon={best_val:.5f})")
                break

        if resample_every > 0 and (epoch + 1) % resample_every == 0 and n_dead > 0:
            dead_mask = neuron_activity == 0
            sae.resample_dead_neurons(train_t, dead_mask, device)
            neuron_activity.zero_()

    if best_state is not None:
        sae.load_state_dict(best_state)
    sae.eval()

    # 最终 val 指标 (用 best 权重)
    with torch.no_grad():
        v_recon, v_sparse, v_topk = sae(val_t)
        val_recon = float(torch.nn.functional.mse_loss(v_recon, val_t).item())
        val_cos = float(torch.nn.functional.cosine_similarity(v_recon, val_t, dim=-1).mean().item())
    counts = np.bincount(np.concatenate(all_topk), minlength=hidden_dim)
    metrics = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "k": k,
        "n_rows_total": int(len(rows)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "val_recon_mse": val_recon,
        "val_cos": val_cos,
        "dead_features": int((counts == 0).sum()),
        "active_features": int((counts > 0).sum()),
        "gini_feature_usage": gini(counts),
    }
    ckpt = {
        "model_state_dict": sae.state_dict(),
        "config": {"input_dim": input_dim, "hidden_dim": hidden_dim, "k": k},
        "mean": mean,
        "std": std,
        "layer": None,  # 由调用方填充
        "epoch": best_epoch,
        "loss": best_val,
        "metrics": metrics,
    }
    return sae, ckpt, metrics, history


def run_model(args, config, base_dir, m, out_root):
    _, model_cfg = resolve_model_cfg(config, m)
    sae_cfg = model_cfg["_train"]["text_sae"]
    target_layers = list(model_cfg["target_layers"])
    if args.layers:
        target_layers = [int(x) for x in args.layers.split(",")]

    act_dir = base_dir / apply_path_template(config["paths"]["text_activation_dir"], m)
    npz_files = sorted(act_dir.glob("text_activations_*.npz"))
    if not npz_files:
        print(f"[错误] {m}: 未找到激活 npz (需先跑 03)")
        return []
    npz_path = npz_files[-1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[{m}] npz={npz_path.name} target_layers={target_layers} device={device}")

    results = []
    for layer in target_layers:
        out_dir = out_root / m / f"layer{layer}"
        out_dir.mkdir(parents=True, exist_ok=True)
        resume = ResumeManager(out_dir, f"35_layer_sae_s{int(getattr(args, 'seed', 42))}")
        mfile = out_dir / "per_layer_sae_metrics.json"
        if resume.is_completed() and mfile.exists() and not args.force:
            print(f"  [跳过] layer{layer} 已完成")
            results.append(json.load(open(mfile, encoding="utf-8")))
            continue

        rows = load_layer_rows(npz_path, layer)
        if rows is None:
            print(f"  [跳过] layer{layer}: npz 中无该层数据")
            continue
        print(f"  [layer{layer}] rows={rows.shape}")

        seed = int(getattr(args, "seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        sae, ckpt, metrics, history = train_one_layer(
            rows, sae_cfg, device, seed=seed,
            num_epochs=args.num_epochs, patience=args.patience)
        ckpt["layer"] = layer
        metrics["layer"] = layer
        metrics["model"] = m
        metrics["npz"] = str(npz_path.name)

        torch.save(ckpt, str(out_dir / "best_sae.pt"))
        with open(out_dir / "loss_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(mfile, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        resume.mark_completed({"layer": layer, "val_recon_mse": metrics["val_recon_mse"]})
        print(f"  [layer{layer}] 完成: val_recon={metrics['val_recon_mse']:.5f} "
              f"cos={metrics['val_cos']:.4f} dead={metrics['dead_features']}/{metrics['hidden_dim']}")
        results.append(metrics)
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Step2-1: per-layer SAE 训练 (因果效度前置)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layers", type=str, default=None, help="逗号分隔, 默认用 config target_layers")
    parser.add_argument("--num_epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="训练随机种子(初始化/打散/val 划分), 用于估计 SAE 训练方差")
    parser.add_argument("--out_dir", type=str, default="output/phaseStep2/layer_sae",
                        help="输出根目录 (相对 base_dir), 默认 output/phaseStep2/layer_sae")
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = Path(args.config).parent

    models = [args.model] if args.model else list(config["models"].keys())
    out_root = base_dir / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(log_dir / f"35_layer_sae_{ts}.log"))
    sys.stdout = log_saver

    summary = {"timestamp": ts, "models": {}}
    try:
        for m in models:
            try:
                res = run_model(args, config, base_dir, m, out_root)
                summary["models"][m] = res
            except Exception as e:
                print(f"[失败] {m}: {e}")
                summary["models"][m] = {"error": str(e)[:400]}
            # 每模型落盘一次, 便于中断续跑
            with open(out_root / "per_layer_sae_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
    finally:
        log_saver.close()
        sys.stdout = log_saver.terminal

    print(f"[完成] 汇总: {out_root / 'per_layer_sae_summary.json'}")


if __name__ == "__main__":
    main()
