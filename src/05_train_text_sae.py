import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model_utils import load_config, resolve_model_cfg, apply_path_template


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


class BatchTopKSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='linear')
        nn.init.zeros_(self.decoder.bias)
        self._normalize_decoder()
    
    def _normalize_decoder(self):
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            self.decoder.weight.data = self.decoder.weight.data / norms
    
    def resample_dead_neurons(self, dataset_samples, dead_mask, device):
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return
        dead_indices = torch.where(dead_mask)[0]
        sample_idx = torch.randint(0, len(dataset_samples), (n_dead,))
        samples = dataset_samples[sample_idx].to(device)
        with torch.no_grad():
            recon = self.forward(samples)[0]
            errors = samples - recon
            for i, dead_idx in enumerate(dead_indices):
                err_vec = errors[i]
                err_norm = err_vec.norm() + 1e-8
                new_dir = err_vec / err_norm
                self.encoder.weight.data[dead_idx] = new_dir * 0.1
                self.encoder.bias.data[dead_idx] = 0.0
                self.decoder.weight.data[:, dead_idx] = new_dir
        self._normalize_decoder()
    
    def encode(self, x):
        hidden = self.encoder(x)
        topk_vals, topk_idx = torch.topk(hidden, self.k, dim=-1)
        sparse = torch.zeros_like(hidden)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse, topk_idx
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        sparse_hidden, topk_idx = self.encode(x)
        reconstructed = self.decode(sparse_hidden)
        return reconstructed, sparse_hidden, topk_idx
    
    def loss(self, x, reconstructed, sparse_hidden, l1_lambda=1e-4):
        recon_loss = nn.functional.mse_loss(reconstructed, x)
        l1_loss = sparse_hidden.abs().mean()
        return recon_loss + l1_lambda * l1_loss, recon_loss, l1_loss


class ActivationDataset(Dataset):
    def __init__(self, activation_file, max_samples=None):
        data = np.load(activation_file)
        self.samples = []
        
        for key in data.files:
            arr = data[key]
            if arr.ndim == 2:
                for row in arr:
                    self.samples.append(row.astype(np.float32))
            elif arr.ndim == 1:
                self.samples.append(arr.astype(np.float32))
            if max_samples and len(self.samples) >= max_samples:
                break
        
        self.samples = np.array(self.samples) if self.samples else np.zeros((1, 2048), dtype=np.float32)
        self.mean = self.samples.mean(axis=0)
        self.std = self.samples.std(axis=0) + 1e-8
        self.samples = (self.samples - self.mean) / self.std
        
        print(f"[SAE数据] 样本数: {len(self.samples)}, 维度: {self.samples.shape[-1]}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)


def train_sae(activation_file, output_dir, sae_cfg, device):
    input_dim = sae_cfg["input_dim"]
    hidden_dim = sae_cfg["hidden_dim"]
    k = sae_cfg["k"]
    lr = sae_cfg["learning_rate"]
    batch_size = sae_cfg["batch_size"]
    num_epochs = sae_cfg["num_epochs"]
    weight_decay = sae_cfg.get("weight_decay", 0.01)
    l1_lambda = sae_cfg.get("l1_lambda", 1e-4)
    warmup_epochs = sae_cfg.get("warmup_epochs", 2)
    resample_every = sae_cfg.get("resample_dead_every", 5)
    use_decoder_norm = sae_cfg.get("decoder_norm", True)
    
    dataset = ActivationDataset(activation_file)
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, num_workers=0)
    
    actual_dim = dataset.samples.shape[-1]
    if actual_dim != input_dim:
        print(f"[调整] input_dim: {input_dim} -> {actual_dim}")
        input_dim = actual_dim
    
    sae = BatchTopKSAE(input_dim, hidden_dim, k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_steps = num_epochs * len(dataloader)
    warmup_steps = warmup_epochs * len(dataloader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_loss = float("inf")
    loss_history = []
    global_step = 0
    
    neuron_activity = torch.zeros(hidden_dim, dtype=torch.long, device=device)
    dataset_tensor = torch.tensor(dataset.samples, dtype=torch.float32)
    
    for epoch in range(num_epochs):
        sae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_l1 = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
            batch = batch.to(device)
            reconstructed, sparse_hidden, topk_idx = sae(batch)
            total_loss, recon_loss, l1_loss = sae.loss(batch, reconstructed, sparse_hidden, l1_lambda)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if use_decoder_norm:
                sae._normalize_decoder()
            
            neuron_activity.scatter_add_(0, topk_idx.flatten(), torch.ones_like(topk_idx.flatten(), dtype=torch.long))
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_l1 += l1_loss.item()
            num_batches += 1
            global_step += 1
        
        avg_loss = epoch_loss / num_batches
        avg_recon = epoch_recon / num_batches
        avg_l1 = epoch_l1 / num_batches
        
        n_dead = (neuron_activity == 0).sum().item()
        n_active = hidden_dim - n_dead
        
        loss_history.append({
            "epoch": epoch + 1,
            "total": avg_loss,
            "recon": avg_recon,
            "l1": avg_l1,
            "active_features": n_active,
            "dead_features": n_dead,
        })
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.6f}, Recon={avg_recon:.6f}, L1={avg_l1:.6f}, "
              f"Active={n_active}/{hidden_dim}, Dead={n_dead}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": sae.state_dict(),
                "config": {"input_dim": input_dim, "hidden_dim": hidden_dim, "k": k},
                "mean": dataset.mean,
                "std": dataset.std,
                "epoch": epoch,
                "loss": best_loss,
            }, str(output_dir / "best_sae.pt"))
            print(f"  [保存] 最佳模型 (Loss: {best_loss:.6f})")
        
        if resample_every > 0 and (epoch + 1) % resample_every == 0 and n_dead > 0:
            dead_mask = neuron_activity == 0
            print(f"  [重采样] 重置 {n_dead} 个死亡神经元")
            sae.resample_dead_neurons(dataset_tensor, dead_mask, device)
            neuron_activity.zero_()
    
    torch.save({
        "model_state_dict": sae.state_dict(),
        "config": {"input_dim": input_dim, "hidden_dim": hidden_dim, "k": k},
        "mean": dataset.mean,
        "std": dataset.std,
        "epoch": num_epochs,
        "loss": avg_loss,
    }, str(output_dir / "final_sae.pt"))
    
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    
    return sae, dataset


def analyze_features(sae, dataset, output_dir, device, top_k_features=50):
    print(f"\n[分析] SAE特征分析...")
    sae.eval()
    all_topk = []
    
    with torch.no_grad():
        for i in range(0, min(len(dataset), 2000), 100):
            end = min(i + 100, len(dataset))
            batch = torch.tensor(dataset.samples[i:end], dtype=torch.float32).to(device)
            sparse_hidden, topk_idx = sae.encode(batch)
            all_topk.append(topk_idx.cpu().numpy())
    
    if all_topk:
        all_topk = np.concatenate(all_topk, axis=0)
        feature_counts = np.bincount(all_topk.flatten(), minlength=sae.hidden_dim)
        top_features = np.argsort(feature_counts)[::-1][:top_k_features]
        
        analysis = {
            "top_features": top_features.tolist(),
            "feature_counts": feature_counts[top_features].tolist(),
            "total_features": sae.hidden_dim,
            "active_features": int((feature_counts > 0).sum()),
            "dead_features": int((feature_counts == 0).sum()),
            "sparsity": float((all_topk == 0).mean()),
        }
        
        with open(output_dir / "feature_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"  活跃特征: {analysis['active_features']}/{analysis['total_features']}")
        print(f"  死亡特征: {analysis['dead_features']}")
        print(f"  Top-5: {top_features[:5].tolist()}")


def main():
    parser = argparse.ArgumentParser(description="阶段2: SAE训练 - 文本SAE")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model", type=str, default=None, help="模型名(见 config.models), 默认用 default_model")
    parser.add_argument("--force", action="store_true", help="强制重新运行，忽略断点续运行状态")
    args = parser.parse_args()

    config = load_config(args.config)

    base_dir = Path(args.config).parent
    model_name, model_cfg = resolve_model_cfg(config, args.model)
    # 模型级 sae (input_dim/hidden_dim/k) 已合并通用 phase2_sae 训练超参到 _train.text_sae
    sae_cfg = model_cfg["_train"]["text_sae"]
    output_dir = base_dir / apply_path_template(config["paths"]["text_sae_dir"], model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    resume = ResumeManager(output_dir, "05_text_sae")
    if args.force:
        resume.reset()
        print("[强制] 重置断点续运行状态")
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(log_dir / f"05_text_sae_{timestamp}.log"))
    sys.stdout = log_saver

    if resume.is_completed() and (output_dir / "best_sae.pt").exists():
        print("[跳过] 文本SAE训练已完成")
        print(f"  模型: {output_dir / 'best_sae.pt'}")
        print("  如需重新训练，请使用 --force 参数")
        log_saver.close()
        sys.stdout = log_saver.terminal
        return
    
    act_dir = base_dir / apply_path_template(config["paths"]["text_activation_dir"], model_name)
    npz_files = sorted(act_dir.glob("text_activations_*.npz"))
    
    if not npz_files:
        print("[错误] 未找到文本激活文件！请先运行 03_collect_text_activations.py")
        return
    
    act_file = npz_files[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{'='*60}")
    print(f"阶段2: 训练文本SAE (BatchTopK)")
    print(f"{'='*60}")
    print(f"模型: {model_name}")
    print(f"激活文件: {act_file}")
    print(f"输入维度: {sae_cfg['input_dim']}")
    print(f"隐藏维度: {sae_cfg['hidden_dim']}")
    print(f"Top-K: {sae_cfg['k']}")
    
    sae, dataset = train_sae(str(act_file), output_dir, sae_cfg, device)
    analyze_features(sae, dataset, output_dir, device)
    
    print(f"\n{'='*60}")
    print(f"文本SAE训练完成!")
    print(f"模型: {output_dir / 'best_sae.pt'}")
    print(f"{'='*60}")

    resume.mark_completed({"model_path": str(output_dir / "best_sae.pt")})

    log_saver.close()
    sys.stdout = log_saver.terminal


if __name__ == "__main__":
    main()
