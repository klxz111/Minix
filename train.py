import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.char_dataset import build_datasets
from src.models import build_model
from src.utils import (
    append_jsonl,
    count_parameters,
    ensure_dir,
    get_amp_dtype,
    get_autocast_context,
    load_yaml,
    save_json,
    set_seed,
)


# ===== 新增：Early Stopping 类（防过拟合）=====
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001, mode="min"):
        self.patience = patience  # tiny_shakespeare 建议设为2
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "min" and score > self.best_score - self.min_delta) or \
             (self.mode == "max" and score < self.best_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️ Early Stopping 触发！最优验证Loss={self.best_score:.4f}")
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=False, amp_dtype=torch.float16, max_batches=50):
    model.eval()
    losses = []
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with get_autocast_context(device, amp_enabled, amp_dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    device = cfg["system"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    amp_enabled = cfg["train"].get("amp", True) if device.startswith("cuda") else False
    amp_dtype = get_amp_dtype(cfg["train"].get("amp_dtype", "float16"))

    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)
    # ===== 新增：清空旧 metrics.jsonl（修复）=====
metrics_path = output_dir / "metrics.jsonl"
if metrics_path.exists():
    metrics_path.unlink()  # 删除旧文件
    print(f"✅ 已清空旧日志文件: {metrics_path}")


save_json(cfg, output_dir / "config.json")

train_ds, val_ds, meta = build_datasets(
    root=cfg["data"]["root"],
    block_size=cfg["data"]["block_size"],
)
    save_json(cfg, output_dir / "config.json")

    train_ds, val_ds, meta = build_datasets(
        root=cfg["data"]["root"],
        block_size=cfg["data"]["block_size"],
    )
    save_json(meta, output_dir / "meta.json")

    pin_memory = device.startswith("cuda")
    num_workers = cfg["data"].get("num_workers", 0)  # WSL 兼容

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # ===== 模型规模完全不变 =====
    model = build_model(
        cfg["model"],
        vocab_size=meta["vocab_size"],
        block_size=cfg["data"]["block_size"],
    ).to(device)

    print(f"model: {cfg['model']['name']}")
    print(f"params: {count_parameters(model):,}")
    print(f"device: {device}")

    # ===== 强化正则化：AdamW + 权重衰减 =====
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),  # 核心正则化
    )

    use_scaler = amp_enabled and device.startswith("cuda") and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_val = float("inf")
    global_step = 0
    token_counter = 0
    time_start = time.time()

    # ===== 初始化 Early Stopping =====
    early_stopper = EarlyStopping(
        patience=cfg["train"].get("early_stop_patience", 2),
        min_delta=cfg["train"].get("early_stop_min_delta", 0.001),
        mode="min"
    )

    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    for epoch in range(cfg["train"]["max_epochs"]):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with get_autocast_context(device, amp_enabled, amp_dtype):
                _, loss = model(x, y)

            # ===== 梯度裁剪（防梯度爆炸+正则化）=====
            if use_scaler:
                scaler.scale(loss).backward()
                if cfg["train"].get("grad_clip", None) is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["train"].get("grad_clip", None) is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
                optimizer.step()

            global_step += 1
            token_counter += x.numel()

            elapsed = max(time.time() - time_start, 1e-6)
            toks_per_sec = token_counter / elapsed
            pbar.set_postfix(loss=f"{loss.item():.4f}", tok_s=f"{toks_per_sec:.0f}")

            if global_step % cfg["train"]["eval_interval"] == 0:
                val_loss = evaluate(
                    model,
                    val_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    max_batches=cfg["train"].get("eval_batches", 50),
                )

                record = {
                    "step": global_step,
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                    "tokens_per_sec": float(toks_per_sec),
                }
                append_jsonl(record, output_dir / "metrics.jsonl")
                print(record)

                ckpt = {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "meta": meta,
                }

                # ===== 保存 last.pt =====
                torch.save(ckpt, output_dir / "last.pt")
                torch.save({"model": model.state_dict(), "epoch": epoch}, output_dir / "best.pt")
                # ===== 核心：只保存最优 checkpoint =====
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(ckpt, output_dir / "best.pt")
                    print(f"[best] val_loss={best_val:.4f}")

                # ===== 检查 Early Stopping =====
                if early_stopper(val_loss):
                    pbar.close()
                    break
        if early_stopper.early_stop:
            break

    print("training done")


if __name__ == "__main__":
    main()