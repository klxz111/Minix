import argparse

import torch
from torch.utils.data import DataLoader

from src.data.char_dataset import build_datasets
from src.models import build_model
from src.utils import get_amp_dtype, get_autocast_context, safe_perplexity


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    meta = ckpt["meta"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_enabled = cfg["train"].get("amp", True)
    amp_dtype = get_amp_dtype(cfg["train"].get("amp_dtype", "float16"))

    _, val_ds, _ = build_datasets(
        root=cfg["data"]["root"],
        block_size=cfg["data"]["block_size"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 2),
        pin_memory=device.startswith("cuda"),
    )

    model = build_model(
        cfg["model"],
        vocab_size=meta["vocab_size"],
        block_size=cfg["data"]["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= args.batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with get_autocast_context(device, amp_enabled, amp_dtype):
            _, loss = model(x, y)

        losses.append(loss.item())

    val_loss = sum(losses) / len(losses)
    print({
        "ckpt": args.ckpt,
        "val_loss": round(val_loss, 4),
        "perplexity": round(safe_perplexity(val_loss), 4),
    })


if __name__ == "__main__":
    main()