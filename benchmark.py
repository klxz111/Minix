import argparse
import time

import torch

from src.models import build_model
from src.utils import count_parameters, get_amp_dtype, get_autocast_context


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    meta = ckpt["meta"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "benchmark expects CUDA"

    amp_enabled = cfg["train"].get("amp", True)
    amp_dtype = get_amp_dtype(cfg["train"].get("amp_dtype", "float16"))

    model = build_model(
        cfg["model"],
        vocab_size=meta["vocab_size"],
        block_size=cfg["data"]["block_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.randint(
        low=0,
        high=meta["vocab_size"],
        size=(args.batch_size, args.seq_len),
        device=device,
        dtype=torch.long,
    )

    for _ in range(args.warmup):
        with get_autocast_context(device, amp_enabled, amp_dtype):
            _ = model(x)[0]
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    for _ in range(args.iters):
        with get_autocast_context(device, amp_enabled, amp_dtype):
            _ = model(x)[0]
    torch.cuda.synchronize()
    elapsed = time.time() - start

    total_tokens = args.batch_size * args.seq_len * args.iters
    toks_per_sec = total_tokens / elapsed
    latency_ms = elapsed / args.iters * 1000
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    print({
        "model": cfg["model"]["name"],
        "params": count_parameters(model),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "tokens_per_sec": round(toks_per_sec, 2),
        "latency_ms_per_iter": round(latency_ms, 3),
        "peak_mem_mb": round(peak_mem_mb, 2),
    })


if __name__ == "__main__":
    main()