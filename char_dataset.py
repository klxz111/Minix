import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(self, encoded_data, block_size):
        self.data = torch.tensor(encoded_data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def build_datasets(root: str, block_size: int, split_ratio: float = 0.9):
    root = Path(root)
    input_path = root / "input.txt"
    assert input_path.exists(), f"missing data file: {input_path}"

    text = input_path.read_text(encoding="utf-8")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    encoded = [stoi[ch] for ch in text]
    split = int(len(encoded) * split_ratio)

    train_ids = encoded[:split]
    val_ids = encoded[split:]

    meta = {
        "vocab_size": len(chars),
        "stoi": stoi,
        "itos": itos,
        "num_tokens": len(encoded),
    }

    meta_path = root / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    train_ds = CharDataset(train_ids, block_size)
    val_ds = CharDataset(val_ids, block_size)
    return train_ds, val_ds, meta