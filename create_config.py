# create_config.py
import torch
from pathlib import Path

# 配置参数（根据你的项目实际情况修改，以下是 Shakespeare 数据集的通用配置）
config = {
    "data": {
        "root": "data/shakespeare",  # 数据集根目录，和训练时保持一致
        "block_size": 256,           # 序列长度
        "batch_size": 64             # 批次大小（生成时用不到，但配置里需要）
    },
    "model": {
        "vocab_size": 100,           # Shakespeare 数据集的词汇表大小通常约100
        "n_layer": 6,                # Transformer 层数
        "n_head": 6,                 # 注意力头数
        "n_embed": 384,              # 嵌入维度
        "dropout": 0.2,              # Dropout 概率
        "block_size": 256            # 序列长度，和data里保持一致
    },
    "train": {
        "max_iters": 5000,
        "learning_rate": 3e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}

# 保存配置到权重文件同目录
ckpt_dir = Path("outputs/transformer")
ckpt_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在
torch.save(config, ckpt_dir / "config.pt")

print(f"✅ 配置文件已生成：{ckpt_dir / 'config.pt'}")