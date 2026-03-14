import os
from collections import Counter

def build_vocab(data_root):
    """适配 Shakespeare 数据集的字符级词汇表构建函数"""
    data_path = os.path.join(data_root, "input.txt")
    if not os.path.exists(data_path):
        # 降级：如果数据集不存在，用默认字符表（保证能跑）
        chars = sorted(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;!?\'-"\n'))
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
    
    word2idx = {ch:i for i,ch in enumerate(chars)}
    idx2word = {i:ch for i,ch in enumerate(chars)}
    return word2idx, idx2word
