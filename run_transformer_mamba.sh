#!/bin/bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba2rnn

# 进入项目根目录
cd ~/projects/mini-sequence-lab

echo "===== 开始训练Transformer（5 Epochs）====="
python -m src.train --config configs/tranformer.yaml

echo "===== Transformer训练完成，开始训练Mamba2 ====="
python -m src.train --config configs/mamba.yaml

echo "===== 所有模型训练完成！====="
# 自动生成Transformer/Mamba的可视化（复用lstm的脚本，只需改路径）
#python visualize_lstm.py --model transformer  # 后续改可视化脚本支持多模型
#python visualize_lstm.py --model mamba