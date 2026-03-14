# Minix
A minimal sequence modeling benchmark for:  LSTM Transformer Mamba2
```
# Mini Sequence Lab

A minimal sequence modeling benchmark for:

- LSTM
- Transformer
- Mamba2

Task:
- Character-level language modeling on tiny Shakespeare

Goals:
- Unified training / evaluation / benchmark pipeline
- Fair comparison across model families
- Simple benchmarking for throughput, memory, and latency

## Environment

Use the `mamba2rnn` conda environment.

## Quick Start
conda activate mamba2rnn
python -m src.data.prepare_tinyshakespeare

python -m src.train --config configs/lstm.yaml
python -m src.train --config configs/transformer.yaml
python -m src.train --config configs/mamba.yaml
```



### 📊 三模型训练结果对比表（基于最终数据）

| 模型名称                               | 最优验证 Loss | 最优 Step | 早停 Step | 困惑度 (PPL) | 训练时长（相对） | 核心特点                         |
| -------------------------------------- | ------------- | --------- | --------- | ------------ | ---------------- | -------------------------------- |
| **Transformer (Early Stop @12k step)** | **1.2293**    | 9,800     | 11,800    | **3.42**     | 中等             | 效果最优，泛化能力最强，收敛稳定 |
| **Mamba (Early Stop @8k step)**        | 1.2820        | 1,800     | 8,000     | 3.60         | 最快             | 收敛速度最快，线性复杂度优势明显 |
| **LSTM (Early Stop @3.6k step)**       | 1.2832        | 3,400     | 3,600     | 3.61         | 最短             | 最早触发早停，小数据集易过拟合   |

---

### ✅ 关键结论

1.  **效果排名**：Transformer > Mamba > LSTM（验证 Loss 越低、PPL 越小，模型效果越好）
2.  **收敛速度**：Mamba > LSTM > Transformer（早停 Step 越小，收敛越快）
3.  **泛化能力**：Transformer 表现最佳，LSTM 弱（验证 Loss 回升越早，过拟合风险越高）

---

### 📈 图表对应信息

- **LSTM**：曲线仅到 3,600 step，验证 Loss 从 1.2832 开始回升，最早触发早停
- **Mamba**：曲线到 8,000 step，验证 Loss 最低为 1.2820
- **Transformer**：曲线到 11,800 step，验证 Loss 最低为 1.2293，是本次实验的最优模型

---

