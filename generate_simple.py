import torch
import torch.nn.functional as F
import sys
from pathlib import Path  # 关键：导入 Path
import argparse

# 把项目根目录加入路径
sys.path.append(str(Path(__file__).parent.parent))

# --------------------------
# 1. 导入你真实的模型类（src/models/transformer_lm.py 里的 TransformerLM）
# --------------------------
try:
    from src.models.transformer_lm import TransformerLM
    from src.data import build_vocab
    print("✅ 成功导入 TransformerLM 和 build_vocab")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# --------------------------
# 2. 主生成逻辑
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="模型权重路径")
    parser.add_argument("--prompt", required=True, help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成token数")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --------------------------
    # 3. 加载词汇表（和训练时一致）
    # --------------------------
    vocab = build_vocab("data/shakespeare")
    vocab_stoi = vocab.stoi
    vocab_itos = vocab.itos
    print(f"✅ 加载词汇表，大小: {len(vocab_stoi)}")

    # --------------------------
    # 4. 加载 checkpoint + 构建模型
    # --------------------------
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]  # 从权重里读取训练配置

    # 用你的 TransformerLM 类构建模型
    model = TransformerLM(
        vocab_size=len(vocab_stoi),
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        block_size=cfg["model"]["block_size"]
    ).to(device)

    # 加载权重
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("✅ 成功加载 TransformerLM 模型！")

    # --------------------------
    # 5. 编码 Prompt
    # --------------------------
    try:
        prompt_tokens = [vocab_stoi[c] for c in args.prompt]
        print(f"✅ Prompt 编码完成: '{args.prompt}'")
    except KeyError as e:
        print(f"⚠️ 字符 '{e}' 不在词汇表，自动替换为空格")
        prompt_tokens = [vocab_stoi.get(c, vocab_stoi[' ']) for c in args.prompt]

    x = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # --------------------------
    # 6. 生成文本
    # --------------------------
    generated = prompt_tokens.copy()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
            generated.append(next_token.item())

    # --------------------------
    # 7. 解码输出
    # --------------------------
    generated_text = "".join([vocab_itos[t] for t in generated])
    print("\n" + "="*60)
    print("🎯 Transformer 生成结果:")
    print("="*60)
    print(generated_text)
    print("="*60)

if __name__ == "__main__":
    main()
