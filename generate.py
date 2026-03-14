import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.models import build_model
from src.data import build_vocab

def main():
    parser = argparse.ArgumentParser(description="Transformer 文本生成脚本")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重文件路径 (best.pt)")
    parser.add_argument("--prompt", type=str, required=True, help="输入提示文本")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="生成最大新token数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度系数")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K 采样")
    args = parser.parse_args()

    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")

    # 2. 加载配置
    ckpt_path = Path(args.ckpt)
    cfg_path = ckpt_path.parent / "config.pt"
    if not cfg_path.exists():
        print(f"❌ 错误: 未找到配置文件 {cfg_path}")
        return
    cfg = torch.load(cfg_path, map_location=device)
    print(f"✅ 加载配置成功")

    # 3. 构建词汇表
    vocab = build_vocab(cfg["data"]["root"])
    word2idx, idx2word = vocab
    vocab_size = len(word2idx)
    vocab = {"word2idx": word2idx, "idx2word": idx2word, "vocab_size": vocab_size}
    print(f"✅ 词汇表构建成功，大小: {vocab_size}")

    # 4. 构建模型
    model = build_model(cfg["model"], vocab["vocab_size"], cfg["data"]["block_size"]).to(device)
    print(f"✅ 模型构建成功")

    # 5. 加载权重（修复：兼容两种保存格式）
    try:
        checkpoint = torch.load(args.ckpt, map_location=device)
        # 情况1：权重是字典格式，包含 "model" 键
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        # 情况2：权重直接是模型参数字典（训练时直接保存 model.state_dict()）
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    model.eval()

    # 6. 编码 Prompt
    try:
        prompt_tokens = [vocab["word2idx"].get(c, vocab["word2idx"].get('<unk>', 0)) for c in args.prompt]
        print(f"✅ Prompt 编码完成")
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        return

    # 7. 生成逻辑
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = prompt_tokens.copy()
    
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            x_cond = x[:, -cfg["model"]["block_size"]:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / args.temperature
            
            # Top-K 采样
            if args.top_k > 0:
                top_k = min(args.top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
            generated.append(next_token.item())

    # 8. 解码输出
    generated_text = "".join([vocab["idx2word"].get(t, "[UNK]") for t in generated])
    print("\n" + "="*60)
    print("📝 生成结果:")
    print("="*60)
    print(generated_text)

if __name__ == "__main__":
    main()
