import json
import yaml
from pathlib import Path
import torch

# 1. 追加写入jsonl日志
def append_jsonl(data, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

# 2. 统计模型参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 3. 确保目录存在
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# 4. 加载yaml配置（别名load_config，兼容之前的导入）
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
load_config = load_yaml  # 别名

# 5. 保存json文件
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 6. 加载meta.json
def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 7. 设置随机种子
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 8. 获取AMP dtype
def get_amp_dtype(dtype_str):
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    else:
        return torch.float32

# 9. 获取autocast上下文
def get_autocast_context(device, amp_enabled, amp_dtype):
    if amp_enabled and device.startswith("cuda"):
        return torch.amp.autocast(device_type=device, dtype=amp_dtype)
    else:
        from contextlib import nullcontext
        return nullcontext()