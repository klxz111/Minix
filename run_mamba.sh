#!/usr/bin/env bash
set -e

python -m src.data.prepare_tinyshakespeare
python -m src.train --config configs/mamba.yaml
python -m src.evaluate --ckpt outputs/mamba/best.pt
python -m src.benchmark --ckpt outputs/mamba/best.pt
python -m src.generate --ckpt outputs/mamba/best.pt --prompt "To be"