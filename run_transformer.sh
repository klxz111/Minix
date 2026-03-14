#!/usr/bin/env bash
set -e

python -m src.data.prepare_tinyshakespeare
python -m src.train --config configs/transformer.yaml
python -m src.evaluate --ckpt outputs/transformer/best.pt
python -m src.benchmark --ckpt outputs/transformer/best.pt
python -m src.generate --ckpt outputs/transformer/best.pt --prompt "To be"