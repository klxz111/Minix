#!/usr/bin/env bash
set -e

python -m src.data.prepare_tinyshakespeare
python -m src.train --config configs/lstm.yaml
python -m src.evaluate --ckpt outputs/lstm/best.pt
python -m src.benchmark --ckpt outputs/lstm/best.pt
python -m src.generate --ckpt outputs/lstm/best.pt --prompt "To be"