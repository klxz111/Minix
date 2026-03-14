#!/usr/bin/env bash
set -e

bash scripts/run_lstm.sh
bash scripts/run_transformer.sh
bash scripts/run_mamba.sh