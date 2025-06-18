#!/usr/bin/env bash
# Example training invocation using a YAML configuration file
CONFIG_FILE=${1:-examples/scripts/train_config.yaml}
OUT_DIR=${2:-examples/scripts/out}
poetry run python -m causal_consistency_nn.train \
  --config "$CONFIG_FILE" \
  --output-dir "$OUT_DIR"
