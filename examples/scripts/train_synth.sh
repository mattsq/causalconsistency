#!/usr/bin/env bash
# Example training invocation using a YAML configuration file
CONFIG_FILE=${1:-examples/scripts/train_config.yaml}
poetry run python src/train.py --config "$CONFIG_FILE"
