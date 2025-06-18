# Documentation

This project implements the causal-consistency neural network described in `Prompt.txt`. The codebase now exposes a YAML-based configuration system and a modular set of model components.

## Configuration system
Configuration files live in the `configs/` directory and are parsed with `pydantic` dataclasses. All hyperparameters, such as backbone size or loss weights, can be changed in the YAML file or overridden via CLI flags:

```bash
python src/train.py --config configs/train_synth.yaml optimizer.lr=1e-3
```

## Model components
The network is composed of a shared `Backbone` and three heads:

- `ZgivenXY` predicts post-treatment variables `Z` from `X` and `Y`.
- `YgivenXZ` estimates missing treatments from `X` and `Z`.
- `XgivenYZ` reconstructs `X` given `Y` and `Z`.

These modules can be swapped or extended through the configuration system.

## Data generation
Synthetic datasets following the $X \to Y \to Z$ structure can be generated with
`examples/scripts/generate_synth.py`. The script outputs a small CSV file that is
used throughout the examples and tests.

## Training workflow
`train.py` launches a semi-supervised training loop that mixes supervised losses
with an unsupervised term for rows missing `Y`. A run is configured via a YAML
file and optional CLI overrides:

```bash
python src/train.py --config examples/scripts/train_config.yaml --model-hidden-dim 16
```

Each run stores its checkpoint alongside the resolved configuration for full
reproducibility.

## Configuration reference
The configuration is split into three dataclasses:

- **ModelConfig** – `hidden_dim`, `num_layers`.
- **LossWeights** – `z_yx`, `y_xz`, `x_yz`, `unsup`.
- **TrainingConfig** – `batch_size`, `epochs`, `learning_rate`, `device`.

Fields may be overridden on the command line using flags such as
`--model-hidden-dim` or via environment variables like `MODEL__NUM_LAYERS=4`.

## Evaluation
After training, run `src/eval.py` to compute metrics on the saved checkpoint.
The current implementation simply prints a placeholder message but forms the
entry point for future evaluation utilities.
