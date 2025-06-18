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

## Training loop
`train.py` runs a semi-supervised training loop that mixes supervised losses with an unsupervised term for rows missing `Y`. Parameters are updated with gradient descent and each run saves its checkpoint together with the YAML config for full reproducibility.
