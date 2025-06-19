# Documentation

This project implements the causal-consistency neural network described in
`Prompt.txt`. The documentation is organised into several sections covering
training, evaluation and configuration. The API reference is generated
automatically via MkDocs.

For a guided introduction see
the [training guide](training.md), the [metrics documentation](metrics.md) and
the [configuration reference](configuration.md).

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
- `WgivenX` models instruments `W` conditional on `X`.

These modules can be swapped or extended through the configuration system.

## Data generation
Synthetic datasets following the $X \to Y \to Z$ structure can be generated with
`examples/scripts/generate_synth.py`. An alternative generator with an
instrumental variable `W → Y` is also provided and activated via the
`instrumental` flag in the data configuration. Both are used throughout the
examples and tests.

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
`eval.py` loads a saved model and reports causal metrics. The script currently
computes the average treatment effect (ATE) by drawing counterfactual `Z`
predictions for both treatment arms and compares them. It also returns the
average log likelihood of observed `Z` values under the model.

```bash
python src/eval.py --config configs/train_synth.yaml --model-path run/model.pt
```

The `metrics.py` module exposes reusable helpers for ATE and log-likelihood
calculations.

## Serving
`serve.py` provides a minimal API for inference with a trained model. Functions
`predict_z`, `counterfactual_z` and `impute_y` wrap the network's heads for
easy integration in production services.

A lightweight FastAPI application is available in
[`fastapi_app.py`](../src/causal_consistency_nn/fastapi_app.py). Use
`create_app(model)` in your own code or run the module directly to expose the
three inference endpoints:

```bash
python -m causal_consistency_nn.fastapi_app --model-path run/model.pt --config examples/scripts/train_config.yaml
```

## Docker workflow
Both CPU and CUDA images can be built from the provided `Dockerfile`. Use
`docker compose` to orchestrate common tasks:

```bash
docker compose build
# Train using the current configuration
docker compose run train

# Launch the inference server
docker compose run --service-ports serve
```

Set `DEVICE=cuda` when building if GPUs are available.
