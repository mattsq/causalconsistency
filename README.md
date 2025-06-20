# Causal Consistency NN

This repository provides an implementation of the causal-consistency neural network described in `Prompt.txt`.

## Quick start

```bash
poetry install  # installs torch, pyro, pydantic, pydantic-settings, PyYAML, fastapi and uvicorn
poetry run python src/train.py        # plain PyTorch EM loop
poetry run python src/train.py --use-pyro  # train with Pyro SVI
poetry run python src/train.py --use-lightning  # train with PyTorch Lightning
```

Lightning training requires the optional `pytorch-lightning` package:

```bash
pip install pytorch-lightning
```

If you prefer using `pip` directly, install the dependencies first:

```bash
pip install torch pyro-ppl pydantic pydantic-settings PyYAML fastapi uvicorn
pip install -e .
```

### Reproducing the environment

The repository provides a fully pinned `conda-lock.yml`. Create the environment
with micromamba or conda:

```bash
micromamba create -n cc --file conda-lock.yml
# or: conda env create -f conda-lock.yml
micromamba activate cc
```

You can then run training or the tests inside this environment.

The model enforces the factorisation $X \to Y \to Z$ by sharing a backbone encoder and three output heads. See the documentation in `docs/` for details.
For an overview of the causal assumptions, fairness considerations and robustness tests see the [model card](docs/model_card.md).

## Running tests

Use `pytest` together with `pytest-cov` to measure coverage locally. Coverage must remain above 90 %:

```bash
pip install -e . pytest pytest-cov
pytest --cov=src --cov=tests --cov-report=term --cov-fail-under=90
```

## Pre-commit hooks

Format and lint code automatically using [pre-commit](https://pre-commit.com/).
Install the hooks once and they will run `black` and `ruff` on staged files:

```bash
pip install pre-commit
pre-commit install
```

You can run all hooks manually with:

```bash
pre-commit run --all-files
```

## Containerised workflow

The project ships with a `Dockerfile` and `docker-compose.yml` to simplify
training and serving. Build the CPU image and run a training job with:

```bash
docker compose build
docker compose run train
```

To use a CUDA image set `DEVICE=cuda` before building:

```bash
DEVICE=cuda docker compose build
DEVICE=cuda docker compose run train
```

A simple inference server can be launched with:

```bash
docker compose run --service-ports serve
```

The compose file mounts the repository in `/app` so outputs are written back to
your local filesystem.

## Package version

The installed version of the library is exposed as `causal_consistency_nn.__version__`.
It is read from `pyproject.toml` so it matches the package metadata:

```python
import causal_consistency_nn
print(causal_consistency_nn.__version__)
```

