# Causal Consistency NN

This repository provides an implementation of the causal-consistency neural network described in `Prompt.txt`.

## Quick start

```bash
poetry install
poetry run python src/train.py
```

The model enforces the factorisation $X \to Y \to Z$ by sharing a backbone encoder and three output heads. See the documentation in `docs/` for details.

## Running tests

Use `pytest` together with `pytest-cov` to measure coverage locally. Coverage must remain above 90 %:

```bash
pip install -e . pytest pytest-cov
pytest --cov=src --cov=tests --cov-report=term --cov-fail-under=90
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
