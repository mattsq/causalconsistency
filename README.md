# Causal Consistency NN

This repository provides an implementation of the causal-consistency neural network described in `Prompt.txt`.

## Quick start

```bash
poetry install
poetry run python src/train.py
```

The model enforces the factorisation $X \to Y \to Z$ by sharing a backbone encoder and three output heads. See the documentation in `docs/` for details.
