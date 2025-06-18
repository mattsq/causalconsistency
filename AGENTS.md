# AGENTS.md

## Purpose
This repository will house `causal_consistency_nn`, a PyTorch implementation of the causal-consistency model described in `Prompt.txt`. The goal is a production-ready package that can model the joint distribution of pretreatment covariates **X**, treatment **Y** (possibly multi-class), and post-treatment covariates **Z**. The network will enforce the three conditionals

```
p_theta(Z|Y,X),
p_theta(Y|X,Z),
p_theta(X|Y,Z)
```

to be mutually consistent so that the implied joint distribution respects the causal direction **X → Y → Z**.

## Repository layout
The project should follow this scaffold:

```
causal_consistency_nn/
├── pyproject.toml            # Poetry/PEP‑621 with pinned torch & pyro
├── README.md                 # quick-start and theoretical recap
├── CONTRIBUTING.md           # coding style and DCO
├── LICENSE                   # MIT
├── docs/                     # MkDocs or Sphinx with auto‑generated API
├── src/
│   ├── data/                 # dataloaders and synthetic simulations
│   ├── model/                # backbone.py, heads.py, losses.py
│   ├── train.py              # Hydra-configurable entry point
│   ├── eval.py               # causal metrics and ablations
│   └── utils/                # logging, metrics, seeding utilities
├── tests/                    # pytest unit and integration tests
├── examples/
│   ├── notebook_intro.ipynb
│   └── scripts/
│        ├── generate_synth.py
│        └── train_synth.sh
└── .github/
    ├── workflows/ci.yml
    └── CODEOWNERS
```

## Coding conventions
* Use **typed dataclasses** for configuration objects (e.g. `@dataclass class HParams`).
* Hydra or `pydantic-settings` should allow architecture and loss weights to be adjusted via YAML or CLI flags.
* Keep losses modular so different research and production settings only adjust weightings or add new heads.
* Optionally integrate Torch Lightning for checkpointing and early stopping.
* Style checks:
  * Format with `black`.
  * Lint with `ruff`.
* Unit tests with **pytest** must achieve at least **90 %** coverage. CI should run tests on Python 3.10–3.12 and both CPU and CUDA builds.
* CI should also publish the package automatically: pushes to `main` go to TestPyPI; tags go to PyPI.
* Provide Docker images via Dockerfile and docker‑compose with pinned package versions.

## Model outline
Implementation should include a shared encoder (`Backbone`) and three heads:
* `ZgivenXY`: outputs a Gaussian or categorical distribution for `Z` given `X` and `Y`.
* `YgivenXZ`: predicts `Y` from `X` and `Z` using a categorical distribution.
* `XgivenYZ`: reconstructs `X` from `Y` and `Z` (e.g. Gaussian).
A semi‑supervised EM training loop will combine supervised loss terms
```
L_sup = λ1 L_ZYX + λ2 L_YXZ + λ3 L_XYZ
```
with an unsupervised variant for rows where `Y` is missing. The total objective is
```
L = L_sup + β L_unsup.
```
Pseudo‑labelling or soft EM should be used for the missing‑`Y` rows, optionally with an entropy regulariser `τ` to avoid over‑confident guesses. Starting with a few epochs of supervised‑only training can stabilise pseudo‑labels.

## Testing and validation
* Unit tests must verify tensor shapes, gradient flow, and that log‑probabilities are negative when expected.
* Semi‑supervised loops should show decreasing loss and (for EM) monotonic improvement on synthetic data.
* Causal metrics like Average Treatment Effect (ATE) should be estimated within the confidence interval on a known SCM.
* Include integration tests that simulate various missing‑`Y` mechanisms (MAR/MNAR) and ensure the semi‑supervised approach outperforms a supervised baseline on held‑out log likelihood and ATE RMSE.

## Deployment
* Saved artefacts consist of a checkpoint `.pt`, the config `.yaml`, and a `conda-lock.yml` describing the environment.
* Provide a serving module (`serve.py`) exposing functions:
  * `predict_z(x, y)` – draw samples or compute the mean of `Z | X=x, Y=y`.
  * `counterfactual_z(x, y_prime)` – estimate `Z` had the treatment been `y'`.
  * `impute_y(x, z)` – posterior over `Y` given `X` and `Z`.
* Support batch inference via Torch‑Serve or FastAPI.
* Create a model card documenting the DAG, assumptions, fairness considerations, and robustness checks.

## Future extensions
* Potential extensions include instrumental‑variable heads (`W → Y`), normalising flows for richer continuous variables, graph‑neural‑network backbones for structured `X`, and active learning for labelling the most informative rows.

## Summary for agents
When adding code to this repository:
1. Follow the directory structure above.
2. Ensure Python files pass `black` and `ruff`.
3. Maintain >90 % pytest coverage and include integration tests for semi‑supervised features.
4. Use dataclasses and Hydra/pydantic‑settings for configuration.
5. Document new components in the `docs/` site and update examples if behaviour changes.
6. Keep the causal assumptions and factorisations from `Prompt.txt` in mind: the three heads must be mutually consistent and encode **X → Y → Z**.

This AGENTS.md is the canonical set of instructions for maintainers and autonomous agents working on `causal_consistency_nn`.
