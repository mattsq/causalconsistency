# Training Usage

This section covers how to launch a training run using the provided command line interface.

## Basic invocation

The simplest way to start training on the synthetic data generator is:

```bash
python src/causal_consistency_nn/train.py --config examples/scripts/train_config.yaml
```

To train using Pyro's stochastic variational inference instead of the plain
PyTorch loop add the `--use-pyro` flag:

```bash
python src/causal_consistency_nn/train.py --config examples/scripts/train_config.yaml --use-pyro
```

To run the same experiment with PyTorch Lightning instead use `--use-lightning`:

```bash
python src/causal_consistency_nn/train.py --config examples/scripts/train_config.yaml --use-lightning
```

The script reads the YAML file and constructs the `Settings` dataclass which groups
all configuration sections. A run directory is created under the current working
folder to store the checkpoint and the resolved configuration.

## Adjusting hyperparameters

Command line flags override values from the YAML file. For example, to increase
the hidden dimension and enable a small amount of unsupervised loss you can run:

```bash
python src/causal_consistency_nn/train.py --config examples/scripts/train_config.yaml \
    --model-hidden-dim 128 --loss-unsup 0.1
```

Environment variables may also override nested fields using the `SECTION__FIELD`
convention, e.g. `TRAIN__EPOCHS=20`.

## Checkpoints and logging

After each epoch the model state is written to `<run_dir>/model.pt`. Logs are
printed to stdout and can be captured with any standard tool such as
`tee` or `wandb` integration if desired.

## Soft-EM option

When labels for `Y` are missing the training loop can perform soft updates.
Predicted class probabilities are used instead of hard argmaxes and an entropy
regulariser scaled by `tau` encourages exploration. The unsupervised objective
is weighted by `beta` and includes the entropy term:

```math
L_{unsup} = \beta \mathbb{E}_{p(y|x,z)}[L_z + L_x] - \tau H[p(y|x,z)].
```

Increasing `tau` yields smoother pseudo-labels while decreasing it approaches
standard EM.
