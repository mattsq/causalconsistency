# Training with Pyro SVI

`causal_consistency_nn` supports an alternative optimisation route based on
[Pyro](https://pyro.ai) and its stochastic variational inference (SVI) API.
While the default trainer implements a simple EM loop in plain PyTorch, SVI
allows more flexible probabilistic modelling and provides automatic
optimisation of variational parameters.

The Pyro variant uses a `PyroConsistencyModel` and the `train_svi` function
from `model/pyro_svi.py`. It mirrors the behaviour of the EM trainer but runs
the updates inside Pyro's tracing machinery and optimises an ELBO.

## When to use it

- **Richer guides** – the default guide is an `AutoNormal` which can be easily
  replaced with a custom variational family for more expressive posteriors.
- **Probabilistic programming features** – SVI integrates with Pyro's plate
  notation, subsampling and other conveniences that simplify experimentation.
- **Uncertainty calibration** – variational inference keeps track of parameter
  uncertainty which can be useful when propagating counterfactual queries.

For quick experiments or when Pyro is not installed the standard EM trainer is
sufficient. SVI comes with a small runtime overhead but may produce better
uncertainty estimates on complex datasets.

## CLI example

Enable SVI training with the `--use-pyro` flag:

```bash
python src/causal_consistency_nn/train.py \
    --config examples/scripts/train_config.yaml \
    --use-pyro --loss-unsup 0.1 --train-epochs 20
```

All regular hyperparameters apply. The learning rate and number of epochs can be
controlled via the configuration or command line just like with the standard
trainer. The `SVIConfig` dataclass inherits from `EMConfig` so options like
`beta`, `tau` and `pretrain_epochs` are available as well.

## Customising the guide

By default `train_svi` builds an `AutoNormal` guide over all model parameters:

```python
from pyro.infer.autoguide import AutoNormal

guide = AutoNormal(model)
```

To experiment with a different guide you can replicate `train_svi` in your own
script and swap the guide instance. This can be helpful when certain parameters
need structured approximations or when using normalising flows.

```python
from pyro.infer.autoguide import AutoDiagonalNormal
from causal_consistency_nn.model import train_svi, SVIConfig

# custom_guide could also wrap AutoGuideList etc.
custom_guide = AutoDiagonalNormal(model)
train_svi(model, sup_loader, unsup_loader, SVIConfig(), guide=custom_guide)
```

## Tips

- Start from a few epochs of supervised-only training by setting
  `pretrain_epochs` in the configuration. This stabilises the initial variational
  parameters when many labels are missing.
- Monitor the ELBO printed during training; it should increase (become less
  negative) over time.
- GPU acceleration works as usual – pass `--device cuda` or set
  `TRAIN__DEVICE=cuda`.

SVI is fully optional but provides a powerful alternative when probabilistic
programming features are desired.
