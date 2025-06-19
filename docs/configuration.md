# Customising Configurations

All settings are managed by the `Settings` class in `config.py`. The default
values are reasonable for small experiments but most projects will want to tweak
them. Configuration can come from three sources:

1. A YAML file supplied via `--config`.
2. Environment variables using the `SECTION__FIELD` notation.
3. Command line flags which take highest priority.

Below is a minimal configuration file illustrating the available sections:

```yaml
model:
  hidden_dim: 64
  num_layers: 2
  w_dim: 1
loss:
  z_yx: 1.0
  y_xz: 1.0
  x_yz: 1.0
  w_x: 1.0
  unsup: 0.0
train:
  batch_size: 32
  epochs: 10
  learning_rate: 1e-3
data:
  n_samples: 1000
  noise_std: 0.1
  missing_y_prob: 0.0
  instrumental: false
  w_dim: 1
  w_y_strength: 1.0
```

To override a single value without editing the file you can pass something like
`--train-batch-size 64` or set `LOSS__UNSUP=0.5` in the environment. The
resolved configuration is printed at the start of each run so that experiments
are fully reproducible.
