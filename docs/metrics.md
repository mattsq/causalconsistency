# Evaluation Metrics

The project provides helper utilities to evaluate trained models. The main entry
point is `eval.py` which reports both log-likelihood and causal metrics.

## Running the evaluator

```bash
python src/causal_consistency_nn/eval.py --config examples/scripts/train_config.yaml \
    --model-path <run_dir>/model.pt
```

The script loads the saved `Settings` from the YAML file, restores the model and
runs evaluation on a freshly generated synthetic dataset. The resulting metrics
are printed in YAML format for easy consumption.

## Available metrics

- **Log Likelihood** – Average negative log probability of observed `Z` values.
- **Average Treatment Effect (ATE)** – The difference in mean predicted `Z`
  between treatment and control groups.

Additional metrics can be implemented by extending `metrics.py` and importing
those functions in `eval.py`.
