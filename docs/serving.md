# Serving with Torch-Serve

This project can be deployed using [Torch-Serve](https://pytorch.org/serve/). After training you will have a `model.pt` checkpoint and the `config.yaml` used during training. These files are required by the custom handler.

```bash
# Package the model
torch-model-archiver \
    --model-name causal \
    --version 1.0 \
    --serialized-file <run_dir>/model.pt \
    --handler src/causal_consistency_nn/torchserve_handler.py \
    --extra-files <run_dir>/config.yaml

mkdir model_store
mv causal.mar model_store/

# Launch the server
torchserve --start --ncs --model-store model_store --models causal=causal.mar
```

Requests are sent to the `predictions` endpoint. The payload must include an `action` field selecting one of the inference helpers:

- `predict_z` – return `E[Z|X,Y]`
- `counterfactual_z` – predict `Z` under an alternate treatment
- `impute_y` – posterior probabilities of `Y`

Example request using `curl`:

```bash
curl -X POST http://localhost:8080/predictions/causal \
    -d '{"action":"impute_y","x": [[0.2]], "z": [[1.0]]}'
```


