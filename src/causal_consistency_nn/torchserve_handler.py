from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import Settings
from .eval import load_model
from .serve import counterfactual_z, impute_y, predict_z


_model = None


def _get_model(context: Any) -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model
    model_dir = Path(getattr(context, "system_properties", {}).get("model_dir", "."))
    model_path = model_dir / "model.pt"
    cfg_path = model_dir / "config.yaml"
    settings = Settings.from_yaml(cfg_path)
    _model = load_model(model_path, settings)
    return _model


def _parse(data: Any) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    if isinstance(data, str):
        return json.loads(data)
    return data


def handle(data: list[dict[str, Any]], context: Any) -> Any:
    """Entry point used by Torch-Serve."""
    if not data:
        return None
    payload = _parse(data[0].get("body", data[0].get("data", data[0])))
    model = _get_model(context)

    action = payload.get("action")
    if action == "predict_z":
        x = torch.tensor(payload["x"], dtype=torch.float32)
        y = torch.tensor(payload["y"], dtype=torch.long)
        out = predict_z(model, x, y)
        return out.tolist()
    if action == "counterfactual_z":
        x = torch.tensor(payload["x"], dtype=torch.float32)
        y_prime = torch.tensor(payload["y_prime"], dtype=torch.long)
        out = counterfactual_z(model, x, y_prime)
        return out.tolist()
    if action == "impute_y":
        x = torch.tensor(payload["x"], dtype=torch.float32)
        z = torch.tensor(payload["z"], dtype=torch.float32)
        out = impute_y(model, x, z)
        return out.tolist()

    raise ValueError(f"Unknown action: {action}")
