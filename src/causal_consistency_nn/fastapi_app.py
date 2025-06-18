from __future__ import annotations

from pathlib import Path
import argparse

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from .eval import load_model
from .serve import counterfactual_z, impute_y, predict_z
from .config import Settings
from .train import ConsistencyModel


class PredictRequest(BaseModel):
    x: list[list[float]]
    y: list[int]


class CounterfactualRequest(BaseModel):
    x: list[list[float]]
    y_prime: list[int]


class ImputeRequest(BaseModel):
    x: list[list[float]]
    z: list[list[float]]


def create_app(model: ConsistencyModel) -> FastAPI:
    """Return a FastAPI app exposing the inference helpers."""

    app = FastAPI()
    app.state.model = model

    @app.post("/predict_z")
    def _predict(req: PredictRequest) -> dict[str, list[list[float]]]:
        x = torch.tensor(req.x, dtype=torch.float32)
        y = torch.tensor(req.y, dtype=torch.long)
        out = predict_z(app.state.model, x, y)
        return {"z": out.tolist()}

    @app.post("/counterfactual_z")
    def _counter(req: CounterfactualRequest) -> dict[str, list[list[float]]]:
        x = torch.tensor(req.x, dtype=torch.float32)
        y_prime = torch.tensor(req.y_prime, dtype=torch.long)
        out = counterfactual_z(app.state.model, x, y_prime)
        return {"z": out.tolist()}

    @app.post("/impute_y")
    def _impute(req: ImputeRequest) -> dict[str, list[list[float]]]:
        x = torch.tensor(req.x, dtype=torch.float32)
        z = torch.tensor(req.z, dtype=torch.float32)
        out = impute_y(app.state.model, x, z)
        return {"y_prob": out.tolist()}

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a trained model via FastAPI")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    settings = Settings.from_yaml(args.config)
    model = load_model(args.model_path, settings)
    app = create_app(model)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
