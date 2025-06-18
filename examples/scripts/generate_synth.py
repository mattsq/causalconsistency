"""Generate and save a synthetic dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from causal_consistency_nn.config import SyntheticDataConfig
from causal_consistency_nn.data import generate_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--out", type=Path, default=Path("synthetic.pt"))
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--missing-y-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = SyntheticDataConfig(
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        missing_y_prob=args.missing_y_prob,
    )
    ds = generate_synthetic(cfg, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds.tensors, args.out)
    print(f"Saved dataset to {args.out}")


if __name__ == "__main__":
    main()
