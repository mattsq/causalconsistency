from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib


try:
    __version__ = version("causal_consistency_nn")
except PackageNotFoundError:  # pragma: no cover - fallback for editable install
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    __version__ = data["tool"]["poetry"]["version"]

__all__ = ["__version__"]
