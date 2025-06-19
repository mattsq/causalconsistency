from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:  # pragma: no cover - Python <3.11 requires optional dependency
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback to tomli on 3.10
    import tomli as tomllib


try:
    __version__ = version("causal_consistency_nn")
except PackageNotFoundError:  # pragma: no cover - fallback for editable install
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject.open("r", encoding="utf-8") as f:
        data = tomllib.loads(f.read())
    __version__ = data["tool"]["poetry"]["version"]

__all__ = ["__version__"]
