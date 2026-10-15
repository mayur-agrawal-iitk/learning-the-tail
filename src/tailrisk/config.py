"""Configuration loading and project paths."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CONFIG_DIR = ROOT / "configs"


def load_config(name_or_path: str = "main") -> dict:
    """Load a YAML config by name (``configs/<name>.yaml``) or explicit path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["alphas"] = [float(a) for a in cfg["alphas"]]
    return cfg


def results_dir(cfg: dict) -> Path:
    """Directory holding forecasts, tables and figures for one experiment."""
    d = ROOT / "results" / cfg["name"]
    for sub in ("forecasts", "tables", "figures"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d
