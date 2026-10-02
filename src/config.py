"""Paths, config loading and small IO helpers shared by every script."""
import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def best_arch_config(cfg, arch):
    """Architecture block from the config, overridden by the validation-tuned choice if it exists."""
    block = dict(cfg[arch])
    best = RESULTS / "rq1" / "best_config.json"
    if best.exists():
        block.update(read_json(best)[arch])
    return block


def output_max_len(cfg):
    """Response max length; the timing step may lower it to respect the compute budget."""
    timing = RESULTS / "timing.json"
    if timing.exists():
        return read_json(timing)["max_len"]
    return cfg["lengths"]["output"]


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_jsonable)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
