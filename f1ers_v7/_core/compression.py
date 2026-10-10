"""Strict compressed JSON serialization; extracted unchanged."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gzip
import json
from pathlib import Path
import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()
                if k != "continuous_route"}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return jsonable(asdict(value))
    return value


def write_compressed(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as stream:
        json.dump(jsonable(value), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
