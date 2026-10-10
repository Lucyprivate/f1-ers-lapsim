"""Input errors, JSON files and SHA-256; extracted unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DataUnavailable(RuntimeError):
    """Real data or a provenance boundary is unavailable; no invented replacement."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
