"""Portable JSON artifacts and executable-source provenance."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
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


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes(root):
    root = Path(root)
    paths = [p for package in ("f1ers", "f1ers_v3")
             for p in (root / package).rglob("*.py")]
    paths += [root / name for name in ("run.ps1", "pyproject.toml", "requirements-lock.txt")]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def freeze_configuration(root, output, vehicle, candidate, stage, **extra):
    import platform
    import scipy
    import numba
    root, output = Path(root), Path(output)
    protocol = root / "config/protocol_austria_2026_v3.json"
    records = {p.relative_to(root).as_posix(): sha256(p)
               for p in sorted((root / "data/v3").rglob("*"))
               if p.is_file() and p.name != "test_access.json"
               and (stage == "cross_session" or "fp2" not in p.relative_to(root / "data/v3").parts)
               and not p.name.endswith(".tmp")}
    lock = root / "data/v3/protocol_lock.json"
    if lock.exists():
        for entry in read_json(lock).get("frozen_inputs", []):
            records[entry["path"]] = sha256(root / entry["path"])
    for inherited in ("data/processed/track.npz",):
        if (root / inherited).is_file():
            records[inherited] = sha256(root / inherited)
    value = {"schema_version": 3, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
             "stage": stage, "candidate": candidate, "vehicle": vehicle.to_dict(),
             "source_sha256": source_hashes(root), "data_sha256": records,
             "calculation_sha256": {p.relative_to(root).as_posix(): sha256(p)
                 for p in sorted(output.rglob("*.json"))
                 if p.name in ("calibration.json", "trajectories.json", "mesh.json",
                               "mass_sensitivity.json", "condition_adaptation.json")},
             "protocol_sha256": sha256(protocol),
             "dependencies": {"python": platform.python_version(),
                 "numpy": np.__version__, "scipy": scipy.__version__, "numba": numba.__version__},
             **extra}
    path = output / ("frozen_configuration_cross_session.json"
                     if stage == "cross_session" else "frozen_configuration.json")
    write_json(path, value)
    return path, value


def check_freeze(root, freeze):
    root = Path(root)
    for category in ("source_sha256", "data_sha256", "calculation_sha256"):
        for relative, digest in freeze.get(category, {}).items():
            if not (root / relative).is_file() or sha256(root / relative) != digest:
                raise ValueError(f"Frozen artifact changed: {relative}")
    if sha256(root / "config/protocol_austria_2026_v3.json") != freeze["protocol_sha256"]:
        raise ValueError("Frozen v3 protocol changed")

