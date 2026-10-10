"""Strict artifact I/O and source/data freeze validation."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.metadata, json, platform, subprocess
import numpy as np

def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items() if k != "continuous_route"}
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
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root, *args):
    result = subprocess.run(["git", "-C", str(Path(root).resolve()), *args],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise ValueError(f"Git command failed ({args[0]}): {result.stderr.strip()}")
    return result.stdout.strip()


def git_state(root, require_clean=True):
    """Ignored result/cache files do not make a formal source checkout dirty."""
    root = Path(root).resolve()
    git_root = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if git_root != root:
        raise ValueError("The experiment root must be the repository root")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if require_clean and status:
        raise ValueError("Formal experiment requires a clean Git worktree")
    return {"code_commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "status": status}


def source_hashes(root):
    root = Path(root)
    paths = [p for package in ("f1ers_v7",)
             for p in (root / package).rglob("*.py")]
    paths += [root / name for name in ("run.ps1", "pyproject.toml", "requirements-lock.txt")
              if (root / name).is_file()]
    paths += list((root / "config").rglob("*.json"))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def local_data_hashes(root):
    """Hash local history only; no FP2 raw file is opened by provenance capture."""
    root = Path(root)
    paths = [p for prefix in ("data/raw/tum", "data/raw/openf1", "data/processed", "data/v3/local")
             for p in (root / prefix).rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    if (root / "data/v4/protocol_lock.json").is_file():
        paths.append(root / "data/v4/protocol_lock.json")
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def dependency_versions():
    import scipy
    import numba
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "numba": numba.__version__}


def capture_provenance(root, require_clean=True):
    root = Path(root).resolve()
    return {"schema_version": 4, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            **git_state(root, require_clean=require_clean), "source_sha256": source_hashes(root),
            "data_sha256": local_data_hashes(root), "dependencies": dependency_versions(),
            "dependency_lock_sha256": sha256(root / "requirements-lock.txt"),
            "protocol_sha256": sha256(root / "config/data_protocol.json")}


def check_freeze(root, freeze):
    root = Path(root).resolve()
    if isinstance(freeze, (str, Path)):
        freeze = read_json(freeze)
    if freeze.get("schema_version") != 4 or freeze.get("dirty"):
        raise ValueError("Invalid formal v4 freeze")
    for category in ("source_sha256", "data_sha256", "calculation_sha256", "fp2_sha256"):
        for relative, digest in freeze.get(category, {}).items():
            source = (root / relative).resolve()
            if not source.is_relative_to(root) or not source.is_file() or sha256(source) != digest:
                raise ValueError(f"Frozen artifact changed: {relative}")
    if source_hashes(root) != freeze.get("source_sha256"):
        raise ValueError("Executable source inventory changed after freeze")
    if sha256(root / "config/data_protocol.json") != freeze["protocol_sha256"]:
        raise ValueError("Frozen v4 protocol changed")
    _git(root, "cat-file", "-e", freeze["code_commit"] + "^{commit}")
    return True
