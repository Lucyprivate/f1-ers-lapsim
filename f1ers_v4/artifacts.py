"""Git-aware experiment provenance and restricted restoration of frozen artifacts."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import zipfile

import numpy as np

BASELINE_ARCHIVE_SHA256 = "a464e90c6bcaac167bed05a80da8353552e4c625a830b333ca02151e065e6da0"
BASELINE_ARCHIVE_PREFIX = "f1-ers-lab-v3"


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
    paths = [p for package in ("f1ers", "f1ers_v3", "f1ers_v4")
             for p in (root / package).rglob("*.py")]
    paths += [root / name for name in ("run.ps1", "pyproject.toml", "requirements-lock.txt")
              if (root / name).is_file()]
    paths += list((root / "config").glob("*.json"))
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
            "protocol_sha256": sha256(root / "config/protocol_austria_2026_v4.json")}


def freeze_configuration(root, output, vehicle, candidate="spline", stage="local", *,
                         provenance=None, require_clean=True, **extra):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("A frozen configuration must be inside the repository")
    if stage not in ("local", "cross_session") or candidate != "spline":
        raise ValueError("v4 uses one fixed spline model and two freeze stages")
    provenance = provenance or extra.pop("code_provenance", None) or capture_provenance(root, require_clean=require_clean)
    if provenance.get("dirty"):
        raise ValueError("A formal freeze cannot use a dirty source checkout")
    vehicle_value = vehicle.to_dict() if hasattr(vehicle, "to_dict") else jsonable(vehicle)
    value = {**provenance, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
             "stage": stage, "candidate": candidate, "vehicle": vehicle_value,
             "calculation_sha256": {p.relative_to(root).as_posix(): sha256(p)
                 for p in sorted(output.rglob("*")) if p.is_file() and (p.name in
                 ("calibration.json", "screening.json", "mesh.json", "numerical_audit.json",
                  "mass_sensitivity.json", "condition_adaptation.json", "trajectories.json.gz",
                  "numerical_failures.json.gz", "adaptation_trajectories.json.gz")
                  or ("failures" in p.relative_to(output).parts and p.name.endswith(".json.gz")))}, **extra}
    if stage == "cross_session":
        fp2 = root / "data/v4/fp2"
        value["fp2_sha256"] = {p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(fp2.rglob("*.json")) if not p.name.endswith(".tmp")}
    path = output / ("frozen_configuration_cross_session.json" if stage == "cross_session"
                     else "frozen_configuration.json")
    write_json(path, value)
    return path, read_json(path)


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
    if sha256(root / "config/protocol_austria_2026_v4.json") != freeze["protocol_sha256"]:
        raise ValueError("Frozen v4 protocol changed")
    _git(root, "cat-file", "-e", freeze["code_commit"] + "^{commit}")
    return True


def restore_baseline_artifacts(root, archive, *, expected_sha256=BASELINE_ARCHIVE_SHA256):
    """Restore ignored v3 results/wheels only after validating the entire ZIP.

    Existing identical files are idempotent. A differing existing file, a Git-
    tracked destination, an alias, or a traversal is rejected before any write.
    The CLI uses the fixed published checksum; checksum injection is for tests.
    """
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if sha256(archive) != expected_sha256:
        raise ValueError("Baseline archive SHA256 mismatch")
    tracked = set(_git(root, "ls-files", "-z").split("\0"))
    selected, seen, identical, skipped_tracked = [], set(), [], []
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            # ZipInfo normalizes Windows backslashes on construction; inspect
            # the original central-directory name before that normalization.
            name = info.orig_filename
            path = PurePosixPath(name)
            if ("\\" in name or "\0" in name or path.is_absolute() or any(part in ("", ".", "..") or ":" in part
                for part in path.parts) or not path.parts or path.parts[0] != BASELINE_ARCHIVE_PREFIX):
                raise ValueError(f"Unsafe archive member: {name}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Archive links are forbidden: {name}")
            if info.is_dir():
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative.casefold() in seen:
                raise ValueError(f"Duplicate archive destination: {relative}")
            seen.add(relative.casefold())
            if not (relative.startswith("results/") or relative.startswith("vendor/wheels/")):
                continue
            destination = root.joinpath(*PurePosixPath(relative).parts)
            if not destination.resolve().is_relative_to(root):
                raise ValueError(f"Archive destination escapes repository: {relative}")
            for parent in (destination, *destination.parents):
                if parent == root:
                    break
                if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
                    raise ValueError(f"Archive destination follows a link: {relative}")
            payload = bundle.read(info)
            digest = hashlib.sha256(payload).hexdigest()
            if relative in tracked:
                if destination.is_file() and sha256(destination) == digest:
                    skipped_tracked.append(relative)
                    continue
                raise ValueError(f"Refusing to overwrite a Git-tracked destination: {relative}")
            ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", relative],
                                     capture_output=True).returncode == 0
            if not ignored:
                raise ValueError(f"Restore destination must be Git-ignored: {relative}")
            if destination.exists():
                if not destination.is_file() or sha256(destination) != digest:
                    raise ValueError(f"Refusing to overwrite existing artifact: {relative}")
                identical.append(relative)
            else:
                selected.append((destination, payload, relative))
    for destination, payload, _ in selected:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
    return {"archive_sha256": expected_sha256, "restored": [p for _, _, p in selected],
            "already_identical": identical, "tracked_identical_unchanged": skipped_tracked,
            "source_files_overwritten": False}
