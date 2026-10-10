"""Package clean v5 history and offline artifacts without opening FP2 arrays."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

TAG = "v5-0.5.0"
LEGACY_STATUS = "data/v3/fp2/status.json"
LEGACY_STATUS_SHA256 = "d2ff18b420210fd8512cd282838da4e93bac09d2ccf89464a55688670fcc2f16"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          check=True).stdout.decode("utf-8").strip()


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if git(root, "status", "--porcelain", "--untracked-files=all") or git(root, "remote"):
        raise ValueError("Delivery requires a clean local-only repository")
    head = git(root, "rev-parse", "HEAD")
    if git(root, "rev-parse", TAG + "^{commit}") != head:
        raise ValueError("Delivery tag must identify the result commit")
    experiment = json.loads((root / "results/austria_2026_v5/experiment.json").read_text(encoding="utf-8"))
    verification = json.loads((root / "results/austria_2026_v5/verification.json").read_text(encoding="utf-8"))
    if not verification.get("exact_local_replay_passed") or not verification.get("artifact_integrity_passed"):
        raise ValueError("Fixed-parameter verification must pass before delivery")
    if experiment["cross_session"].get("status") != "pending":
        raise ValueError("This offline release does not package FP2 final data")
    output.mkdir(parents=True, exist_ok=True)
    archive, bundle = output / "F1_ERS_Lab_v5.zip", output / "F1_ERS_Lab_v5.bundle"
    if archive.exists() or bundle.exists():
        raise ValueError("Published delivery files must not be overwritten")
    git(root, "bundle", "create", str(bundle), "--all")
    checked = subprocess.run(["git", "-C", str(root), "bundle", "verify", str(bundle)],
                             capture_output=True, text=True, check=True)
    names = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True,
                           check=True).stdout.decode("utf-8").split("\0")
    paths = {root / name for name in names if name}
    for prefix in ("vendor/wheels", "results/austria_2026_v4", "results/austria_2026_v5"):
        paths.update(p for p in (root / prefix).rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
    if any("fp2" in p.relative_to(root).parts and p.relative_to(root).as_posix() != LEGACY_STATUS for p in paths):
        raise ValueError("FP2 raw and processed arrays must remain excluded")
    if digest(root / LEGACY_STATUS) != LEGACY_STATUS_SHA256:
        raise ValueError("Frozen historical FP2 metadata changed")
    records = {p.relative_to(root).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
               for p in sorted(paths)}
    manifest = {"schema_version": 5, "project": "F1 ERS Lab v5", "tag": TAG,
                "code_commit": experiment["code_provenance"]["code_commit"], "result_commit": head,
                "baseline_commit": git(root, "rev-parse", "baseline-v3-0.3.0^{commit}"),
                "v4_tag_commit": git(root, "rev-parse", "v4-0.4.0^{commit}"),
                "remote": None, "git_bundle_sha256": digest(bundle), "files": records,
                "gates": experiment["gates"], "cross_session_status": "pending", "training_run": False,
                "scope": "Tracked source and historical data, offline Windows Python 3.12 wheels, full v4/v5 results. Git history is a separate bundle. No FP2 arrays."}
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in sorted(paths):
            stream.write(path, path.relative_to(root).as_posix())
        stream.writestr("DELIVERY-MANIFEST-V5.json", encoded)
    with zipfile.ZipFile(archive) as stream:
        for name, record in records.items():
            if hashlib.sha256(stream.read(name)).hexdigest() != record["sha256"]:
                raise ValueError("Archive member changed: " + name)
    result = {"schema_version": 5, "tag": TAG, "code_commit": manifest["code_commit"],
              "result_commit": head, "files": len(records), "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
              "zip": {"name": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size},
              "bundle": {"name": bundle.name, "sha256": digest(bundle), "bytes": bundle.stat().st_size},
              "bundle_verify_passed": True, "bundle_verify_output": checked.stdout + checked.stderr,
              "archive_members_verified": True, "gates": experiment["gates"],
              "independent_clone_verification": "Recorded separately after offline clone recovery"}
    (output / "F1_ERS_Lab_v5_delivery.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in (archive, bundle):
        (output / (path.name + ".sha256")).write_text(digest(path) + "  " + path.name + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.output)
