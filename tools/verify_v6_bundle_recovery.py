"""Independently clone a fixed bundle and restore/verify a fixed v6 ZIP.

Prepared only; invoke after packaging and supplying the separately recorded
archive/bundle/delivery-record hashes. Does not install dependencies, train, or
open a test session. Runtime/tests and saved-policy replay are separate checks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

try:  # tools/verify_v6_bundle_recovery.py beside tools/restore_v6.py.
    from restore_v6 import digest, reject_links, restore, safe_relative
except ModuleNotFoundError:
    try:
        from tools.restore_v6 import digest, reject_links, restore, safe_relative
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "v6_restore_tool"))
        from restore_v6 import digest, reject_links, restore, safe_relative


def verify(*, bundle, bundle_sha256, archive, archive_sha256, delivery_record,
           delivery_record_sha256, clone_root, report):
    bundle, archive, delivery_record = map(lambda p: Path(p).resolve(), (bundle, archive, delivery_record))
    for path, expected in ((bundle, bundle_sha256), (archive, archive_sha256),
                           (delivery_record, delivery_record_sha256)):
        if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected.lower():
            raise ValueError("Separately recorded delivery SHA256 mismatch: " + str(path))
    delivery = json.loads(delivery_record.read_text(encoding="utf-8"))
    if delivery.get("schema_version") != 6 or delivery["zip"]["sha256"] != archive_sha256.lower() or delivery["bundle"]["sha256"] != bundle_sha256.lower():
        raise ValueError("Delivery record does not identify both supplied files")
    clone_root = Path(os.path.abspath(clone_root))
    report = Path(os.path.abspath(report))
    reject_links(clone_root.parent, clone_root)
    reject_links(report.parent, report)
    if clone_root.exists() or report.exists():
        raise ValueError("Independent clone/report paths must be new")
    # Prevalidate record paths before any clone or artifact writes.
    expected_members = delivery.get("files")
    if not isinstance(expected_members, dict) or not expected_members:
        raise ValueError("Delivery file inventory is missing")
    for name in expected_members:
        safe_relative(name)
    if len({name.casefold() for name in expected_members}) != len(expected_members):
        raise ValueError("Duplicate/case-colliding delivery paths")
    clone_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--branch", delivery["tag"], str(bundle), str(clone_root)],
                   capture_output=True, check=True)
    head = subprocess.run(["git", "-C", str(clone_root), "rev-parse", "HEAD"],
                          capture_output=True, check=True).stdout.decode().strip()
    if head != delivery["result_commit"]:
        raise ValueError("Independent clone checked out the wrong result commit")
    restored = restore(clone_root, archive, archive_sha256)
    members = {row["path"]: row for row in restored["member_manifest"] if row["sha256"] is not None}
    if members.keys() != expected_members.keys():
        raise ValueError("Recovered ZIP member list differs from delivery record")
    for name, expected in expected_members.items():
        target = clone_root.joinpath(*safe_relative(name).parts)
        reject_links(clone_root, target)
        if (not target.is_file() or digest(target) != expected["sha256"]
                or target.stat().st_size != expected["bytes"] or members[name]["sha256"] != expected["sha256"]):
            raise ValueError("Independent recovered bytes mismatch: " + name)
    repeated = restore(clone_root, archive, archive_sha256)
    if repeated["restored"]:
        raise ValueError("Repeated restoration was not idempotent")
    result = {"schema_version": 6, "independent_bundle_recovery_verified": True,
              "result_commit": head, "code_commit": delivery["code_commit"],
              "bundle_sha256": bundle_sha256.lower(), "archive_sha256": archive_sha256.lower(),
              "delivery_record_sha256": delivery_record_sha256.lower(),
              "additional_experiments": delivery.get("additional_experiments", []),
              "clone_root": str(clone_root), "tracked_sources_verified": len(restored["verified_sources"]),
              "restored_artifact_count": len(restored["restored"]), "all_files_verified": len(expected_members),
              "repeat_restore_idempotent": True, "source_overwritten": False,
              "runtime_installation_tested": False, "computational_replay_tested": False,
              "browser_validation_tested": False, "FP2_accessed": False,
              "verification_tool_sha256": digest(Path(__file__))}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "bundle-sha256", "archive", "archive-sha256", "delivery-record",
                 "delivery-record-sha256", "clone-root", "report"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
