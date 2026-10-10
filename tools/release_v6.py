"""Package sealed v6 results, matching tracked source, and offline wheels.

This utility is prepared outside the repository and must only be invoked after
the runner has exited, the results inventory has been sealed, verification has
passed, and the reviewed result commit/tag has been created. Failed scientific
gates remain evidence and do not prevent packaging a diagnostic delivery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile

try:  # tools/release_v6.py beside tools/restore_v6.py after approved copying.
    from restore_v6 import (ZIP_PREFIX, LEGACY_STATUS, LEGACY_STATUS_SHA256, LEGACY_METADATA_SHA256,
                            allowed_artifact, digest, forbidden_arrays,
                            reject_links, safe_relative)
except ModuleNotFoundError:
    try:  # Also supports import tools.release_v6 from a repository test.
        from tools.restore_v6 import (ZIP_PREFIX, LEGACY_STATUS, LEGACY_STATUS_SHA256, LEGACY_METADATA_SHA256,
                                      allowed_artifact, digest, forbidden_arrays,
                                      reject_links, safe_relative)
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "v6_restore_tool"))
        from restore_v6 import (ZIP_PREFIX, LEGACY_STATUS, LEGACY_STATUS_SHA256, LEGACY_METADATA_SHA256,
                                allowed_artifact, digest, forbidden_arrays,
                                reject_links, safe_relative)

RESULTS = "results/austria_2026_v6"
EXTRA_ROOTS = (RESULTS, "vendor/wheels", "vendor/reporting-wheels")
EXCLUDED_COMPONENTS = {".git", ".venv", "venv", "__pycache__", ".cache", "node_modules"}
EXCLUDED_SUFFIXES = (".tmp", ".pyc", ".pyo", ".nbc", ".nbi")


def git(root, *args):
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if process.returncode:
        raise ValueError("Git check failed: " + process.stderr.decode(errors="replace"))
    return process.stdout


def unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key: " + key)
        value[key] = item
    return value


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_pairs)


def require_sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("A separately recorded full SHA256 is required for " + label)
    return value.lower()


def excluded(name):
    path = PurePosixPath(name)
    return bool(set(part.casefold() for part in path.parts) & EXCLUDED_COMPONENTS
                or path.name.casefold().endswith(EXCLUDED_SUFFIXES))


def regular_file(root, name):
    relative = safe_relative(name)
    target = root.joinpath(*relative.parts)
    reject_links(root, target)
    item = os.lstat(target)
    if not stat.S_ISREG(item.st_mode) or not target.resolve().is_relative_to(root):
        raise ValueError("Not an unlinked regular repository file: " + name)
    if forbidden_arrays(name):
        raise ValueError("FP2 raw/processed or late arrays must remain excluded: " + name)
    if name in LEGACY_METADATA_SHA256 and digest(target) != LEGACY_METADATA_SHA256[name]:
        raise ValueError("Frozen historical metadata changed")
    return target


def tracked_files(root):
    entries = {}
    for line in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not line:
            continue
        header, raw_name = line.split(b"\t", 1)
        mode, object_id, stage = header.decode("ascii").split()
        name = raw_name.decode("utf-8")
        if mode not in ("100644", "100755") or stage != "0":
            raise ValueError("Only regular, resolved Git index entries may be packaged: " + name)
        if excluded(name):
            raise ValueError("Environment/cache entry unexpectedly tracked: " + name)
        entries[name] = {"git_blob": object_id, "mode": mode}
    if not entries:
        raise ValueError("The matching Git source must already exist")
    if len({name.casefold() for name in entries}) != len(entries):
        raise ValueError("Case-colliding tracked source paths")
    return entries


def blob_digest(path, algorithm):
    value = hashlib.new(algorithm)
    size = path.stat().st_size
    value.update(("blob " + str(size)).encode("ascii") + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def collect(root, tracked):
    names = set(tracked)
    for prefix in EXTRA_ROOTS:
        base = root / prefix
        reject_links(root, base)
        if not base.exists():
            continue
        if not base.is_dir():
            raise ValueError("Ignored artifact root is not a directory: " + prefix)
        for current, directories, files in os.walk(base, followlinks=False):
            current = Path(current)
            # Inspect links/junctions before os.walk can descend into them.
            for name in directories:
                child = current / name
                reject_links(root, child)
                if not stat.S_ISDIR(os.lstat(child).st_mode):
                    raise ValueError("Non-directory artifact ancestor: " + str(child))
            directories[:] = [name for name in directories
                              if not excluded((current / name).relative_to(root).as_posix())]
            for basename in files:
                name = (current / basename).relative_to(root).as_posix()
                regular_file(root, name)  # Also reject unsafe/link members being excluded.
                if excluded(name):
                    continue
                if name not in tracked:
                    if not allowed_artifact(name):
                        raise ValueError("Unlisted ignored artifact: " + name)
                    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", name],
                                             capture_output=True)
                    if ignored.returncode:
                        raise ValueError("Allowed artifact is not Git ignored: " + name)
                names.add(name)
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Case-colliding package members")
    if any(any(parent.as_posix() in names for parent in PurePosixPath(name).parents
               if parent != PurePosixPath(".")) for name in names):
        raise ValueError("File/descendant package collision")
    return sorted(names)


def validate_seal(root, manifest_sha256, code_commit):
    manifest_path = regular_file(root, RESULTS + "/manifest.json")
    if digest(manifest_path) != require_sha(manifest_sha256, "sealed results manifest"):
        raise ValueError("Sealed results manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    experiment = read_json(regular_file(root, RESULTS + "/experiment.json"))
    verification = read_json(regular_file(root, RESULTS + "/verification.json"))
    if manifest.get("schema_version") != 6 or not isinstance(manifest.get("files"), dict):
        raise ValueError("A complete v6 results manifest is required")
    if manifest.get("code_commit") != code_commit or experiment.get("provenance", {}).get("code_commit") != code_commit:
        raise ValueError("The calculation code commit differs from the fixed delivery expectation")
    if experiment.get("new_test_opened") is not False or verification.get("new_test_opened") is not False:
        raise ValueError("This delivery must preserve the unopened FP2 protocol")
    if not all(verification.get(key) is True for key in
               ("passed", "selected_trajectory_exactly_reproduced", "calibration_and_development_reproduced",
                "all_training_and_failure_logs_hash_verified")):
        raise ValueError("Saved selected-policy reproduction/integrity verification has not passed")
    seen = set()
    for relative, expected in manifest["files"].items():
        safe_relative(relative)
        if relative == "manifest.json" or relative.casefold() in seen:
            raise ValueError("Recursive or duplicate result manifest entry: " + relative)
        seen.add(relative.casefold())
        target = regular_file(root, RESULTS + "/" + relative)
        if digest(target) != require_sha(expected, "manifest member " + relative):
            raise ValueError("Sealed calculation artifact changed: " + relative)
    for section in ("source_sha256", "data_sha256"):
        expected_files = experiment.get("provenance", {}).get(section)
        if not isinstance(expected_files, dict):
            raise ValueError("Missing frozen input inventory: " + section)
        for name, expected in expected_files.items():
            if digest(regular_file(root, name)) != require_sha(expected, "frozen input " + name):
                raise ValueError("Frozen input changed: " + name)
    return experiment, verification, manifest


def validate_r7(root, expected_sha256, head):
    """R7 has its own source freeze and inventory; never modify the v6 seal."""
    prefix = RESULTS + "/R7/"
    experiment_path = root / (prefix + "experiment.json")
    if not experiment_path.exists():
        if expected_sha256 is not None:
            raise ValueError("An R7 seal was supplied but the experiment is absent")
        directory = root / prefix
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("An R7 run has started but has not completed/sealed")
        return []
    expected_sha256 = require_sha(expected_sha256, "independent R7 results manifest")
    manifest_path = regular_file(root, prefix + "manifest.json")
    if digest(manifest_path) != expected_sha256:
        raise ValueError("Independent R7 results manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    experiment = read_json(regular_file(root, prefix + "experiment.json"))
    verification = read_json(regular_file(root, prefix + "verification.json"))
    provenance = experiment.get("provenance", {})
    source_commit = experiment.get("source_commit")
    if (manifest.get("schema_version") != 7 or experiment.get("schema_version") != 7
            or not isinstance(manifest.get("files"), dict)
            or not isinstance(source_commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_commit)
            or manifest.get("source_commit") != source_commit
            or provenance.get("code_commit") != source_commit
            or verification.get("source_commit") != source_commit):
        raise ValueError("The independent R7 source/inventory/verification identities disagree")
    git(root, "merge-base", "--is-ancestor", source_commit, head)
    if (verification.get("passed") is not True or verification.get("checkpoint_hash_passed") is not True
            or verification.get("source_hashes_passed") is not True
            or verification.get("dependencies_match") is not True
            or verification.get("FP2_accessed") is not False
            or experiment.get("FP2_accessed") is not False
            or not verification.get("manifest_inputs_checked")):
        raise ValueError("R7 saved verification must pass before delivery")
    exact_names = ("exact_selected_trace", "exact_selected_terminal", "exact_selected_proxy",
                   "exact_selected_calibration", "exact_selected_development", "exact_selected_actions")
    if any(verification.get(key) is not True for key in exact_names):
        raise ValueError("R7 exact saved-result verification has a failed field")
    seen = set()
    for name, expected in manifest["files"].items():
        safe_relative(name)
        if name == "manifest.json" or name.casefold() in seen:
            raise ValueError("Recursive or duplicate R7 manifest entry: " + name)
        seen.add(name.casefold())
        if digest(regular_file(root, prefix + name)) != require_sha(expected, "R7 member " + name):
            raise ValueError("Sealed R7 artifact changed: " + name)
    source_inventory = provenance.get("source_sha256")
    data_inventory = provenance.get("data_sha256")
    if not isinstance(source_inventory, dict) or not source_inventory or not isinstance(data_inventory, dict):
        raise ValueError("R7 frozen source/data inventories are missing")
    for label, inventory in (("source", source_inventory), ("data", data_inventory)):
        for name, expected in inventory.items():
            if digest(regular_file(root, name)) != require_sha(expected, "R7 frozen " + label + " " + name):
                raise ValueError("R7 frozen " + label + " changed: " + name)
    configuration_sha256 = require_sha(provenance.get("config_sha256"), "R7 frozen configuration")
    if (manifest.get("config_sha256") != configuration_sha256
            or experiment.get("config_sha256") != configuration_sha256):
        raise ValueError("R7 configuration identity differs from the inventory")
    config_path = experiment.get("config_path")
    frozen_protocol = experiment.get("frozen_protocol_path")
    if not isinstance(config_path, str) or not isinstance(frozen_protocol, str):
        raise ValueError("R7 original/frozen configuration paths are missing")
    if (digest(regular_file(root, config_path)) != configuration_sha256
            or digest(regular_file(root, prefix + safe_relative(frozen_protocol).as_posix())) != configuration_sha256):
        raise ValueError("R7 original/frozen configuration bytes differ")
    # The configuration must be anchored to actual frozen file bytes, rather
    # than two equal claims in JSON objects. Either inventory can contain it.
    config_anchors = [name for name, value in source_inventory.items() if value == configuration_sha256]
    config_anchors += [prefix + name for name, value in manifest["files"].items() if value == configuration_sha256]
    if not config_anchors or not all(digest(regular_file(root, name)) == configuration_sha256 for name in config_anchors):
        raise ValueError("R7 config SHA256 has no verified frozen file anchor")
    checkpoints = experiment.get("checkpoint_inventory")
    if not isinstance(checkpoints, dict) or manifest.get("checkpoint_inventory") != checkpoints:
        raise ValueError("R7 checkpoint freeze differs between experiment and manifest")
    for path_key, hash_key in (("checkpoint_relative_to_prior_run", "checkpoint_sha256"),
                               ("metadata_relative_to_prior_run", "metadata_sha256")):
        relative = checkpoints.get(path_key)
        if not isinstance(relative, str):
            raise ValueError("R7 checkpoint/metadata source path is missing")
        target = regular_file(root, RESULTS + "/" + safe_relative(relative).as_posix())
        if digest(target) != require_sha(checkpoints.get(hash_key), "R7 " + path_key):
            raise ValueError("R7 frozen checkpoint input changed: " + relative)
    copied_checkpoint = checkpoints.get("copied_checkpoint_relative")
    if not isinstance(copied_checkpoint, str) or digest(regular_file(root, prefix + safe_relative(copied_checkpoint).as_posix())) != checkpoints["checkpoint_sha256"]:
        raise ValueError("R7 frozen copied checkpoint differs from its prior input")
    if digest(root / RESULTS / "manifest.json") != require_sha(checkpoints.get("prior_manifest_sha256"), "R7 prior inventory"):
        raise ValueError("R7 parent result inventory differs from its frozen input")
    prior = read_json(root / RESULTS / "experiment.json").get("provenance", {}).get("code_commit")
    if checkpoints.get("prior_code_commit") != prior:
        raise ValueError("R7 parent calculation identity differs from its checkpoint freeze")
    return [{"name": "R7", "kind": experiment.get("kind"), "source_commit": source_commit,
             "manifest_sha256": expected_sha256, "config_sha256": configuration_sha256,
             "checkpoint_inventory": checkpoints, "recorded_gates": experiment.get("gates"),
             "saved_verification": verification,
             "scope": "independently frozen follow-up experiment; original v6 calculation and gates unchanged"}]


def build(root, output, *, expected_head, code_commit, tag, manifest_sha256, results_sealed=False,
          r7_manifest_sha256=None):
    if results_sealed is not True:
        raise ValueError("Only invoke after observing runner exit and final result sealing")
    root = Path(os.path.abspath(root))
    output = Path(os.path.abspath(output))
    reject_links(root, root / ".v6-package-root-check")
    if root.resolve() != root or Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("The source must be the unlinked Git repository root")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    if head != expected_head or git(root, "rev-parse", tag + "^{commit}").decode().strip() != head:
        raise ValueError("The reviewed result commit, HEAD and release tag must match")
    if git(root, "status", "--porcelain", "--untracked-files=all") or git(root, "remote").strip():
        raise ValueError("Delivery requires a clean local-only repository")
    git(root, "cat-file", "-e", code_commit + "^{commit}")
    experiment, verification, seal = validate_seal(root, manifest_sha256, code_commit)
    additional = validate_r7(root, r7_manifest_sha256, head)
    tracked = tracked_files(root)
    names = collect(root, tracked)
    if output.is_relative_to(root):
        relative_output = output.relative_to(root).as_posix()
        if any(relative_output == prefix or relative_output.startswith(prefix + "/") for prefix in EXTRA_ROOTS):
            raise ValueError("Package output must be outside collected artifact directories")
    archive = output / "F1_ERS_Lab_v6.zip"
    bundle_path = output / "F1_ERS_Lab_v6.bundle"
    record_path = output / "F1_ERS_Lab_v6_delivery.json"
    outputs = [archive, bundle_path, record_path]
    outputs += [path.with_name(path.name + ".sha256") for path in outputs]
    for path in outputs:
        reject_links(output, path)
        if path.exists():
            raise ValueError("Published output must not be overwritten: " + str(path))
    algorithm = git(root, "rev-parse", "--show-object-format").decode().strip()
    records, tokens = {}, {}
    for name in names:
        path = regular_file(root, name)
        token = path.stat()
        if name in tracked and blob_digest(path, algorithm) != tracked[name]["git_blob"]:
            raise ValueError("Tracked bytes differ from the Git index: " + name)
        records[name] = {"sha256": digest(path), "bytes": token.st_size,
                         "kind": "tracked_source_or_history" if name in tracked else "ignored_artifact"}
        tokens[name] = (token.st_size, token.st_mtime_ns)
    # All source, result, path, and existing-output validation is finished.
    output.mkdir(parents=True, exist_ok=True)
    git(root, "bundle", "create", str(bundle_path), "--all")
    bundle_check = subprocess.run(["git", "-C", str(root), "bundle", "verify", str(bundle_path)],
                                  capture_output=True, text=True, check=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name in names:
            path = regular_file(root, name)
            item = path.stat()
            if (item.st_size, item.st_mtime_ns) != tokens[name] or digest(path) != records[name]["sha256"]:
                raise ValueError("Input changed after package prevalidation: " + name)
            member = zipfile.ZipInfo(ZIP_PREFIX + name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            copied = hashlib.sha256()
            with package.open(member, "w", force_zip64=True) as destination, path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    copied.update(block)
                    destination.write(block)
            if copied.hexdigest() != records[name]["sha256"]:
                raise ValueError("Input changed while copying: " + name)
    with zipfile.ZipFile(archive) as package:
        if package.namelist() != [ZIP_PREFIX + name for name in names]:
            raise ValueError("ZIP member inventory differs from the fixed collection")
        for name in names:
            checksum = hashlib.sha256()
            with package.open(ZIP_PREFIX + name) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != records[name]["sha256"]:
                raise ValueError("ZIP member verification failed: " + name)
    if collect(root, tracked) != names:
        raise ValueError("Artifact member inventory changed while packaging")
    for name in names:
        path = regular_file(root, name)
        item = path.stat()
        if (item.st_size, item.st_mtime_ns) != tokens[name] or digest(path) != records[name]["sha256"]:
            raise ValueError("Input changed before publication: " + name)
    if git(root, "rev-parse", "HEAD").decode().strip() != head or git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Source checkout changed while packaging")
    if digest(root / RESULTS / "manifest.json") != manifest_sha256.lower():
        raise ValueError("The result seal changed while packaging")
    if additional and digest(root / RESULTS / "R7/manifest.json") != r7_manifest_sha256.lower():
        raise ValueError("The independent R7 result seal changed while packaging")
    result = {"schema_version": 6, "project": "F1 ERS Lab v6", "zip_prefix": ZIP_PREFIX,
              "tag": tag, "code_commit": code_commit, "result_commit": head,
              "source_remote": None, "results_manifest_sha256": manifest_sha256.lower(),
              "archive_members_verified": True, "bundle_verify_passed": True,
              "bundle_verify_output": bundle_check.stdout + bundle_check.stderr,
              "zip": {"name": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size},
              "bundle": {"name": bundle_path.name, "sha256": digest(bundle_path), "bytes": bundle_path.stat().st_size},
              "files": records, "file_count": len(records), "tracked_file_count": len(tracked),
              "ignored_artifact_count": len(names) - len(tracked),
              "saved_verification": verification, "recorded_scientific_gates": experiment.get("gates"),
              "additional_experiments": additional,
              "release_qualification": "diagnostic delivery; failed fidelity/proxy gates are not converted into a pass",
              "FP2_arrays_included": False, "legacy_fp2_status": "fixed-hash tracked metadata only",
              "independent_bundle_recovery_verified": False,
              "scope": "Git-tracked source/history plus ignored v6 results and direct wheel files; older ignored artifacts, environments and caches excluded",
              "package_tool_sha256": digest(Path(__file__)), "sealed_manifest_member_count": len(seal["files"])}
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for path in (archive, bundle_path, record_path):
        with path.with_name(path.name + ".sha256").open("x", encoding="ascii") as stream:
            stream.write(digest(path) + "  " + path.name + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results-manifest-sha256", required=True)
    parser.add_argument("--r7-manifest-sha256", help="Required if the independently frozen R7 experiment exists")
    parser.add_argument("--results-sealed", action="store_true")
    args = parser.parse_args()
    result = build(args.root, args.output, expected_head=args.expected_head,
                   code_commit=args.code_commit, tag=args.tag,
                   manifest_sha256=args.results_manifest_sha256, results_sealed=args.results_sealed,
                   r7_manifest_sha256=args.r7_manifest_sha256)
    print(json.dumps({key: value for key, value in result.items() if key not in ("files", "saved_verification")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
