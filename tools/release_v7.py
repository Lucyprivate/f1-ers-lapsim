"""Package independently sealed v7 A/B/C/D results, matching tracked source, and offline wheels.

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

try:
    from restore_v7 import (ZIP_PREFIX, LEGACY_METADATA_SHA256, allowed_artifact, digest,
                            forbidden_arrays, reject_links, safe_relative, SINGLE_ARTIFACTS)
except ModuleNotFoundError:
    from tools.restore_v7 import (ZIP_PREFIX, LEGACY_METADATA_SHA256, allowed_artifact, digest,
                                  forbidden_arrays, reject_links, safe_relative, SINGLE_ARTIFACTS)

EXPERIMENT_ROOTS = ("results/austria_2026_v7", "results/austria_2026_v7B", "results/austria_2026_v7C", "results/austria_2026_v7D")
R7_ROOT = "results/austria_2026_v6/R7"
DELIVERY_ROOT = "results/austria_2026_v7_delivery"
R7_SOURCE_COMMIT = "b98443c7efe42fd3bb2541c25675e0b51362256a"
R7_MANIFEST_SHA256 = "9b3e1c3d30b96f5ae49cd3fcc847bc84ca760671dc6bb40f0628d8fdadf0c5a1"
R7_CHECKPOINT_SHA256 = "c9fd321a130dffdb370b0aaf87adb7e54d51c60d97c2d25da5f3dc464f4bb611"
MAX_NEW_SAC_EPISODES = 120
EXTRA_ROOTS = (*EXPERIMENT_ROOTS, DELIVERY_ROOT, R7_ROOT, "vendor/wheels", "vendor/reporting-wheels")
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
        if prefix == R7_ROOT:
            # Historical R7 was sealed before its separate prose report existed.
            # Collect only the 76 fixed manifest members and the manifest itself.
            files = read_json(regular_file(root, prefix+"/manifest.json")).get("files")
            if not isinstance(files, dict) or not files:
                raise ValueError("Fixed historical R7 inventory is missing")
            for relative in (*files, "manifest.json"):
                name = prefix+"/"+safe_relative(relative).as_posix()
                regular_file(root, name)
                if name not in tracked and subprocess.run(
                        ["git", "-C", str(root), "check-ignore", "-q", "--", name],
                        capture_output=True).returncode:
                    raise ValueError("Allowed artifact is not Git ignored: " + name)
                names.add(name)
            continue
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
    for name in SINGLE_ARTIFACTS:
        if not (root/name).exists():
            continue
        regular_file(root, name)
        if name not in tracked:
            if subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", name],
                              capture_output=True).returncode:
                raise ValueError("Allowed singleton artifact is not Git ignored: " + name)
        names.add(name)
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Case-colliding package members")
    if any(any(parent.as_posix() in names for parent in PurePosixPath(name).parents
               if parent != PurePosixPath(".")) for name in names):
        raise ValueError("File/descendant package collision")
    return sorted(names)


def require_commit(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError("A fixed full Git commit is required for " + label)
    return value


def git_object_digest(root, commit, name):
    """Hash the frozen Git tree blob without altering the current checkout."""
    safe_relative(name)
    checksum = hashlib.sha256()
    with subprocess.Popen(["git", "-C", str(root), "cat-file", "blob", commit+":"+name],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for block in iter(lambda: process.stdout.read(1024*1024), b""):
            checksum.update(block)
        error = process.stderr.read()
        if process.wait():
            raise ValueError("Frozen input has no Git tree blob: " + name + "; " + error.decode(errors="replace"))
    return checksum.hexdigest()


def validate_inventory(root, prefix, manifest, extra_files=(), *, strict_directory=True):
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("A nonempty sealed artifact inventory is required: " + prefix)
    seen = set()
    for relative, expected in files.items():
        safe_relative(relative)
        if relative == "manifest.json" or relative.casefold() in seen:
            raise ValueError("Recursive or duplicate result inventory: " + relative)
        seen.add(relative.casefold())
        if digest(regular_file(root, prefix+"/"+relative)) != require_sha(expected, "artifact "+relative):
            raise ValueError("Sealed calculation artifact changed: " + prefix+"/"+relative)
    if not strict_directory:
        return files
    permitted = set(files) | {"manifest.json"} | set(extra_files)
    actual = set()
    for current, directories, names in os.walk(root/prefix, followlinks=False):
        current = Path(current)
        for name in directories:
            reject_links(root, current/name)
        directories[:] = [name for name in directories if not excluded((current/name).relative_to(root).as_posix())]
        for name in names:
            path = current/name
            relative = path.relative_to(root/prefix).as_posix()
            regular_file(root, path.relative_to(root).as_posix())
            if not excluded(relative):
                actual.add(relative)
    if actual != permitted:
        raise ValueError("Sealed result directory has missing or unlisted files: " + prefix +
                         "; extra="+repr(sorted(actual-permitted))+"; missing="+repr(sorted(permitted-actual)))
    return files


def validate_frozen_inputs(root, freeze, code_commit):
    if freeze.get("code_commit") != code_commit:
        raise ValueError("Freeze calculation identity differs from the fixed expectation")
    tracked = freeze.get("tracked_input_sha256")
    wheels = freeze.get("dependency_wheels_sha256")
    if not isinstance(tracked, dict) or not tracked or not isinstance(wheels, dict):
        raise ValueError("Frozen source/data/dependency inventories are missing")
    for name, expected in tracked.items():
        expected = require_sha(expected, "frozen tracked input "+name)
        if (digest(regular_file(root, name)) != expected
                or git_object_digest(root, code_commit, name) != expected):
            raise ValueError("Frozen input differs from current bytes or its calculation Git tree: " + name)
    for name, expected in wheels.items():
        if (PurePosixPath(name).parent.as_posix() != "vendor/wheels"
                or not name.endswith(".whl")
                or digest(regular_file(root, name)) != require_sha(expected, "frozen wheel "+name)):
            raise ValueError("Frozen dependency wheel changed: " + name)
    return tracked, wheels


def validate_receipts(root, tracked, prefix, expectation, experiment):
    receipts = expectation.get("receipts")
    if not isinstance(receipts, dict) or set(receipts) != {"summary", "verification"}:
        raise ValueError("Fixed tracked summary and verification receipts are required: " + prefix)
    values = {}
    for kind, spec in receipts.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
            raise ValueError("Receipt path is missing: " + kind)
        name = safe_relative(spec["path"]).as_posix()
        if name not in tracked or not name.startswith("summaries/v7/") or not name.endswith(".json"):
            raise ValueError("Calculation receipt must be an exact tracked summaries/v7 JSON: " + name)
        path = regular_file(root, name)
        if digest(path) != require_sha(spec.get("sha256"), "fixed receipt "+name):
            raise ValueError("Tracked calculation receipt changed: " + name)
        value = read_json(path)
        if (value.get("source_commit") != expectation["source_commit"]
                or value.get("results_manifest_sha256") != expectation["manifest_sha256"]):
            raise ValueError("Receipt calculation identity mismatch: " + name)
        if kind == "verification":
            if value.get("passed") is not True or value.get("FP2_accessed") is not False:
                raise ValueError("Saved exact verification must pass without FP2 access: " + name)
        elif value.get("gates") != experiment.get("gates"):
            raise ValueError("Tracked scientific summary gates differ from the frozen experiment: " + name)
        if "training_status" in value and value["training_status"] != experiment.get("training_status"):
            raise ValueError("Tracked training status differs from the frozen experiment: " + name)
        values[kind] = {"path": name, "sha256": digest(path), "value": value}
    return values


def validate_bound_c_inputs(root, prefix, manifest, experiment):
    """Respect C's frozen interface without rewriting its older manifest."""
    binding_name = "inputs/input_bindings.json"
    if prefix != "results/austria_2026_v7C" or binding_name not in manifest.get("files", {}):
        return []
    config = read_json(regular_file(root, prefix+"/frozen_protocol.json"))
    if config.get("prior_name") != "austria_2026_v7B":
        raise ValueError("C input bindings must name the fixed B calculation")
    prior = "results/austria_2026_v7B"
    seal = require_sha(config.get("prior_manifest_sha256"), "C fixed B manifest")
    if (experiment.get("prior_manifest_sha256") != seal
            or digest(regular_file(root, prior+"/manifest.json")) != seal):
        raise ValueError("C prior manifest is not its fixed B manifest")
    prior_manifest = read_json(root/prior/"manifest.json")
    if (config.get("prior_source_commit") != prior_manifest.get("source_commit")
            or experiment.get("prior_source_commit") != prior_manifest.get("source_commit")):
        raise ValueError("C prior source identity differs from B")
    binding_path = regular_file(root, prefix+"/"+binding_name)
    if digest(binding_path) != require_sha(manifest["files"][binding_name], "C input bindings"):
        raise ValueError("Sealed C input bindings changed")
    bindings = read_json(binding_path)
    inputs = bindings.get("prior_input_sha256")
    if (not isinstance(inputs, dict) or not {"manifest.json", "selection.json"} <= set(inputs)
            or bindings.get("controller_weights_changed") is not False):
        raise ValueError("C prior copy inventory or unchanged controller declaration is missing")
    prior_selection = read_json(regular_file(root, prior+"/selection.json"))
    source_controller = prior_selection.get("winner", {}).get("controller")
    if bindings.get("source_controller_spec") != source_controller:
        raise ValueError("C source controller is not the selected B controller")
    extra = []
    checkpoint = source_controller.get("checkpoint") if isinstance(source_controller, dict) else None
    for name, expected in inputs.items():
        safe_relative(name)
        expected = require_sha(expected, "C inherited B input "+name)
        if name not in {"manifest.json", "selection.json", checkpoint} or forbidden_arrays(name):
            raise ValueError("Unlisted C prior copy binding: "+name)
        prior_sha = seal if name == "manifest.json" else prior_manifest.get("files", {}).get(name)
        if (expected != prior_sha or digest(regular_file(root, prior+"/"+name)) != expected
                or (name == "selection.json" and expected != config.get("prior_selection_sha256"))):
            raise ValueError("C copied input differs from its fixed B inventory: "+name)
        copied = "inputs/"+(name if name in ("manifest.json", "selection.json") else "selected_controller_v7.npz")
        if digest(regular_file(root, prefix+"/"+copied)) != expected:
            raise ValueError("C copied B input changed: "+copied)
        if copied not in manifest["files"]:
            if copied != "inputs/manifest.json":
                raise ValueError("Only C's fixed copied B manifest may be outside its result inventory")
            extra.append(copied)
    return extra


def saved_sac_counts(root, prefix, files):
    """Check actual D worker receipts against retained start/trajectory files."""
    cases = {}
    for name in files:
        if (PurePosixPath(name).name != "sac_metadata.json"
                or name.startswith("inputs/") or "/inputs/" in name):
            continue
        metadata = read_json(regular_file(root, prefix+"/"+name))
        requested, completed = metadata.get("episodes_requested"), metadata.get("episodes_completed")
        if (type(requested) is not int or type(completed) is not int
                or not 0 <= completed <= requested <= 20):
            raise ValueError("Actual bounded SAC episode counts are missing or invalid: " + name)
        cases[PurePosixPath(name).parent.as_posix()] = {
            "path": name, "episodes_requested": requested, "episodes_completed": completed,
            "episodes_attempted": completed}
    if prefix != "results/austria_2026_v7D":
        return list(cases.values())
    for name in files:
        matched = re.fullmatch(r"(D[34]/(?:2026|2027|2028))/worker_receipt\.json", name)
        if not matched:
            continue
        directory = matched.group(1)
        receipt = read_json(regular_file(root, prefix+"/"+name))
        started_name = directory+"/worker_started.json"
        if started_name not in files:
            raise ValueError("D SAC worker has no fixed request receipt: "+directory)
        started = read_json(regular_file(root, prefix+"/"+started_name))
        requested = receipt.get("episodes_requested")
        attempted = receipt.get("episodes_attempted")
        completed = receipt.get("episodes_completed")
        if (any(type(value) is not int for value in (requested, attempted, completed))
                or requested != 20 or requested != started.get("episodes_requested")
                or not 0 <= completed <= attempted <= requested):
            raise ValueError("D worker episode counters exceed or differ from their fixed request")
        starts, trajectories = {}, {}
        for relative in files:
            start = re.fullmatch(re.escape(directory)+r"/episode_(\d{3})_started\.json", relative)
            trajectory = re.fullmatch(re.escape(directory)+r"/training_episode_(\d{3})\.json\.gz", relative)
            if start:
                starts[int(start.group(1))] = relative
            if trajectory:
                trajectories[int(trajectory.group(1))] = relative
        if (set(starts) != set(range(1, attempted+1))
                or set(trajectories) != set(range(1, completed+1))):
            raise ValueError("D worker completed/attempted counts differ from saved episode receipts")
        full_laps = 0
        for relative in trajectories.values():
            import gzip
            with gzip.open(regular_file(root, prefix+"/"+relative), "rt", encoding="utf-8") as stream:
                trajectory = json.load(stream, object_pairs_hook=unique_pairs)
            if not trajectory.get("dones") or trajectory["dones"][-1] is not True:
                raise ValueError("A D completed episode must have a saved terminal transition")
            full_laps += trajectory.get("result", {}).get("summary", {}).get("completed_lap") is True
        if (receipt.get("physical_full_laps_completed") != full_laps
                or receipt.get("completed_includes_executor_terminal_failure_episodes") is not True):
            raise ValueError("D worker physical/full versus terminal-failure episode count differs")
        if directory in cases and (cases[directory]["episodes_requested"] != requested
                                  or cases[directory]["episodes_completed"] != completed):
            raise ValueError("D SAC metadata differs from the durable worker episode receipts")
        cases[directory] = {"path": name, "episodes_requested": requested,
            "episodes_attempted": attempted, "episodes_completed": completed,
            "physical_full_laps_completed": full_laps,
            "sac_metadata_retained": directory+"/sac_metadata.json" in files,
            "episode_zero_counted_as_new_SAC": False,
            "terminal_failure_episodes_count_as_completed": True}
    for directory, record in cases.items():
        if not re.fullmatch(r"D[34]/(?:2026|2027|2028)", directory) or record["path"].endswith("sac_metadata.json"):
            raise ValueError("D SAC cases require their exact durable seed-worker receipt")
    return list(cases.values())


def validate_experiment(root, prefix, expectation, head, tracked):
    if prefix not in EXPERIMENT_ROOTS or not isinstance(expectation, dict):
        raise ValueError("Only the four independently frozen v7 experiments are accepted")
    code = require_commit(expectation.get("source_commit"), prefix)
    seal = require_sha(expectation.get("manifest_sha256"), prefix+" manifest")
    git(root, "merge-base", "--is-ancestor", code, head)
    path = regular_file(root, prefix+"/manifest.json")
    if digest(path) != seal:
        raise ValueError("Sealed results manifest SHA256 mismatch: " + prefix)
    manifest = read_json(path)
    experiment = read_json(regular_file(root, prefix+"/experiment.json"))
    freeze = read_json(regular_file(root, prefix+"/freeze.json"))
    if (manifest.get("source_commit") != code or experiment.get("source_commit") != code
            or experiment.get("schema_version") != 7 or not isinstance(experiment.get("winner"), dict)
            or experiment["winner"].get("qualification", {}).get("passed") is not True):
        raise ValueError("A complete independently certified winner and matching calculation identity are required: " + prefix)
    if experiment.get("FP2_accessed") is not False or experiment.get("full_FIA_compliance_proven") is not False:
        raise ValueError("The diagnostic delivery must retain its public-rule and unopened FP2 scope")
    validate_frozen_inputs(root, freeze, code)
    inherited = experiment.get("inherited_ignored_inputs_sha256", {})
    if not isinstance(inherited, dict):
        raise ValueError("Inherited ignored input inventory is malformed")
    extra = []
    basenames = set()
    for name, expected in inherited.items():
        safe_relative(name)
        if forbidden_arrays(name):
            raise ValueError("Inherited FP2/late array paths cannot be hidden by copying their basenames")
        if name in LEGACY_METADATA_SHA256 and expected != LEGACY_METADATA_SHA256[name]:
            raise ValueError("Inherited historical metadata does not have its fixed SHA256")
        base = PurePosixPath(name).name
        if base.casefold() in basenames:
            raise ValueError("Inherited input copies collide by basename")
        basenames.add(base.casefold())
        copied = "inputs/"+base
        if digest(regular_file(root, prefix+"/"+copied)) != require_sha(expected, "inherited input "+name):
            raise ValueError("Frozen inherited input copy changed: " + name)
        if copied not in manifest.get("files", {}):
            extra.append(copied)
    extra.extend(validate_bound_c_inputs(root, prefix, manifest, experiment))
    files = validate_inventory(root, prefix, manifest, extra)
    for required in ("experiment.json", "freeze.json", "frozen_protocol.json", "selection.json",
                     "selected_calibration.json", "development_score.json", "numerical_audit.json"):
        if required not in files:
            raise ValueError("Complete experiment inventory is missing: " + required)
    config_sha = digest(regular_file(root, prefix+"/frozen_protocol.json"))
    if config_sha not in freeze["tracked_input_sha256"].values():
        raise ValueError("Frozen protocol has no calculation Git tree input anchor")
    selection = read_json(regular_file(root, prefix+"/selection.json"))
    if selection.get("source_commit") != code or selection.get("winner") != experiment["winner"]:
        raise ValueError("Selected controller differs between freeze and experiment")
    receipts = validate_receipts(root, tracked, prefix, expectation, experiment)
    sac_cases = saved_sac_counts(root, prefix, files)
    actual_episodes = sum(case["episodes_completed"] for case in sac_cases)
    requested_episodes = sum(case["episodes_requested"] for case in sac_cases)
    attempted_episodes = sum(case["episodes_attempted"] for case in sac_cases)
    for field, actual in (("actual_new_SAC_episodes_completed", actual_episodes),
                          ("actual_new_SAC_episodes_requested", requested_episodes),
                          ("actual_new_SAC_episodes_attempted", attempted_episodes)):
        if field in experiment and (type(experiment[field]) is not int or experiment[field] != actual):
            raise ValueError("Recorded SAC episode counter differs from saved training metadata: "+field)
    training_status = experiment.get("training_status", "not_recorded")
    if (actual_episodes == 0 and isinstance(training_status, str) and
            (training_status == "bounded_cold_and_warm_experiments_completed" or
             ("sac" in training_status.lower() and "completed" in training_status.lower()))):
        raise ValueError("Zero SAC episodes cannot be recorded as completed SAC training")
    summary_value = receipts["summary"]["value"]
    if ("actual_new_SAC_episodes" in summary_value
            and summary_value["actual_new_SAC_episodes"] != actual_episodes):
        raise ValueError("Tracked summary SAC episode count differs from saved training metadata")
    return {"path": prefix, "source_commit": code, "manifest_sha256": seal,
            "recorded_scientific_gates": experiment.get("gates"), "receipts": receipts,
            "training_status": training_status, "actual_new_SAC_episodes": actual_episodes,
            "actual_new_SAC_episodes_requested": requested_episodes,
            "actual_new_SAC_episodes_attempted": attempted_episodes,
            "saved_SAC_case_counts": sac_cases,
            "manifest_member_count": len(files), "interpretation": "finite tested diagnostic experiment; scientific gates are preserved"}


def validate_prior_r7(root, head):
    path = regular_file(root, R7_ROOT+"/manifest.json")
    if digest(path) != R7_MANIFEST_SHA256:
        raise ValueError("The fixed historical R7 manifest changed")
    manifest = read_json(path)
    if manifest.get("source_commit") != R7_SOURCE_COMMIT or len(manifest.get("files", {})) != 76:
        raise ValueError("Historical R7 calculation identity or 76-member inventory differs")
    git(root, "merge-base", "--is-ancestor", R7_SOURCE_COMMIT, head)
    validate_inventory(root, R7_ROOT, manifest, strict_directory=False)
    freeze = read_json(regular_file(root, R7_ROOT+"/freeze.json"))
    if freeze.get("code_commit") != R7_SOURCE_COMMIT:
        raise ValueError("Historical R7 freeze calculation identity differs")
    for section in ("source_sha256", "data_sha256"):
        inventory = freeze.get(section)
        if not isinstance(inventory, dict) or (section == "source_sha256" and not inventory):
            raise ValueError("Historical R7 frozen source/data inventory is missing")
        for name, expected in inventory.items():
            expected = require_sha(expected, "R7 frozen input "+name)
            if (digest(regular_file(root, name)) != expected
                    or git_object_digest(root, R7_SOURCE_COMMIT, name) != expected):
                raise ValueError("Historical R7 frozen input changed: " + name)
    verification = read_json(regular_file(root, R7_ROOT+"/verification.json"))
    if (verification.get("source_commit") != R7_SOURCE_COMMIT
            or verification.get("passed") is not True or verification.get("FP2_accessed") is not False):
        raise ValueError("Historical R7 verification is not its frozen successful reproduction")
    checkpoint = regular_file(root, SINGLE_ARTIFACTS[0])
    if digest(checkpoint) != R7_CHECKPOINT_SHA256:
        raise ValueError("The fixed raw BC2027 checkpoint dependency changed")
    return {"path": R7_ROOT, "source_commit": R7_SOURCE_COMMIT,
            "manifest_sha256": R7_MANIFEST_SHA256, "manifest_member_count": 76,
            "raw_checkpoint_sha256": R7_CHECKPOINT_SHA256}


def validate_delivery(root, expectation, head):
    if not isinstance(expectation, dict):
        raise ValueError("A separately sealed diagnostic report inventory is required")
    seal = require_sha(expectation.get("manifest_sha256"), "delivery manifest")
    code = require_commit(expectation.get("source_commit"), "delivery source")
    git(root, "merge-base", "--is-ancestor", code, head)
    path = regular_file(root, DELIVERY_ROOT+"/manifest.json")
    if digest(path) != seal:
        raise ValueError("Diagnostic delivery manifest SHA256 mismatch")
    manifest = read_json(path)
    if code != head or manifest.get("source_commit") != code or manifest.get("result_commit") != head:
        raise ValueError("Report calculation/result commit identities differ")
    validate_inventory(root, DELIVERY_ROOT, manifest)
    if not any(name.endswith(".html") for name in manifest["files"]):
        raise ValueError("The separately sealed self-contained HTML report is absent")
    return {"path": DELIVERY_ROOT, "source_commit": code, "result_commit": head,
            "manifest_sha256": seal, "manifest_member_count": len(manifest["files"])}


def build(root, output, *, expected_head, tag, experiments, delivery, results_sealed=False):
    if results_sealed is not True:
        raise ValueError("Only invoke after observing all runner exits and final result sealing")
    root, output = Path(os.path.abspath(root)), Path(os.path.abspath(output))
    reject_links(root, root/".v7-package-root-check")
    if root.resolve() != root or Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("The source must be the unlinked Git repository root")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    require_commit(expected_head, "reviewed result commit")
    if not isinstance(tag, str) or tag.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", tag):
        raise ValueError("One fixed release tag required")
    git(root, "check-ref-format", "refs/tags/"+tag)
    if head != expected_head or git(root, "rev-parse", "refs/tags/"+tag+"^{commit}").decode().strip() != head:
        raise ValueError("The reviewed result commit, HEAD and release tag must match")
    if git(root, "status", "--porcelain", "--untracked-files=all") or git(root, "remote").strip():
        raise ValueError("Delivery requires a clean local-only repository")
    if not isinstance(experiments, dict) or set(experiments) != set(EXPERIMENT_ROOTS):
        raise ValueError("Fixed manifest SHA and code commit expectations are required for A/B/C/D")
    tracked = tracked_files(root)
    calculations = [validate_experiment(root, prefix, experiments[prefix], head, tracked)
                    for prefix in EXPERIMENT_ROOTS]
    actual_sac_episodes = sum(c["actual_new_SAC_episodes"] for c in calculations)
    requested_sac_episodes = sum(c.get("actual_new_SAC_episodes_requested", c["actual_new_SAC_episodes"])
                                 for c in calculations)
    attempted_sac_episodes = sum(c.get("actual_new_SAC_episodes_attempted", c["actual_new_SAC_episodes"])
                                 for c in calculations)
    if not 0 <= actual_sac_episodes <= attempted_sac_episodes <= requested_sac_episodes <= MAX_NEW_SAC_EPISODES:
        raise ValueError("Saved actual SAC training exceeds the fixed 120-episode continuation budget")
    historical = validate_prior_r7(root, head)
    report = validate_delivery(root, delivery, head)
    names = collect(root, tracked)
    if output.is_relative_to(root):
        raise ValueError("Published outputs must be outside the repository")
    archive, bundle_path, record_path = (output/"F1_ERS_Lab_v7.zip", output/"F1_ERS_Lab_v7.bundle",
                                       output/"F1_ERS_Lab_v7_delivery.json")
    outputs = [archive, bundle_path, record_path]
    outputs += [path.with_name(path.name+".sha256") for path in outputs]
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
        tokens[name] = token.st_size, token.st_mtime_ns
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
            member = zipfile.ZipInfo(ZIP_PREFIX+name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system, member.external_attr = 3, (stat.S_IFREG|0o644) << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            copied = hashlib.sha256()
            with package.open(member, "w", force_zip64=True) as destination, path.open("rb") as source:
                for block in iter(lambda: source.read(1024*1024), b""):
                    copied.update(block)
                    destination.write(block)
            if copied.hexdigest() != records[name]["sha256"]:
                raise ValueError("Input changed while copying: " + name)
    with zipfile.ZipFile(archive) as package:
        if package.namelist() != [ZIP_PREFIX+name for name in names]:
            raise ValueError("ZIP member inventory differs from the fixed collection")
        for name in names:
            checksum = hashlib.sha256()
            with package.open(ZIP_PREFIX+name) as source:
                for block in iter(lambda: source.read(1024*1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != records[name]["sha256"]:
                raise ValueError("ZIP member verification failed: " + name)
    if collect(root, tracked) != names:
        raise ValueError("Artifact inventory changed while packaging")
    for name in names:
        path = regular_file(root, name)
        item = path.stat()
        if (item.st_size, item.st_mtime_ns) != tokens[name] or digest(path) != records[name]["sha256"]:
            raise ValueError("Input changed before publication: " + name)
    if (git(root, "rev-parse", "HEAD").decode().strip() != head
            or git(root, "rev-parse", "refs/tags/"+tag+"^{commit}").decode().strip() != head
            or git(root, "status", "--porcelain", "--untracked-files=all") or git(root, "remote").strip()):
        raise ValueError("Source checkout or tag changed while packaging")
    # Revalidate all calculation identities before writing a publication receipt.
    for prefix in EXPERIMENT_ROOTS:
        validate_experiment(root, prefix, experiments[prefix], head, tracked)
    validate_prior_r7(root, head)
    validate_delivery(root, delivery, head)
    result = {"schema_version": 7, "project": "F1 ERS Lab v7", "zip_prefix": ZIP_PREFIX,
              "tag": tag, "result_commit": head, "calculation_code_commits": [c["source_commit"] for c in calculations],
              "source_remote": None, "calculations": calculations, "historical_R7": historical,
              "actual_new_SAC_episodes": actual_sac_episodes, "new_SAC_episode_budget": MAX_NEW_SAC_EPISODES,
              "actual_new_SAC_episodes_requested": requested_sac_episodes,
              "actual_new_SAC_episodes_attempted": attempted_sac_episodes,
              "diagnostic_report": report, "archive_members_verified": True, "bundle_verify_passed": True,
              "bundle_verify_output": bundle_check.stdout+bundle_check.stderr,
              "zip": {"name": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size},
              "bundle": {"name": bundle_path.name, "sha256": digest(bundle_path), "bytes": bundle_path.stat().st_size},
              "files": records, "file_count": len(records), "tracked_file_count": len(tracked),
              "ignored_artifact_count": len(names)-len(tracked), "FP2_arrays_included": False,
              "legacy_fp2_status": "fixed-hash tracked historical metadata only",
              "release_qualification": "diagnostic delivery; failed accuracy/new-session/full-FIA/true-SOC gates remain failed",
              "independent_bundle_recovery_verified": False,
              "scope": "exact tracked source/history plus separately sealed A/B/C/D/report/R7 and one raw checkpoint; direct wheels only",
              "package_tool_sha256": digest(Path(__file__))}
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    for path in (archive, bundle_path, record_path):
        with path.with_name(path.name+".sha256").open("x", encoding="ascii") as stream:
            stream.write(digest(path)+"  "+path.name+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, required=True,
                        help="External fixed JSON with expected_head, tag, experiments and delivery identities")
    parser.add_argument("--results-sealed", action="store_true")
    args = parser.parse_args()
    value = read_json(args.expectations)
    result = build(args.root, args.output, expected_head=value["expected_head"], tag=value["tag"],
                   experiments=value["experiments"], delivery=value["delivery"], results_sealed=args.results_sealed)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
