"""Additional v5 inventory around the unmodified v4 freeze/ledger contract."""
from pathlib import Path
from f1ers_v4.artifacts import (sha256, read_json, write_json, capture_provenance,
                               freeze_configuration, check_freeze as check_v4)


def source_hashes(root):
    root = Path(root)
    paths = list((root / "f1ers_v5").glob("*.py")) + list((root / "config/v5").glob("*.json"))
    paths += [root / "run-v5.ps1"] if (root / "run-v5.ps1").is_file() else []
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def provenance(root, require_clean=True):
    return {**capture_provenance(root, require_clean=require_clean), "v5_source_sha256": source_hashes(root),
            "v5_protocol_sha256": sha256(Path(root) / "config/v5/protocol.json")}


def freeze(root, output, vehicle, *, stage="local", code_provenance=None, **extra):
    root, output = Path(root), Path(output)
    code_provenance = code_provenance or provenance(root)
    # schema_version=4 is deliberately retained for the shared external one-shot ledger.
    # v5's own inventory extends the certificate; no new test identity is created.
    path, value = freeze_configuration(root, output, vehicle, stage=stage,
                                       provenance=code_provenance, v5_schema_version=5, **extra)
    excluded = {"experiment.json", "verification.json", "report.html", "frozen_configuration.json",
                "frozen_configuration_cross_session.json", "test_evaluation.json", "test_trajectories.json.gz"}
    value["v5_calculation_sha256"] = {p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name not in excluded and p.suffix in (".json", ".gz")}
    write_json(path, value)
    return path, read_json(path)


def check_freeze(root, value):
    root = Path(root).resolve()
    value = read_json(value) if isinstance(value, (str, Path)) else value
    check_v4(root, value)
    if value.get("v5_schema_version") != 5 or value.get("v5_source_sha256") != source_hashes(root):
        raise ValueError("Frozen v5 executable inventory changed")
    for relative, checksum in value.get("v5_calculation_sha256", {}).items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != checksum:
            raise ValueError("Frozen v5 calculation changed: " + relative)
    return True
