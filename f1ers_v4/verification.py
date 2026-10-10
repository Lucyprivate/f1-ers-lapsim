"""Offline reproduction of frozen local trajectories, never re-open late laps."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from f1ers_v3.artifacts import read_json, write_json
from f1ers_v3.verification import assert_diagnostic_equal
from .artifacts import check_freeze, sha256
from .calibration import compare_paths, metrics_only


def verify(root, name="austria_2026_v4", save=True):
    from .data import load_local_dataset
    from .vehicle import VehicleParamsV4
    from .protocol import read_saved_test_score
    root=Path(root).resolve(); output=root/"results"/name
    bundle=read_json(output/"experiment.json"); freeze=read_json(root/bundle["freeze_path"])
    if freeze != bundle["freeze"]:
        raise ValueError("Saved freeze differs from experiment")
    if freeze["vehicle"] != bundle["vehicle"] or bundle["report"]["fitted_vehicle"] != bundle["vehicle"]:
        raise ValueError("Frozen and fitted parameters differ")
    check_freeze(root, freeze)
    with gzip.open(output/"trajectories.json.gz", "rt", encoding="utf-8") as stream:
        stored=json.load(stream)
    data=load_local_dataset(root); vehicle=VehicleParamsV4.from_dict(bundle["vehicle"])
    for role in ("calibration", "development"):
        actual=compare_paths(data,vehicle,role)
        assert_diagnostic_equal(actual,stored[role],role)
        assert_diagnostic_equal(metrics_only(actual),bundle["local"][role],role+"/summary")
    if bundle["cross_session"]["status"] == "evaluated":
        receipt=read_saved_test_score(root, output/"test_evaluation.json")
        if receipt is None:
            raise ValueError("Final test access receipt is missing")
        cross=bundle["cross_session"]
        check_freeze(root,cross["freeze"])
        assert_diagnostic_equal(receipt["score"],cross,"saved-test-score")
        if receipt["result_sha256"] != sha256(output/"test_evaluation.json"):
            raise ValueError("Saved final test score changed")
        if receipt["trajectory_sha256"] != sha256(output/"test_trajectories.json.gz"):
            raise ValueError("Saved final test trajectories changed")
    check_freeze(root,freeze)
    result={"schema_version":4,"artifact_integrity_passed":True,"exact_local_replay_passed":True,
        "late_arrays_reloaded":False,"training_run":False,"code_commit":bundle["code_provenance"]["code_commit"],
        "fidelity_gates":bundle["gates"],"new_session_status":bundle["cross_session"]["status"],
        "interpretation":"Reproduced fixed-parameter calibration/development trajectories, including failures; saved final test score only"}
    if save:write_json(output/"verification.json",result)
    return result
