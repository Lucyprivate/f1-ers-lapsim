"""Offline replay checks without recomputing or tuning against held-out errors."""
import hashlib
import json
from pathlib import Path
import numpy as np

from .model import VehicleParams
from .solvers_v2 import solve_heuristic_v2


def verify_v2(root,experiment_name="austria_2026_v2"):
    from .causal import CausalSimulator
    from .data_v2 import load_dataset_v2
    root=Path(root);output=root/"results"/experiment_name
    bundle=json.loads((output/"experiment.json").read_text(encoding="utf-8"))
    freeze=bundle["runtime"]["freeze"]
    for relative,digest in freeze["source_sha256"].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest,relative
    assert hashlib.sha256((root/"config/protocol_austria_2026_v2.json").read_bytes()).hexdigest()==freeze["protocol_sha256"]
    assert hashlib.sha256((root/"data/v2/manifest.json").read_bytes()).hexdigest()==freeze["data_manifest_sha256"]
    data=load_dataset_v2(root,include_test=False)
    assert all(l["role"]!="test" for l in data["laps"])
    sim=CausalSimulator(data["track"],data["rules"],VehicleParams.from_dict(bundle["vehicle"]),bundle["configuration"]["execution"])
    checks=[];physical=True
    for stored in bundle["controllers"]:
        if stored["id"]=="heuristic":replay=solve_heuristic_v2(sim)
        elif stored["id"]=="no_ers":replay=sim.simulate(np.zeros(sim.n_nodes))
        elif stored.get("checkpoint"):
            from .learning_v2 import load_policy_v2,evaluate_policy_v2
            replay=evaluate_policy_v2(sim,load_policy_v2(output/stored["checkpoint"]))
        elif stored["id"]=="numerical":replay=sim.simulate(np.asarray(stored["result"]["requested_actions"]))
        elif not stored["result"]["summary"].get("feasible"):
            checks.append({"controller":stored["id"],"replay":"no valid checkpoint; rejected candidate retained only for diagnostics"})
            continue
        else:raise AssertionError("No replayable checkpoint for "+stored["id"])
        actual=stored["result"]
        for key in ("speed_mps","dt_s","energy_mj","mguk_dc_kw"):
            np.testing.assert_allclose(replay[key],actual[key],atol=1e-8,rtol=0,err_msg=stored["id"]+":"+key)
        checks.append({"controller":stored["id"],"lap_time_delta_s":abs(replay["summary"]["lap_time_s"]-actual["summary"]["lap_time_s"]),
            "replay_feasible":replay["summary"]["feasible"],"offline_scaling":False})
        # Rejected diagnostic candidates may be present, but an accepted
        # controller's physical failure is separate from artifact integrity.
        if actual["summary"].get("feasible") or stored["id"] in ("heuristic","no_ers"):
            physical=physical and bool(replay["summary"].get("feasible")) and bool(replay["summary"].get("completed_lap"))
    for relative,digest in freeze["checkpoint_sha256"].items():
        assert hashlib.sha256((output/relative).read_bytes()).hexdigest()==digest,relative
    return {"passed":physical,"artifact_integrity_passed":True,"physical_passed":physical,"checks":checks,"source_and_protocol_hashes_match":True,
        "test_arrays_reloaded":False,"checkpoint_hashes_match":True,"execution_protocol":sim.protocol,
        "accuracy_gate_passed":bundle["gates"]["final_test"]["passed"]}
