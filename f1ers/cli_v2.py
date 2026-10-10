"""Frozen-test causal experiment. Development failure is a reported result."""
from __future__ import annotations

from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from .calibration_v2 import fit_vehicle_v2,compare_observables,accuracy_gate
from .model import VehicleParams
from .reporting import write_json
from .reporting_v2 import render_report_v2,export_trace_v2
from .solvers_v2 import solve_heuristic_v2,solve_optimal_v2,valid_result

PROJECT_ROOT=Path(__file__).resolve().parents[1]


def _hash(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run_v2(args):
    from .data_v2 import prepare_data_v2,load_dataset_v2
    from .causal import CausalSimulator
    root=PROJECT_ROOT
    output=root/"results"/args.name
    output.mkdir(parents=True,exist_ok=True)
    if (output/"test_evaluation.json").exists():
        raise RuntimeError("This experiment already evaluated the frozen test. Use its saved report/checkpoints for replay; do not repeatedly tune against the test.")
    test_access=root/"data/v2/test_access.json"
    if test_access.exists():
        raise RuntimeError("This held-out dataset was already unsealed for an experiment. Read saved results; changing experiment names cannot repeat its prospective test.")
    start=time.perf_counter()
    def progress(message):print(message,flush=True)
    progress("[1/6] Prepare precommitted v2 data and exclude test arrays")
    prepare_data_v2(root,refresh=args.refresh)
    dataset=load_dataset_v2(root,include_test=False)
    write_json(output/"audit.json",dataset["audit"])
    previous=VehicleParams.from_dict(json.loads((root/"results/austria_2026/vehicle.json").read_text(encoding="utf-8")))
    progress("[2/6] Fit three fixed starts using calibration laps 3--6 only")
    vehicle,calibration,heuristic=fit_vehicle_v2(dataset,previous,progress=progress)
    write_json(output/"vehicle.json",vehicle.to_dict())
    write_json(output/"calibration.json",calibration)
    development=compare_observables(dataset,heuristic,"development")
    write_json(output/"development.json",development)
    sim=CausalSimulator(dataset["track"],dataset["rules"],vehicle)
    progress("[3/6] Evaluate controllers with the same causal transition and boundary")
    controllers=[{"id":"heuristic","label":"因果启发式参考","result":heuristic},
        {"id":"no_ers","label":"无 ERS 因果对照","result":sim.simulate(np.zeros(sim.n_nodes))}]
    if calibration["ready_for_training"]:
        numerical=solve_optimal_v2(sim,heuristic)
        controllers.insert(1,{"id":"numerical","label":"有限控制族数值教师","result":numerical})
        progress("[4/6] Development gate passed: train BC/DAgger and bounded SAC diagnostics")
        from .learning_v2 import train_and_evaluate_v2
        learned=train_and_evaluate_v2(sim,numerical,output/"models",progress=progress)
        training=learned["metadata"]
        # Learning exports a controller list explicitly, without undocumented
        # candidate assumptions in the experiment coordinator.
        controllers.extend(learned.get("controllers",[]))
    else:
        progress("[4/6] Development/fit gate failed: formal learning blocked by precommitted protocol")
        training={"skipped":True,"reason":"Convergence/physical/development accuracy gate did not pass. No full BC or SAC training was run.",
            "implemented_protocol":"BC600 + up to3 DAgger600; fixed/auto SAC 3seeds x20; conditional 3seeds x60",
            "development_gate":calibration["development_gate"],"calibration_converged":calibration["solver_success"]}
    # Freeze executable algorithms/config/checkpoints before any test arrays
    # are loaded. The final evaluation is written once and never used to tune.
    freeze={"frozen_at":datetime.now(timezone.utc).isoformat(),"vehicle":vehicle.to_dict(),
        "source_sha256":{p.relative_to(root).as_posix():_hash(p) for p in sorted((root/"f1ers").glob("*.py"))},
        "protocol_sha256":_hash(root/"config/protocol_austria_2026_v2.json"),
        "data_manifest_sha256":_hash(root/"data/v2/manifest.json"),
        "checkpoint_sha256":{p.relative_to(output).as_posix():_hash(p) for p in output.glob("models/**/*.npz")},
        "test_access":"first observable error evaluation after freeze; no subsequent parameter selection"}
    write_json(output/"frozen_configuration.json",freeze)
    progress("[5/6] Evaluate newly held-out telemetry once against frozen vehicle")
    test_access.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation survives interruption, and also protects aliases with
    # a different result-directory name. A failed scoring pass is not retried.
    with test_access.open("x",encoding="utf-8") as handle:
        json.dump({"status":"started","experiment":args.name,"started_at":datetime.now(timezone.utc).isoformat(),
            "protocol_sha256":freeze["protocol_sha256"],"configuration_sha256":_hash(output/"frozen_configuration.json")},handle,indent=2)
    full_dataset=load_dataset_v2(root,include_test=True)
    final_test=compare_observables(full_dataset,heuristic,"test")
    final_gate=accuracy_gate(final_test,valid_result(heuristic))
    write_json(output/"test_evaluation.json",{"comparison":final_test,"gate":final_gate,"freeze":freeze})
    access=json.loads(test_access.read_text(encoding="utf-8"))
    access.update(status="completed",completed_at=datetime.now(timezone.utc).isoformat(),result_sha256=_hash(output/"test_evaluation.json"))
    write_json(test_access,access)
    old=json.loads((root/"results/austria_2026/summary.json").read_text(encoding="utf-8"))
    history={"protocol":"v1 periodic coupled solver + disclosed offline deployment scaling",
        "controller_lap_times_s":{c["id"]:c["summary"]["lap_time_s"] for c in old["controllers"]},
        "comparable_to_v2":False}
    bundle={"schema_version":2,"execution_protocol":sim.protocol,"generated_at":datetime.now(timezone.utc).isoformat(),
        "status":"因果执行实验 · "+("新测试精度通过" if final_gate["passed"] else "精度目标未达到"),
        "track":dataset["track"],"rules":dataset["rules"],"audit":full_dataset["audit"],"vehicle":vehicle.to_dict(),
        "calibration":calibration,"validation":final_test,"development":development,"controllers":controllers,"training":training,
        "gates":{"data":bool(dataset["audit"].get("gate_passed",True)),"calibration_converged":calibration["solver_success"],
            "physical":valid_result(heuristic),"development":calibration["development_gate"],"final_test":final_gate,
            "benefit_claim_allowed":bool(calibration["ready_for_training"] and final_gate["passed"])},
        "history":history,"configuration":{"execution":sim.config.to_dict(),"boundary":sim.boundary_report,"test_frozen":True},
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"elapsed_s":time.perf_counter()-start,"freeze":freeze},
        "limitations":["二维固定路线、干地固定轮胎；真实ICE/K分解、SOC及空力状态不可观测。",
            "固定2MJ保底，仅部署本圈已回收电量；能力范围比第一阶段更窄，不直接比较旧圈速收益。",
            "每次从共同起速与2MJ独立重置，不证明连续多圈稳态。",
            "公开规则子集与数值容差内审计；完整DOC058与PU初始hold状态机未实现。",
            "圈速与速度仅验证等效模型，不能据此还原车队真实ECU。",
            "若开发或新测试精度未通过，暂停算法现实收益结论；失败诊断作为阶段成果。"]}
    write_json(output/"experiment.json",bundle)
    write_json(output/"summary.json",{**{k:v for k,v in bundle.items() if k not in ("track","controllers")},
        "controllers":[{"id":c["id"],"label":c["label"],"summary":c["result"]["summary"]} for c in controllers]})
    for c in controllers:export_trace_v2(output/("trace_"+c["id"]+".csv"),c["result"])
    render_report_v2(output/"report.html",bundle)
    progress("[6/6] Complete: "+str(output/"report.html"))
    return bundle
