"""Run data acquisition, calibration, controllers and one portable report."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

from .reporting import export_trace, render_report, serializable, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def progress(message):
    print(message, flush=True)


def validate_telemetry(dataset: dict, result: dict) -> dict:
    """Compare observable quantities on entire held-out laps only.

    Location is approximate. The acquisition module performs the distance
    alignment, keeping its method and uncertainty in the audit.
    """
    s_sim = np.asarray(result["s_m"], float)
    v_sim = np.asarray(result["speed_mps"], float)
    length = float(dataset["track"]["length_m"])
    metrics, references = [], []
    for lap in dataset["laps"]:
        if lap["role"] != "validation":
            continue
        s = np.asarray(lap["s_m"], float)
        v = np.asarray(lap["speed_mps"], float)
        valid = np.isfinite(s) & np.isfinite(v)
        s, v = s[valid], v[valid]
        order = np.argsort(s)
        s, v = s[order], v[order]
        if len(v) < 10:
            raise ValueError(f"Validation lap {lap['lap_number']} has insufficient valid samples")
        predicted = np.interp(s % length, np.r_[s_sim, length], np.r_[v_sim, v_sim[0]])
        real_time = float(lap["lap_time_s"])
        sim_time = float(result["summary"]["lap_time_s"])
        entry = {
            "lap_number": lap["lap_number"], "driver_number": lap["driver_number"],
            "real_lap_time_s": real_time, "sim_lap_time_s": sim_time,
            "lap_error_s": sim_time-real_time, "lap_error_pct": 100*(sim_time-real_time)/real_time,
            "speed_rmse_kmh": float(np.sqrt(np.mean((predicted-v)**2))*3.6),
            "speed_mae_kmh": float(np.mean(np.abs(predicted-v))*3.6),
            "real_vmax_kmh": float(np.max(v)*3.6), "sim_vmax_kmh": float(np.max(v_sim)*3.6),
            "sample_count": len(v), "real_sector_times_s": lap.get("sector_times_s"),
        }
        t = lap.get("t_s")
        sectors = lap.get("sector_times_s")
        if t is not None and sectors and len(sectors)==3 and all(x is not None for x in sectors):
            # Boundaries are inferred from recorded telemetry time / distance,
            # not asserted to be FIA surveyed sector coordinates.
            t = np.asarray(t, float)[valid][order]
            split_s = np.interp(np.cumsum(sectors)[:2], t, s)
            # Use the solver's segment integration, which includes both
            # endpoint speeds, instead of a separate left-endpoint estimate.
            dt = np.asarray(result["dt_s"], float)
            sim_t = np.r_[0., np.cumsum(dt)]
            times = np.diff(np.r_[0., np.interp(split_s, np.r_[s_sim, length], sim_t), sim_time])
            entry["sim_sector_times_s"] = times.tolist()
            entry["sector_error_s"] = (times-np.asarray(sectors)).tolist()
            entry["sector_boundary_method"] = "held-out telemetry time/distance interpolation; approximate"
        metrics.append(entry)
        references.append(np.interp(s_sim, s, v))
    return {"metrics": metrics, "reference_speed_mps": np.mean(references,axis=0) if references else [],
            "reference_method": "mean of independent held-out laps, aligned by acquisition module",
            "interpretation": "Observable vehicle-model validation; real ERS power and SOC are unavailable. Strategy, fuel, wind and tyre differences remain."}


def validate_data_gate(dataset: dict):
    if not dataset.get("laps"):
        raise ValueError("No measured telemetry. Synthetic data cannot pass the real-data gate.")
    training = [l for l in dataset["laps"] if l.get("role")=="calibration"]
    validation = [l for l in dataset["laps"] if l.get("role")=="validation"]
    if len(training)<2 or len(validation)<2:
        raise ValueError("Need at least two calibration laps and two entirely held-out validation laps")
    train_ids={(l['driver_number'],l['lap_number']) for l in training}
    valid_ids={(l['driver_number'],l['lap_number']) for l in validation}
    if train_ids & valid_ids:
        raise ValueError("A lap appears in both calibration and validation")
    track=dataset["track"]
    if len(track["s_m"])<100 or float(track["length_m"])<=0:
        raise ValueError("Track geometry fails the data gate")
    for lap in dataset["laps"]:
        s=np.asarray(lap["s_m"])
        if len(s)<100 or np.ptp(s)<.90*float(track["length_m"]):
            raise ValueError(f"Lap {lap['lap_number']} is too incomplete for validation")


def run(args):
    from .data import prepare_data, load_dataset
    from .model import Simulator
    from .calibration import fit_vehicle
    from .solvers import solve_heuristic, solve_optimal
    start = time.perf_counter()
    output = PROJECT_ROOT/"results"/args.name
    output.mkdir(parents=True, exist_ok=True)
    progress("[1/6] Load and audit measured data")
    prepare_data(PROJECT_ROOT,refresh=args.refresh)
    dataset=load_dataset(PROJECT_ROOT)
    validate_data_gate(dataset)
    write_json(output/"audit.json",dataset["audit"])
    progress("[2/6] Calibrate equivalent vehicle using calibration laps only")
    vehicle, calibration=fit_vehicle(dataset,dataset["rules"])
    write_json(output/"vehicle.json",vehicle.to_dict())
    write_json(output/"calibration.json",calibration)
    if calibration.get("ready_for_training") is False:
        raise RuntimeError("Model validation gate failed: "+json.dumps(serializable(calibration),ensure_ascii=False))
    sim=Simulator(dataset["track"],dataset["rules"],vehicle)
    progress("[3/6] Evaluate sustainable heuristic and numerical optimal-control baseline")
    heuristic=solve_heuristic(sim)
    optimized=solve_optimal(sim,initial=heuristic)
    controllers=[{"id":"heuristic","label":"启发式基线","result":heuristic},
                 {"id":"optimal_control","label":"SLSQP 数值控制基线","result":optimized},
                 {"id":"no_ers","label":"无 ERS 对照","result":sim.simulate(np.zeros(sim.n_nodes))}]
    for item in controllers:
        if not item["result"]["summary"]["feasible"] or not item["result"]["summary"].get("coupling_converged",False):
            write_json(output/(item["id"]+"_failed.json"),item["result"])
            raise RuntimeError(item["id"]+" failed physical/energy constraints; ML training blocked")
        progress("  "+item["id"]+": "+str(round(item["result"]["summary"]["lap_time_s"],3))+" s")
    validation=validate_telemetry(dataset,heuristic)
    validation["reference_controller"]="calibration_reference_heuristic"
    validation["optimized_strategy_observable_comparison"]=validate_telemetry(dataset,optimized)["metrics"]
    write_json(output/"validation.json",validation)
    progress("[4/6] Train imitation + SAC and evaluate independent random seeds")
    training={"skipped":args.skip_learning}
    if not args.skip_learning:
        from .learning import train_and_evaluate
        learned=train_and_evaluate(sim,optimized,output/"models",seeds=tuple(args.seeds),episodes=args.episodes,bc_epochs=args.bc_epochs,progress=progress)
        training=learned["metadata"]
        for seed, candidates in learned["results"].items():
            controllers.append({"id":f"bc_seed_{seed}","label":f"模仿学习 · seed {seed}","result":candidates["bc"]})
            controllers.append({"id":f"sac_last_seed_{seed}","label":f"SAC 末次候选 · seed {seed}","result":candidates["sac_last"]})
            record=next(r for r in training["records"] if str(r["seed"])==seed)
            if record["best_episode"] and record["best_episode"]!=record["episodes_completed"]:
                controllers.append({"id":f"sac_best_seed_{seed}","label":f"SAC 保留最佳 · seed {seed}","result":candidates["selected"]})
    progress("[5/6] Export full traces, experiment manifest and interactive report")
    limitations=[
        "研究级等效车辆模型。真实 ICE 功率、MGU-K 部署、SOC、轮胎与空气动力参数无法由公开速度唯一识别。",
        "行驶路线固定为 TUM 最小曲率路线；路线及地图来源精度有限。平面模型忽略坡度和倾角。",
        "速度与圈速为可观测量验证，电量和电机功率为模型估计，不能据此还原车队 ECU。",
        "FIA 规则只对公开且已实现的约束审计；未公开的完整状态机及位置映射不确定性另列在规则审计中。",
        "本实验固定车辆、轮胎与干地条件，Overtake 关闭；主动空气动力若采用近似区段会明确记录。",
        "当前完成单赛道实验。全赛道覆盖、额外模式与留一赛道泛化须在新增数据审计后单独验证。",
        "校准搜索的收敛状态单独报告；物理可行不等于统计拟合收敛。",
        "报告圈速包含离线全圈重放选取部署缩放的执行层。学习策略本身尚未证明在线可持续；原始请求的末电欠账和缩放幅度另列。",
    ]
    bundle={"schema_version":1,"generated_at":datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "status":"公开数据研究原型 · 模型约束审计",
            "track":dataset["track"],"rules":dataset["rules"],"audit":dataset["audit"],
            "vehicle":vehicle.to_dict(),"calibration":calibration,"validation":validation,
            "controllers":controllers,"training":training,"limitations":limitations,
            "runtime":{"python":platform.python_version(),"numpy":np.__version__,"elapsed_s":time.perf_counter()-start,
                       "dependencies":{m:importlib.metadata.version(m) for m in ["numpy","scipy","numba","llvmlite","pdfplumber"]},
                       "source_sha256":{str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/"f1ers").glob("*.py")}},
            "configuration":{"seeds":args.seeds,"episodes":args.episodes,"bc_epochs":args.bc_epochs,
                             "sustainable":True,"overtake":False,"track":"Spielberg","data_year":2026}}
    write_json(output/"experiment.json",bundle)
    for item in controllers:
        export_trace(output/("trace_"+item["id"]+".csv"),item["result"])
    render_report(output/"report.html",bundle)
    write_json(output/"summary.json",{**{k:v for k,v in bundle.items() if k not in ("track","controllers","training")},
              "controllers":[{"id":c["id"],"label":c["label"],"summary":c["result"]["summary"],"solver_status":c["result"].get("solver_status")} for c in controllers],
              "training":{k:v for k,v in training.items() if k not in ("controllers","results")}})
    progress("[6/6] Complete: "+str(output/"report.html"))
    return bundle


def main():
    parser=argparse.ArgumentParser(description="2026 F1 public-data sustainable-lap ERS lab")
    sub=parser.add_subparsers(dest="command",required=True)
    fetch=sub.add_parser("fetch",help="Download/audit official and public inputs")
    fetch.add_argument("--refresh",action="store_true")
    run_parser=sub.add_parser("run",help="Calibrate, optimize, train and create report")
    run_parser.add_argument("--refresh",action="store_true")
    run_parser.add_argument("--name",default="austria_2026")
    run_parser.add_argument("--episodes",type=int,default=60)
    run_parser.add_argument("--bc-epochs",type=int,default=600)
    run_parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    run_parser.add_argument("--skip-learning",action="store_true",help="Diagnostic baseline run only")
    sub.add_parser("audit",help="Read cached dataset audit")
    causal=sub.add_parser("run-v2",help="Run the frozen-test causal protocol; development failure blocks full learning")
    causal.add_argument("--refresh",action="store_true")
    causal.add_argument("--name",default="austria_2026_v2")
    verification=sub.add_parser("verify-v2",help="Offline artifact/hash and causal replay checks without test retuning")
    verification.add_argument("--name",default="austria_2026_v2")
    args=parser.parse_args()
    if args.command=="verify-v2":
        if Path(args.name).name!=args.name or args.name in (".",".."):
            parser.error("name must be one safe directory name")
        from .verification_v2 import verify_v2
        print(json.dumps(verify_v2(PROJECT_ROOT,args.name),ensure_ascii=False,indent=2))
        return
    if args.command=="run-v2":
        if Path(args.name).name!=args.name or args.name in (".",".."):
            parser.error("name must be one safe directory name")
        from .cli_v2 import run_v2
        run_v2(args)
        return
    if args.command=="run":
        if args.episodes<1 or args.bc_epochs<1:
            parser.error("episodes and bc-epochs must be positive")
        if Path(args.name).name!=args.name or args.name in (".",".."):
            parser.error("name must be one safe directory name")
        run(args)
    else:
        from .data import prepare_data,load_dataset
        if args.command=="fetch":
            print(json.dumps(serializable(prepare_data(PROJECT_ROOT,refresh=args.refresh)),ensure_ascii=False,indent=2))
        else:
            print(json.dumps(serializable(load_dataset(PROJECT_ROOT)["audit"]),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
