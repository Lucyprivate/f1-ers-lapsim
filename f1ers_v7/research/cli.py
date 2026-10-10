"""Audit, bounded physics/teacher/learning execution and artifact verification."""
from pathlib import Path
import argparse
import json
from .._core.provenance import read
from .._core.learning_base import _write


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('audit','physics','teacher','learning','verify'))
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--ledger-root',type=Path)
    parser.add_argument('--name',default='opensource_control_2026_01')
    parser.add_argument('--backend',choices=('numpy','sb3','both'),default='both')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--replay',action='store_true',help='Verify: replay the baseline six carried laps')
    args=parser.parse_args(argv)
    root=args.root.resolve()
    from ..deployment_experiment import result_directory
    from .experiment import ensure_run,make_sim,stage_teacher,verify_saved,final_manifest
    output=result_directory(root,args.name)
    try:
        config=read(root/'config/research/experiment.json')
        if args.stage=='audit':
            result={'config':config,'optional_dependencies':{},'FP2_accessed':False,
                    'historical_arrays_loaded':bool(args.data_root),'training_started':False}
            import importlib.metadata
            for name in ('torch','stable-baselines3','gymnasium','casadi'):
                try: result['optional_dependencies'][name]=importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError: result['optional_dependencies'][name]=None
            from .backends import differential_audit
            try: result['backend_differential']=differential_audit()
            except ImportError as exc:
                result['backend_differential']={'passed':False,'status':'optional_dependency_missing','reason':str(exc)}
            if args.data_root:
                from .data import load_historical
                from .audit import audit_inputs
                cal=load_historical(args.data_root,root,role='calibration')
                result['calibration_diagnostics']=audit_inputs(cal,make_sim(cal,config))
        elif args.stage=='verify':
            if args.data_root is None: raise ValueError('--data-root is required for private-input verification')
            result=verify_saved(root,args.data_root,output,replay=args.replay)
        else:
            if args.data_root is None or args.ledger_root is None:
                raise ValueError('Formal stages require explicit --data-root and shared --ledger-root')
            ensure_run(root,args.data_root,output,args.ledger_root)
            if (output/(args.stage+'.json')).exists():
                raise ValueError('Completed stage cannot be overwritten or rescored')
            from .protocol import write_exclusive,IDENTITY
            write_exclusive(args.ledger_root/IDENTITY/(args.stage+'_started.json'),
                            {'stage':args.stage,'output':str(output),'interruption_consumes_attempt':True})
            if args.stage=='physics':
                from .data import load_historical
                from .physics import run_physics
                cal=load_historical(args.data_root,root,role='calibration')
                dev=load_historical(args.data_root,root,role='development')
                if not cal['audit']['data_gate_passed']:
                    raise ValueError('Calibration quality gate failed')
                result=run_physics(cal,dev,output/'physics',config)
                _write(output/'physics.json',result)
            elif args.stage=='teacher': result=stage_teacher(root,args.data_root,output,config)
            else:
                from .runner_learning import run_learning
                result=run_learning(root,args.data_root,output,args.ledger_root,config,args.backend)
                _write(output/'learning.json',result)
            final_manifest(output)
    except (OSError,ValueError,RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    return 1 if args.stage=='verify' and not result['passed'] else 0


if __name__=='__main__': raise SystemExit(main())
