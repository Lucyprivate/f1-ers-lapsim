from pathlib import Path
import argparse
import json
from f1ers_v4.artifacts import write_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description='Independent sustainable-control v6 research')
    parser.add_argument('command',choices=['audit-v6','run-v6','verify-v6'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--name',default='austria_2026_v6')
    parser.add_argument('--offline',action='store_true',help='All v6 commands are already local and offline')
    args=parser.parse_args(); root=args.root.resolve()
    if args.command=='run-v6':
        from .experiment import run
        result=run(root,args.name)
        print(json.dumps({'selected':result['selected'],'gates':result['gates'],
            'model_gain_s':result['tested_model_gain_s']},indent=2))
    elif args.command=='verify-v6':
        from .verification import verify
        print(json.dumps(verify(root,args.name),indent=2))
    else:
        from .experiment import provenance
        result={'provenance':provenance(root,False),'frozen_kernel':'f1ers_v4.vehicle',
            'old_ml_controller_executed_in_v4_v5':False,
            'confirmed_causes':['ICE-only target acceleration suppresses power sensitivity',
                'action sign confused acceleration with wheel demand','training and scoring used different execution paths',
                'free energy/DC/period reset makes apparent multi-lap gains unreliable'],
            'new_test_opened':False,'full_fia_verified':False}
        write_json(root/'results'/args.name/'audit.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))

