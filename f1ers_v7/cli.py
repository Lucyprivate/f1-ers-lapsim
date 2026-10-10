"""Read-only audit, new bounded run, and exact saved-result verification."""
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('audit-v7', 'run-v7', 'verify-v7'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--name', default='austria_2026_v7')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    from .experiment import run, verify, read
    if args.command == 'run-v7':
        result = run(args.root, args.name)
    elif args.command == 'verify-v7':
        result = verify(args.root, args.name)
    else:
        result = {'protocol': read(args.root/'config/v7/protocol.json'),
            'research_scope': 'public-rule subset; actual FIA ECU and true SOC pending',
            'frozen_kernel': 'f1ers_v4', 'FP2_accessed': False,
            'new_feature_schema': __import__('f1ers_v7.simulator', fromlist=['FEATURES']).FEATURES}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.command == 'verify-v7' and not result['passed'] else 0
