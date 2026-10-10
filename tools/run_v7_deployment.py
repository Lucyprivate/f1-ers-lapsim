"""Guarded Windows multiprocessing entry for the independent frozen D loop."""
from pathlib import Path
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--name', default='austria_2026_v7D')
    parser.add_argument('--prior-name', default='austria_2026_v7C')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.repo.resolve()
    sys.path.insert(0, str(root))
    from f1ers_v7.deployment_experiment import run, verify
    result = verify(root, args.name) if args.verify else run(root, args.prior_name, args.name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.verify and not result['passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
