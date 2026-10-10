"""Audit the current 34-feature controller; guarded current run/verification."""
from pathlib import Path
import argparse
import json


def require_private_inputs(root, command, name, prior_name):
    """No implicit acquisition, synthetic replacement or historical re-scoring."""
    if command == 'run-v7':
        required = [root/'results'/prior_name/'manifest.json',
                    root/'results'/prior_name/'experiment.json',
                    root/'results/austria_2026_v7/experiment.json',
                    root/'results/austria_2026_v7B/experiment.json',
                    root/'data/processed/dataset.json',
                    root/'data/v3/local/registration.json']
    else:
        required = [root/'results'/name/'manifest.json',
                    root/'results'/name/'freeze.json',
                    root/'results'/name/'experiment.json']
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise ValueError('Private frozen prerequisites are missing: '+', '.join(missing)+
                         '. This source-only checkout cannot reproduce the archived experiment. '
                         'See docs/REPRODUCIBILITY.md; no data are acquired automatically.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit-v7', 'run-v7', 'verify-v7'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--name', default='austria_2026_v7D')
    parser.add_argument('--prior-name', default='austria_2026_v7C')
    parser.add_argument('--offline', action='store_true', help='No automatic data acquisition is supported.')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    from .deployment_experiment import result_directory, validate_config
    from ._core.provenance import read
    from .deploy_only import FEATURES, PROTOCOL
    try:
        output = result_directory(root, args.name)
        result_directory(root, args.prior_name)
        config = validate_config(read(root/'config/v7/deployment.json'))
        if args.command == 'audit-v7':
            result = {'protocol': config, 'execution_protocol': PROTOCOL,
                      'feature_count': len(FEATURES), 'features': list(FEATURES),
                      'research_scope': 'Public-rule subset; full FIA ECU and true SOC pending',
                      'kernel': 'f1ers_v7._core.integrator', 'FP2_accessed': False,
                      'new_training_or_selection': False,
                      'source_layout': 'Current package with consolidated internal dependencies'}
        else:
            if args.command == 'run-v7' and output.exists():
                raise ValueError('Existing experiment cannot be overwritten or rescored')
            require_private_inputs(root, args.command, args.name, args.prior_name)
            from .deployment_experiment import run, verify
            result = (run(root, args.prior_name, args.name) if args.command == 'run-v7'
                      else verify(root, args.name))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.command == 'verify-v7' and not result['passed'] else 0
