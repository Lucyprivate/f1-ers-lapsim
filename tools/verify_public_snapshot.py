"""Check unchanged published source bytes; never load telemetry or FP2."""
from pathlib import Path
import hashlib, json

def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'docs/PUBLICATION_SOURCE.json').read_text(encoding='utf-8'))
    included = omitted = 0
    for entry in manifest['original_files']:
        name = entry['public_path']
        if name is None:
            omitted += 1
            continue
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit('Missing or unsafe published source: '+name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise SystemExit('Changed published frozen source: '+name)
        included += 1
    prohibited = list((root/'data/raw/fia').glob('*.pdf')) + list((root/'data/processed/fia_visual_evidence').glob('*.png'))
    if prohibited:
        raise SystemExit('Full FIA documents unexpectedly included')
    print(json.dumps({'passed':True, 'original_files_checked':included,
                      'original_files_omitted':omitted, 'FP2_accessed':False,
                      'scientific_acceptance_implied':False}))

if __name__ == '__main__':
    main()
