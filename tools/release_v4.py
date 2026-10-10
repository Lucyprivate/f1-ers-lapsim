"""Build independent Git history and offline artifacts after a frozen v4 run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def git(root,*args):
    p=subprocess.run(['git','-C',str(root),*args],capture_output=True,check=True)
    return p.stdout.decode('utf-8').strip()


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if git(root,'status','--porcelain','--untracked-files=all'):
        raise ValueError('Delivery requires a clean source/results checkout')
    if git(root,'remote'):
        raise ValueError('This delivery is registered as local-only')
    experiment=json.loads((root/'results/austria_2026_v4/experiment.json').read_text(encoding='utf-8'))
    verification=json.loads((root/'results/austria_2026_v4/verification.json').read_text(encoding='utf-8'))
    if not verification.get('exact_local_replay_passed'):
        raise ValueError('Offline verification must pass before packaging')
    head=git(root,'rev-parse','HEAD')
    if git(root,'rev-parse','v4-0.4.0^{commit}')!=head:
        raise ValueError('The delivery tag must point at the reviewed result commit')
    bundle=output/'F1_ERS_Lab_v4.bundle';archive=output/'F1_ERS_Lab_v4.zip'
    if bundle.exists() or archive.exists():
        raise ValueError('Delivery filenames already exist; do not silently replace them')
    git(root,'bundle','create',str(bundle),'--all')
    bundle_check=subprocess.run(['git','-C',str(root),'bundle','verify',str(bundle)],
                                capture_output=True,text=True,check=True)
    tracked=subprocess.run(['git','-C',str(root),'ls-files','-z'],capture_output=True,check=True).stdout.decode('utf-8').split('\0')
    paths={root/name for name in tracked if name}
    for prefix in ('vendor/wheels','results/austria_2026_v4','data/v4/fp2'):
        paths.update(p for p in (root/prefix).rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    records={p.relative_to(root).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size}
             for p in sorted(paths)}
    manifest={'schema_version':4,'project':'F1 ERS Lab v4','code_commit':experiment['code_provenance']['code_commit'],
        'result_commit':head,'baseline_commit':git(root,'rev-parse','baseline-v3-0.3.0^{commit}'),
        'tag':'v4-0.4.0','branches':['main','improve/v4-dynamics'],'remote':None,
        'git_bundle_sha256':digest(bundle),'gates':experiment['gates'],
        'cross_session_status':experiment['cross_session']['status'],'training_run':False,
        'files':records,'scope':'Tracked source/history data plus offline wheels and complete v4 result artifacts; Git history is a separate bundle'}
    manifest_bytes=json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(paths):z.write(p,'f1-ers-lab-v4/'+p.relative_to(root).as_posix())
        z.writestr('f1-ers-lab-v4/DELIVERY-MANIFEST-V4.json',manifest_bytes)
    # Re-read every member and validate complete payload bytes.
    with zipfile.ZipFile(archive) as z:
        for name,record in records.items():
            if hashlib.sha256(z.read('f1-ers-lab-v4/'+name)).hexdigest()!=record['sha256']:
                raise ValueError('Archive member changed: '+name)
    result={'schema_version':4,'code_commit':manifest['code_commit'],'result_commit':head,
        'baseline_commit':manifest['baseline_commit'],'tag':manifest['tag'],
        'files':len(records),'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'zip':{'name':archive.name,'sha256':digest(archive),'bytes':archive.stat().st_size},
        'bundle':{'name':bundle.name,'sha256':digest(bundle),'bytes':bundle.stat().st_size},
        'bundle_verify_passed':True,'bundle_verify_output':bundle_check.stdout+bundle_check.stderr,
        'archive_members_verified':True,'gates':experiment['gates'],
        'independent_clone_verification':'Recorded separately after bundle clone and offline setup'}
    (output/'F1_ERS_Lab_v4_delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in (archive,bundle):
        (output/(p.name+'.sha256')).write_text(digest(p)+'  '+p.name+'\n',encoding='ascii')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.output)
