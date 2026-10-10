"""Unified finite research stages with an immutable ideal-energy learning model."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from .._core.base_rules import RuleConfig
from .._core.integrator import IntegrationState
from .._core.learning_base import _write
from .._core.metrics import score
from .._core.provenance import read, digest, git
from .data import load_historical, input_files
from .protocol import freeze, EpisodeBudget, IDENTITY, CANONICAL_RESEARCH_LEDGER, dependency_identity,python_identity
from .teacher import BlockController, qualify, plan_ers


def make_sim(data, config):
    from .environment import ResearchSimulator
    return ResearchSimulator(data['track'],data['rules'],
        rule_config=RuleConfig(tail_guard_m=config['tail_guard_m']),
        margin_s=config['geometry_margin_s'],energy_notice_s=config['energy_notice_s'])


def ensure_run(root,data_root,output,ledger):
    root,output=Path(root).resolve(),Path(output).resolve()
    if Path(ledger).resolve()!=CANONICAL_RESEARCH_LEDGER.resolve():
        raise ValueError('The canonical external research-control ledger is mandatory')
    if output.parent!=(root/'results').resolve():
        raise ValueError('Result directory must be immediately inside this repository/results')
    if (output/'freeze.json').exists():
        row=read(output/'freeze.json')
        if git(root,'status','--porcelain') or row['code_commit']!=git(root,'rev-parse','HEAD'):
            raise RuntimeError('Stage continuation requires the unchanged clean frozen commit')
        if row['data_sha256']!=input_files(data_root):
            raise RuntimeError('Private historical input hashes changed')
        if row['dependencies']!=dependency_identity():
            raise RuntimeError('Installed dependency versions or RECORD hashes changed')
        if row['python']!=python_identity():
            raise RuntimeError('Frozen Python identity changed')
        return row
    return freeze(root,data_root,output,ledger)


def stage_teacher(root,data_root,output,config):
    from .environment import numerical_audit
    cal=load_historical(data_root,root,role='calibration')
    if not cal['audit']['data_gate_passed']:
        raise ValueError('Calibration alignment/boundary quality gate failed')
    sim=make_sim(cal,config)
    speed=config['cold_initial_speed_mps']
    baseline,laps=qualify(sim,BlockController(np.zeros(16)),output/'baseline',speed)
    _write(output/'baseline/score_calibration.json',score(laps[-1],cal,'calibration'))
    if not baseline['passed']:
        row={'status':'baseline_physical_failure','learning_allowed':False,'qualification':baseline}
        _write(output/'teacher.json',row);return row
    numeric=numerical_coverage(sim,laps,output/'baseline/numerics')
    _write(output/'baseline/numerics.json',numeric)
    if not numeric['passed']:
        row={'status':'baseline_numerical_failure','learning_allowed':False,'numerics':numeric}
        _write(output/'teacher.json',row);return row
    initial=IntegrationState(0.,speed,0.,2.,0.,0.)
    plan=plan_ers(initial,None,sim,np.linspace(0,sim.length_m,17),
        {'output':output/'teacher_search','maximum':600})
    candidate,teacher_laps=qualify(sim,BlockController(plan.actions),output/'teacher_certification',speed) if plan.actions else (baseline,laps)
    numerical=numerical_coverage(sim,teacher_laps,output/'teacher_certification/numerics') if candidate['passed'] else {'passed':False}
    _write(output/'teacher_certification/numerics.json',numerical)
    gap=baseline['formal_objective_s']-candidate['formal_objective_s'] if candidate['passed'] else None
    improved=bool(gap is not None and gap>.1 and numerical['passed'])
    selected=plan.actions if improved else [0.]*16
    selected_laps=teacher_laps if improved else laps
    # Learning boundary pool is frozen from the baseline; it is never replaced
    # by a learner's terminal boundary or a new physical model candidate.
    last=laps[-1]
    pool={'cold':{'speed_mps':speed,'energy_mj':2.,'carried_rule_state':None},
          'warm':{'speed_mps':last['terminal_speed_mps'],
                  'energy_mj':last['summary']['energy_end_mj'],
                  'carried_rule_state':last['rule_certificate']['terminal_boundary_state']}}
    _write(output/'boundary_pool.json',pool)
    for i,lap in enumerate(selected_laps): _write(output/f'bc_teacher/lap_{i+1}.json.gz',lap)
    row={'status':'qualified_model_teacher','learning_allowed':True,
         'baseline_objective_s':baseline['formal_objective_s'], 'candidate':candidate,
         'candidate_gap_s':gap,'candidate_numerics_passed':numerical['passed'],
         'model_improvement_demonstrated':improved,'selected_actions':selected,
         'exact_search_evaluations':plan.search_evaluations,
         'final_certification_and_numerical_audits_outside_search_cap':True,
         'constant_BC_is_analytical_clone':not improved, 'real_car_gain_claimed':False}
    _write(output/'teacher.json',row)
    # Development observations enter only after the teacher action selection.
    dev=load_historical(data_root,root,role='development')
    _write(output/'baseline/score_development.json',score(laps[-1],dev,'development'))
    _write(output/'teacher_certification/score_development.json',score(selected_laps[-1],dev,'development'))
    return row


def numerical_coverage(sim,laps,output):
    from .environment import numerical_audit
    rows={}
    for index,name in ((0,'cold_first'),(-1,'carried_late')):
        rows[name]=numerical_audit(sim,laps[index],Path(output)/name)
    return {'passed':bool(all(r['passed'] for r in rows.values())),
            'coverage':'first cold and final actual carried lap', 'boundaries':rows,
            'output_resampling_and_internal_convergence_separate':True}


def final_manifest(output):
    output=Path(output)
    files={p.relative_to(output).as_posix():digest(p) for p in sorted(output.rglob('*'))
           if p.is_file() and p!=output/'manifest.json'}
    value={'schema_version':1,'identity':IDENTITY,'files':files,'FP2_accessed':False}
    _write(output/'manifest.json',value)
    return value


def verify_saved(root,data_root,output,*,replay=False):
    manifest=read(output/'manifest.json');frozen=read(output/'freeze.json')
    canonical_claim=read(CANONICAL_RESEARCH_LEDGER/IDENTITY/'run_claim.json')
    claim_equal=canonical_claim==frozen
    import subprocess,hashlib
    commit_tree_equal=all(hashlib.sha256(subprocess.check_output(
        ['git','show',frozen['code_commit']+':'+name],cwd=root)).hexdigest()==sha
        for name,sha in frozen['tracked_sha256'].items())
    problems=[]
    for name,expected in manifest['files'].items():
        path=(output/name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or digest(path)!=expected:
            problems.append(name)
    data_equal=frozen['data_sha256']==input_files(data_root)
    source_equal=all(digest(Path(root)/name)==sha for name,sha in frozen['tracked_sha256'].items()
        if name.endswith('.py') or name.startswith('config/'))
    runtime_equal=(frozen['dependencies']==dependency_identity() and
                   frozen['python']==python_identity())
    replay_report=None
    if replay and data_equal and source_equal and runtime_equal:
        from tempfile import TemporaryDirectory
        data=load_historical(data_root,root,role='calibration')
        sim=make_sim(data,frozen['config'])
        with TemporaryDirectory(dir=output.parent) as tmp:
            replay_report,_=qualify(sim,BlockController([0.]*16),Path(tmp),frozen['config']['cold_initial_speed_mps'])
        expected=read(output/'baseline/qualification.json')
        replay_report={'passed':replay_report==expected,'six_carry_recomputed':True,
                       'formal_objective_s':replay_report['formal_objective_s']}
    return {'passed':bool(not problems and claim_equal and commit_tree_equal and data_equal and source_equal and
                         (not replay or replay_report and replay_report['passed'])),
            'artifact_hash_failures':problems,'private_input_hashes_equal':data_equal,
            'computational_source_hashes_equal':source_equal,'baseline_replay':replay_report,
            'canonical_run_claim_equal':claim_equal,'frozen_git_tree_equal':commit_tree_equal,
            'runtime_identity_equal':runtime_equal,
            'optimization_and_training_repeated':False,'saved_scores_read_only':True,
            'FP2_accessed':False,'original_code_commit':frozen['code_commit']}
