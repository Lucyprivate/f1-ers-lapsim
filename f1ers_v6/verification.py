"""Hash all saved experiments; rerun the deployed winner without retraining."""
from pathlib import Path
import gzip
import json
from f1ers_v4.artifacts import read_json,write_json,sha256
from f1ers_v4.data import load_local_dataset
from f1ers_v4.vehicle import VehicleParamsV4
from f1ers_v3.verification import assert_diagnostic_equal
from .simulator import SimulatorV6
from .rules import RuleConfig
from .teacher import BlockController
from .learning import load_policy
from .metrics import score


def load_controller(root,output,experiment):
    spec=experiment['controller']; space=experiment['selected']['space']
    if spec['kind']=='neutral': return None
    if spec['kind']=='phase_blocks_with_actual_braking': return BlockController(spec['values'],space)
    return load_policy(output/spec['checkpoint'],expected_action_space=space,expected_features=SimulatorV6.features)


def verify(root,name='austria_2026_v6',save=True):
    root=Path(root).resolve(); output=root/'results'/name
    manifest=read_json(output/'manifest.json'); experiment=read_json(output/'experiment.json')
    for relative,digest in manifest['files'].items():
        target=(output/relative).resolve()
        if not target.is_relative_to(output) or sha256(target)!=digest:
            raise ValueError('Artifact integrity failure: '+relative)
    for section in ('source_sha256','data_sha256'):
        for relative,digest in experiment['provenance'][section].items():
            if sha256(root/relative)!=digest: raise ValueError('Frozen input changed: '+relative)
    dataset=load_local_dataset(root)
    sim=SimulatorV6(dataset['track'],dataset['rules'],VehicleParamsV4(**experiment['vehicle']),
        rule_config=RuleConfig(**experiment['rule_config']))
    controller=load_controller(root,output,experiment)
    with gzip.open(output/'selected_trajectory.json.gz','rt',encoding='utf-8') as stream: expected=json.load(stream)
    actual=sim.simulate(controller,initial_speed_mps=expected['summary']['speed_start_mps'],
        initial_energy_mj=expected['internal_trace']['energy_mj'][0],
        carried_rule_state=expected['summary']['initial_rule_state'],action_space=experiment['selected']['space'])
    for key in ('s_m','speed_mps','dt_s','terminal_state','requested_actions','rule_certificate'):
        assert_diagnostic_equal(actual[key],expected[key],'v6/'+key)
    for role in ('calibration','development'):
        assert_diagnostic_equal(score(actual,dataset,role),experiment[role],'v6/'+role)
    result={'passed':True,'artifact_files_verified':len(manifest['files']),
        'selected_trajectory_exactly_reproduced':True,'calibration_and_development_reproduced':True,
        'stochastic_training_reexecuted':False,'teacher_search_reexecuted':False,
        'all_training_and_failure_logs_hash_verified':True,'new_test_opened':False,
        'recorded_gates':experiment['gates'],'full_fia_verified':False}
    if save:write_json(output/'verification.json',result)
    return result
