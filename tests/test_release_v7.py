"""Synthetic release preflights; never package an active real experiment."""
from copy import deepcopy
import hashlib
import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

try:
    import release_v7 as release
except ImportError:
    from tools import release_v7 as release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent
        base = base.parent/'.cache/v7-release-tests' if base.name == 'tests' else base/'test_tmp'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'repo'
        self.root.mkdir()
        self.command('init', '-q')
        self.command('config', 'core.autocrlf', 'false')
        self.command('config', 'user.name', 'Release fixture')
        self.command('config', 'user.email', 'release@local.invalid')
        self.put('.gitignore', b'results/\nvendor/\n')
        self.put('README.md', b'unchanged source\n')
        self.put('model.py', b'PASS = True\n')
        self.put('config/v7/protocol.json', b'{"known_geometry_only":true}\n')
        self.command('add', '.')
        self.command('commit', '-qm', 'calculation source')
        self.code = self.command('rev-parse', 'HEAD')
        self.command('tag', 'calculation-fixture')
        self.put('vendor/wheels/runtime.whl', b'offline dependency')
        self.put('vendor/reporting-wheels/report.whl', b'offline reporting')
        self.expectations = {}
        self.gates = {'accuracy': False, 'formal_release': False, 'cold_and_warm_control': True}
        for index, prefix in enumerate(release.EXPERIMENT_ROOTS):
            letter = chr(ord('A')+index)
            winner = {'name': 'certified numerical fixture', 'qualification': {'passed': True, 'physical_laps': 6}}
            experiment = {'schema_version': 7, 'source_commit': self.code, 'winner': winner,
                'gates': self.gates, 'FP2_accessed': False, 'full_FIA_compliance_proven': False,
                'inherited_ignored_inputs_sha256': {}}
            freeze = {'code_commit': self.code, 'tracked_input_sha256': {
                name: release.digest(self.root/name) for name in ('model.py', 'config/v7/protocol.json')},
                'dependency_wheels_sha256': {'vendor/wheels/runtime.whl': release.digest(self.root/'vendor/wheels/runtime.whl')}}
            self.put(prefix+'/experiment.json', experiment)
            self.put(prefix+'/freeze.json', freeze)
            self.put(prefix+'/frozen_protocol.json', (self.root/'config/v7/protocol.json').read_bytes())
            self.put(prefix+'/selection.json', {'source_commit': self.code, 'winner': winner})
            for name in ('selected_calibration.json', 'development_score.json', 'numerical_audit.json'):
                self.put(prefix+'/'+name, {'fixture': True})
            files = {p.relative_to(self.root/prefix).as_posix(): release.digest(p)
                     for p in (self.root/prefix).rglob('*') if p.is_file()}
            self.put(prefix+'/manifest.json', {'source_commit': self.code, 'files': files})
            seal = release.digest(self.root/prefix/'manifest.json')
            receipts = {}
            for kind in ('summary', 'verification'):
                name = f'summaries/v7/{letter}_{kind}.json'
                self.put(name, {'source_commit': self.code, 'results_manifest_sha256': seal,
                    'passed': True, 'FP2_accessed': False, 'gates': self.gates})
                receipts[kind] = {'path': name, 'sha256': release.digest(self.root/name)}
            self.expectations[prefix] = {'source_commit': self.code, 'manifest_sha256': seal, 'receipts': receipts}
        self.command('add', 'summaries')
        self.command('commit', '-qm', 'independent result receipts')
        self.head = self.command('rev-parse', 'HEAD')
        self.command('tag', 'v7-fixture')
        self.put(release.DELIVERY_ROOT+'/report.html', b'<html><body>Diagnostic result</body></html>')
        self.put(release.DELIVERY_ROOT+'/audit.json', {'FP2_accessed': False})
        self.put(release.DELIVERY_ROOT+'/manifest.json', {'source_commit': self.head, 'result_commit': self.head,
            'files': {name: release.digest(self.root/release.DELIVERY_ROOT/name) for name in ('report.html', 'audit.json')}})
        self.delivery = {'source_commit': self.head, 'manifest_sha256': release.digest(self.root/release.DELIVERY_ROOT/'manifest.json')}
        for i in range(73):
            self.put(release.R7_ROOT+f'/artifact_{i:03d}.json', {'frozen_fixture': i})
        self.put(release.R7_ROOT+'/verification.json', {'source_commit': self.code, 'passed': True, 'FP2_accessed': False})
        self.put(release.R7_ROOT+'/freeze.json', {'code_commit': self.code,
            'source_sha256': {'model.py': release.digest(self.root/'model.py')}, 'data_sha256': {}})
        self.put(release.R7_ROOT+'/experiment.json', {'source_commit': self.code})
        self.put(release.R7_ROOT+'/manifest.json', {'source_commit': self.code,
            'files': {p.relative_to(self.root/release.R7_ROOT).as_posix(): release.digest(p)
                      for p in (self.root/release.R7_ROOT).rglob('*') if p.is_file()}})
        self.put(release.SINGLE_ARTIFACTS[0], b'frozen raw checkpoint fixture')
        self.prior_values = {'R7_SOURCE_COMMIT': self.code,
            'R7_MANIFEST_SHA256': release.digest(self.root/release.R7_ROOT/'manifest.json'),
            'R7_CHECKPOINT_SHA256': release.digest(self.root/release.SINGLE_ARTIFACTS[0])}
        self.output = Path(self.temp.name)/'published'

    def command(self, *args):
        return subprocess.run(['git', '-C', str(self.root), *args], capture_output=True, check=True).stdout.decode().strip()

    def put(self, name, payload):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(payload, sort_keys=True).encode() if isinstance(payload, dict) else payload)
        return path

    def build(self, **overrides):
        arguments = {'expected_head': self.head, 'tag': 'v7-fixture', 'experiments': self.expectations,
                     'delivery': self.delivery, 'results_sealed': True}
        arguments.update(overrides)
        with patch.multiple(release, **self.prior_values):
            return release.build(self.root, self.output, **arguments)

    def reseal(self, prefix, *, exclude=()):
        files = {p.relative_to(self.root/prefix).as_posix(): release.digest(p)
                 for p in (self.root/prefix).rglob('*') if p.is_file()
                 and p.relative_to(self.root/prefix).as_posix() not in {'manifest.json', *exclude}}
        self.put(prefix+'/manifest.json', {'source_commit': self.code, 'files': files})
        seal = release.digest(self.root/prefix/'manifest.json')
        self.expectations[prefix]['manifest_sha256'] = seal
        for spec in self.expectations[prefix]['receipts'].values():
            receipt = release.read_json(self.root/spec['path'])
            receipt['results_manifest_sha256'] = seal
            self.put(spec['path'], receipt)
            spec['sha256'] = release.digest(self.root/spec['path'])

    def c_bindings(self):
        prior, prefix = release.EXPERIMENT_ROOTS[1:3]
        selection = release.read_json(self.root/prior/'selection.json')
        controller = {'kind': 'phase_blocks_with_actual_braking', 'values': [0.5], 'action_space': 'mode_aware'}
        selection['winner']['controller'] = controller
        self.put(prior+'/selection.json', selection)
        self.reseal(prior)
        seal = release.digest(self.root/prior/'manifest.json')
        selection_sha = release.digest(self.root/prior/'selection.json')
        config = {'prior_name': 'austria_2026_v7B', 'prior_source_commit': self.code,
                  'prior_manifest_sha256': seal, 'prior_selection_sha256': selection_sha}
        self.put(prefix+'/frozen_protocol.json', config)
        experiment = release.read_json(self.root/prefix/'experiment.json')
        experiment.update(prior_source_commit=self.code, prior_manifest_sha256=seal)
        self.put(prefix+'/experiment.json', experiment)
        for name in ('manifest.json', 'selection.json'):
            self.put(prefix+'/inputs/'+name, (self.root/prior/name).read_bytes())
        self.put(prefix+'/inputs/input_bindings.json', {'source_controller_spec': controller,
            'portable_controller_spec': controller, 'controller_weights_changed': False,
            'prior_input_sha256': {'manifest.json': seal, 'selection.json': selection_sha}})
        self.reseal(prefix, exclude=('inputs/manifest.json',))
        return prefix, release.read_json(self.root/prefix/'manifest.json'), experiment

    def d_worker(self, directory='D4/2026', *, completed=2, attempted=3):
        prefix = release.EXPERIMENT_ROOTS[3]
        self.put(prefix+'/'+directory+'/worker_started.json', {'episodes_requested': 20})
        for index in range(1, attempted+1):
            self.put(prefix+'/'+directory+f'/episode_{index:03d}_started.json', {'episode': index})
        for index in range(1, completed+1):
            payload = {'dones': [True], 'result': {'summary': {'completed_lap': index == 1}}}
            self.put(prefix+'/'+directory+f'/training_episode_{index:03d}.json.gz',
                     gzip.compress(json.dumps(payload).encode()))
        self.put(prefix+'/'+directory+'/worker_receipt.json', {'episodes_requested': 20,
            'episodes_attempted': attempted, 'episodes_completed': completed,
            'physical_full_laps_completed': 1 if completed else 0,
            'completed_includes_executor_terminal_failure_episodes': True})
        return prefix, directory

    def test_d_worker_exception_without_metadata_retains_actual_episode_counts(self):
        prefix, directory = self.d_worker()
        experiment = release.read_json(self.root/prefix/'experiment.json')
        experiment.update(actual_new_SAC_episodes_requested=20,
                          actual_new_SAC_episodes_attempted=3, actual_new_SAC_episodes_completed=2)
        self.put(prefix+'/experiment.json', experiment)
        self.reseal(prefix)
        value = release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))
        self.assertEqual(value['actual_new_SAC_episodes'], 2)
        self.assertEqual(value['actual_new_SAC_episodes_attempted'], 3)
        self.assertFalse(value['saved_SAC_case_counts'][0]['sac_metadata_retained'])
        self.assertEqual(value['saved_SAC_case_counts'][0]['physical_full_laps_completed'], 1)

    def test_d_worker_counters_are_verified_against_terminal_trajectory_files(self):
        prefix, directory = self.d_worker()
        receipt = release.read_json(self.root/prefix/directory/'worker_receipt.json')
        receipt['episodes_completed'] = 3
        self.put(prefix+'/'+directory+'/worker_receipt.json', receipt)
        self.reseal(prefix)
        with self.assertRaisesRegex(ValueError, 'differ from saved episode receipts'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_c_frozen_copy_bindings_allow_only_anchored_nested_prior_manifest(self):
        prefix, manifest, experiment = self.c_bindings()
        extra = release.validate_bound_c_inputs(self.root, prefix, manifest, experiment)
        self.assertEqual(extra, ['inputs/manifest.json'])
        release.validate_inventory(self.root, prefix, manifest, extra)
        self.put(prefix+'/inputs/other_manifest.json', b'not bound')
        with self.assertRaisesRegex(ValueError, 'unlisted files'):
            release.validate_inventory(self.root, prefix, manifest, extra)

    def test_c_copy_and_prior_identities_cannot_be_substituted(self):
        prefix, manifest, experiment = self.c_bindings()
        (self.root/prefix/'inputs/manifest.json').write_bytes(b'changed copied B manifest')
        with self.assertRaisesRegex(ValueError, 'copied B input changed'):
            release.validate_bound_c_inputs(self.root, prefix, manifest, experiment)
        experiment['prior_manifest_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'not its fixed B manifest'):
            release.validate_bound_c_inputs(self.root, prefix, manifest, experiment)

    def test_unsealed_and_wrong_tag_refuse_before_output_creation(self):
        with self.assertRaisesRegex(ValueError, 'runner exits'):
            self.build(results_sealed=False)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, 'HEAD and release tag'):
            self.build(tag='calculation-fixture')
        self.assertFalse(self.output.exists())

    def test_all_four_fixed_calculation_identities_are_required(self):
        bad = deepcopy(self.expectations)
        bad.pop(release.EXPERIMENT_ROOTS[3])
        with self.assertRaisesRegex(ValueError, 'A/B/C/D'):
            self.build(experiments=bad)
        bad = deepcopy(self.expectations)
        bad[release.EXPERIMENT_ROOTS[0]]['manifest_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'manifest SHA256 mismatch'):
            self.build(experiments=bad)
        self.assertFalse(self.output.exists())

    def test_dirty_or_remote_repository_is_refused(self):
        self.put('unexpected.py', b'not committed')
        with self.assertRaisesRegex(ValueError, 'clean local-only'):
            self.build()
        (self.root/'unexpected.py').unlink()
        self.command('remote', 'add', 'fixture', 'https://example.invalid/unused.git')
        with self.assertRaisesRegex(ValueError, 'clean local-only'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_tracked_receipt_sha_and_calculation_identity_are_checked(self):
        prefix = release.EXPERIMENT_ROOTS[0]
        spec = self.expectations[prefix]['receipts']['verification']
        self.put(spec['path'], {'source_commit': '0'*40, 'passed': True, 'FP2_accessed': False,
                               'results_manifest_sha256': self.expectations[prefix]['manifest_sha256']})
        with self.assertRaisesRegex(ValueError, 'receipt changed'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))
        self.expectations[prefix]['receipts']['verification']['sha256'] = release.digest(self.root/spec['path'])
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_summary_cannot_turn_failed_fidelity_into_a_pass(self):
        prefix = release.EXPERIMENT_ROOTS[0]
        spec = self.expectations[prefix]['receipts']['summary']
        receipt = release.read_json(self.root/spec['path'])
        receipt['gates'] = {'accuracy': True}
        self.put(spec['path'], receipt)
        spec['sha256'] = release.digest(self.root/spec['path'])
        with self.assertRaisesRegex(ValueError, 'summary gates differ'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_frozen_inputs_require_both_current_bytes_and_code_tree_anchor(self):
        freeze = release.read_json(self.root/release.EXPERIMENT_ROOTS[0]/'freeze.json')
        self.put('model.py', b'PASS = False\n')
        with self.assertRaisesRegex(ValueError, 'calculation Git tree'):
            release.validate_frozen_inputs(self.root, freeze, self.code)
        freeze['tracked_input_sha256']['model.py'] = release.digest(self.root/'model.py')
        with self.assertRaisesRegex(ValueError, 'calculation Git tree'):
            release.validate_frozen_inputs(self.root, freeze, self.code)

    def test_all_manifest_files_and_unlisted_directory_members_are_checked(self):
        prefix = release.EXPERIMENT_ROOTS[1]
        self.put(prefix+'/numerical_audit.json', b'changed artifact')
        with self.assertRaisesRegex(ValueError, 'calculation artifact changed'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))
        prefix = release.EXPERIMENT_ROOTS[2]
        self.put(prefix+'/unexpected.json', b'unsealed')
        with self.assertRaisesRegex(ValueError, 'unlisted files'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_dependency_wheels_and_missing_winner_are_not_implicit_passes(self):
        freeze = release.read_json(self.root/release.EXPERIMENT_ROOTS[0]/'freeze.json')
        self.put('vendor/wheels/runtime.whl', b'changed dependency')
        with self.assertRaisesRegex(ValueError, 'dependency wheel changed'):
            release.validate_frozen_inputs(self.root, freeze, self.code)
        prefix = release.EXPERIMENT_ROOTS[1]
        experiment = release.read_json(self.root/prefix/'experiment.json')
        experiment['winner'] = None
        self.put(prefix+'/experiment.json', experiment)
        with self.assertRaisesRegex(ValueError, 'certified winner'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_report_has_independent_fixed_manifest_and_result_commit(self):
        self.put(release.DELIVERY_ROOT+'/manifest.json', {'source_commit': self.head, 'result_commit': self.code,
            'files': {'report.html': release.digest(self.root/release.DELIVERY_ROOT/'report.html')}})
        self.delivery['manifest_sha256'] = release.digest(self.root/release.DELIVERY_ROOT/'manifest.json')
        with self.assertRaisesRegex(ValueError, 'result commit identities'):
            release.validate_delivery(self.root, self.delivery, self.head)
        self.put(release.DELIVERY_ROOT+'/manifest.json', {'source_commit': self.code, 'result_commit': self.head,
            'files': {'report.html': release.digest(self.root/release.DELIVERY_ROOT/'report.html')}})
        self.delivery.update(source_commit=self.code,
            manifest_sha256=release.digest(self.root/release.DELIVERY_ROOT/'manifest.json'))
        with self.assertRaisesRegex(ValueError, 'result commit identities'):
            release.validate_delivery(self.root, self.delivery, self.head)

    def test_global_actual_sac_budget_is_enforced_before_publication(self):
        cases = [{'source_commit': self.code, 'actual_new_SAC_episodes': count}
                 for count in (40, 40, 41, 0)]
        with patch.object(release, 'validate_experiment', side_effect=cases):
            with self.assertRaisesRegex(ValueError, '120-episode continuation budget'):
                self.build()
        self.assertFalse(self.output.exists())

    def test_exact_collection_excludes_broad_v6_and_reports_tree(self):
        self.put('results/austria_2026_v6/R1/old.json', b'excluded')
        self.put('results/austria_2026_v6/R4/2027/bc_metadata.json', b'excluded sibling')
        self.put('results/reports/report.html', b'excluded report tree')
        self.put(release.R7_ROOT+'/results.md', b'unsealed historical prose remains outside the package')
        names = release.collect(self.root, release.tracked_files(self.root))
        for name in ('results/austria_2026_v6/R1/old.json', 'results/austria_2026_v6/R4/2027/bc_metadata.json',
                     'results/reports/report.html'):
            self.assertNotIn(name, names)
        self.assertNotIn(release.R7_ROOT+'/results.md', names)
        for prefix in (*release.EXPERIMENT_ROOTS, release.DELIVERY_ROOT, release.R7_ROOT):
            self.assertTrue(any(name.startswith(prefix+'/') for name in names))
        self.assertIn(release.SINGLE_ARTIFACTS[0], names)

    def test_forbidden_arrays_nested_wheels_and_duplicate_json_are_refused(self):
        for name in (release.EXPERIMENT_ROOTS[0]+'/late_arrays.npz', release.DELIVERY_ROOT+'/fp2/raw.json',
                     'vendor/wheels/nested/runtime.whl'):
            self.put(name, b'forbidden')
            with self.assertRaises(ValueError):
                release.collect(self.root, release.tracked_files(self.root))
            (self.root/name).unlink()
        self.put('duplicate.json', b'{"files":{},"files":{"a":"b"}}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            release.read_json(self.root/'duplicate.json')

    def test_zero_sac_is_not_mistaken_for_completed_training(self):
        prefix = release.EXPERIMENT_ROOTS[1]
        experiment = release.read_json(self.root/prefix/'experiment.json')
        experiment['training_status'] = 'bounded_cold_and_warm_experiments_completed'
        self.put(prefix+'/experiment.json', experiment)
        self.reseal(prefix)
        with self.assertRaisesRegex(ValueError, 'Zero SAC episodes'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_actual_sac_count_is_derived_from_saved_noninherited_metadata(self):
        prefix, directory = self.d_worker(completed=17, attempted=17)
        self.put(prefix+'/D4/2026/sac_metadata.json', {'episodes_requested': 20, 'episodes_completed': 17})
        self.put(prefix+'/inputs/sac_metadata.json', {'episodes_requested': 20, 'episodes_completed': 20})
        self.reseal(prefix)
        value = release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))
        self.assertEqual(value['actual_new_SAC_episodes'], 17)
        self.assertEqual(value['actual_new_SAC_episodes_requested'], 20)
        self.assertEqual(len(value['saved_SAC_case_counts']), 1)
        experiment = release.read_json(self.root/prefix/'experiment.json')
        experiment['actual_new_SAC_episodes_completed'] = 120
        self.put(prefix+'/experiment.json', experiment)
        self.reseal(prefix)
        with self.assertRaisesRegex(ValueError, 'episode counter differs'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_nested_copied_manifest_is_verified_without_rewriting_original_inventory(self):
        prefix = release.EXPERIMENT_ROOTS[0]
        copied = self.put(prefix+'/inputs/manifest.json', b'{"prior_inventory":"fixed"}')
        experiment = release.read_json(self.root/prefix/'experiment.json')
        experiment['inherited_ignored_inputs_sha256'] = {release.R7_ROOT+'/manifest.json': release.digest(copied)}
        self.put(prefix+'/experiment.json', experiment)
        self.reseal(prefix, exclude=('inputs/manifest.json',))
        value = release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))
        self.assertEqual(value['source_commit'], self.code)
        copied.write_bytes(b'changed copied inventory')
        with self.assertRaisesRegex(ValueError, 'inherited input copy changed'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_inherited_fp2_name_cannot_be_hidden_inside_inputs(self):
        prefix = release.EXPERIMENT_ROOTS[0]
        copied = self.put(prefix+'/inputs/car_data.json', b'forbidden data')
        experiment = release.read_json(self.root/prefix/'experiment.json')
        experiment['inherited_ignored_inputs_sha256'] = {'data/v4/fp2/raw/car_data.json': release.digest(copied)}
        self.put(prefix+'/experiment.json', experiment)
        self.reseal(prefix)
        with self.assertRaisesRegex(ValueError, 'cannot be hidden'):
            release.validate_experiment(self.root, prefix, self.expectations[prefix], self.head, release.tracked_files(self.root))

    def test_complete_build_verifies_members_bundle_history_and_independent_result_commit(self):
        result = self.build()
        self.assertTrue(result['archive_members_verified'])
        self.assertTrue(result['bundle_verify_passed'])
        self.assertFalse(result['FP2_arrays_included'])
        self.assertEqual(result['result_commit'], self.head)
        self.assertNotEqual(result['result_commit'], self.code)
        self.assertTrue(all(c['recorded_scientific_gates']['accuracy'] is False for c in result['calculations']))
        with zipfile.ZipFile(self.output/'F1_ERS_Lab_v7.zip') as package:
            self.assertEqual(len(package.namelist()), result['file_count'])
            for name, record in result['files'].items():
                payload = package.read(release.ZIP_PREFIX+name)
                self.assertEqual(len(payload), record['bytes'])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), record['sha256'])
        heads = self.command('bundle', 'list-heads', str(self.output/'F1_ERS_Lab_v7.bundle'))
        self.assertIn('refs/tags/calculation-fixture', heads)
        self.assertIn('refs/tags/v7-fixture', heads)
        for filename in ('F1_ERS_Lab_v7.zip', 'F1_ERS_Lab_v7.bundle', 'F1_ERS_Lab_v7_delivery.json'):
            sidecar = (self.output/(filename+'.sha256')).read_text(encoding='ascii')
            self.assertTrue(sidecar.startswith(release.digest(self.output/filename)))
        with self.assertRaisesRegex(ValueError, 'must not be overwritten'):
            self.build()


if __name__ == '__main__':
    unittest.main()
