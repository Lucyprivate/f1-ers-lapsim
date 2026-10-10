"""Explicit, read-only historical inputs with role-specific views and no FP2 API."""
from pathlib import Path
import numpy as np
from .._core import input_channels as channels
from .._core.geometry import build_track, remap_rules
from .._core.input_io import read_json, DataUnavailable
from .._core.provenance import digest

ROLES = {'calibration': (3, 4, 5, 6), 'development': (7, 9)}


def input_files(data_root):
    root = Path(data_root).resolve()
    names = ['data/processed/dataset.json', 'data/processed/track.npz', 'data/v3/local/registration.json',
             'data/raw/openf1/laps.json', 'data/raw/openf1/car_data_63_laps3to9.json',
             'data/raw/openf1/location_63_laps3to9.json']
    names += [p.relative_to(root).as_posix() for p in sorted((root/'data/raw/tum').glob('*.csv'))]
    return {name: digest(root/name) for name in names}


def load_historical(data_root, public_root, *, role='calibration', ds_m=5.):
    if role not in ROLES:
        raise DataUnavailable('Only calibration or development historical views are supported')
    root = Path(data_root).resolve()
    document, metadata, arrays = channels._local_source(root)
    metadata = [row for row in metadata if row['role'] == role]
    if tuple(int(row['lap_number']) for row in metadata) != ROLES[role]:
        raise DataUnavailable('Historical role/lap identity changed')
    registration = read_json(root/'data/v3/local/registration.json')
    if registration.get('fit_role') != 'calibration' or registration.get('fit_laps') != [3,4,5,6]:
        raise DataUnavailable('A calibration-only frozen registration is required')
    track = build_track(root, 'spline', ds_m, registration)
    # The shared historical source contains 3-9, but only this selected role is
    # processed. No new-session directory is scanned or materialized.
    laps = [channels.process_lap(row, arrays, track, registration) for row in metadata]
    audit = {'role': role, 'registration': registration, 'geometry': track['geometry_audit'],
             'test_arrays_loaded': False, 'read_only_external_inputs': True,
             'data_gate_passed': all(l['alignment_audit']['grid_bracketed'] and
                 .96 <= l['alignment_audit']['projected_span_fraction'] <= 1.04 for l in laps),
             'alignment_per_lap': [{'lap_number': l['lap_number'], **l['alignment_audit']} for l in laps]}
    rules = remap_rules(read_json(Path(public_root)/'config/rules.json'), track)
    return {'track': track, 'rules': rules, 'laps': laps, 'audit': audit,
            'session': document['session'], 'driver': document['driver'],
            'registration': registration,
            'boundary_reference':np.loadtxt(root/'data/raw/tum/Spielberg_centerline.csv',delimiter=',')}


def require_role(data, role):
    if data['audit'].get('role') != role or any(l['role'] != role for l in data['laps']):
        raise ValueError('Role-separated research view required: '+role)
    return data
