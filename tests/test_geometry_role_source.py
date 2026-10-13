"""Original control support is restricted before cross-role channel repair."""
from datetime import datetime, timezone
from unittest.mock import patch
import unittest
from f1ers_v7.research import data
from f1ers_v7._core.input_io import DataUnavailable


def date(t):
    return datetime.fromtimestamp(1700000000.+t, timezone.utc).isoformat()


def fixture():
    numbers = (3, 4, 5, 6, 7, 9)
    metadata = [{'lap_number': n, 'driver_number': 63, 'session_key': 11315,
        'date_start': date((n-3)*10), 'lap_duration': 10.,
        'duration_sector_1': 3., 'duration_sector_2': 3., 'duration_sector_3': 4.}
        for n in numbers]
    car = [{'date': date(t), 'speed': 180., 'throttle': 100., 'brake': 0.}
           for t in (-.1, .1, 9.9, 10.1, 19.9, 20.1, 29.9, 30.1, 39.9, 40.1, 49.9, 60.1, 69.9)]
    location = [{'date': r['date'], 'x': float(i), 'y': float(i)} for i, r in enumerate(car)]
    documents = {'dataset.json': {'laps': [{'lap_number': n, 'compound': 'MEDIUM'} for n in numbers]},
        'laps.json': metadata, 'car_data_63_laps3to9.json': car,
        'location_63_laps3to9.json': location}
    return documents


class RoleSourceTests(unittest.TestCase):
    def test_future_values_not_materialized_or_used_to_repair(self):
        docs = fixture()
        docs['car_data_63_laps3to9.json'][-1]['throttle'] = 200.
        with patch.object(data, 'read_json', lambda p: docs[p.name]):
            _, rows, arrays = data.load_role_source('unused', 'calibration')
        self.assertEqual([r['lap_number'] for r in rows], [3, 4, 5, 6])
        self.assertAlmostEqual(arrays['car_t'][-1], 1700000039.9)
        self.assertLess(arrays['loc_t'][-1], 1700000040.)
        self.assertTrue(arrays['role_source_provenance']['invalid_calibration_repair_uses_early_view_only'])

    def test_invalid_early_tail_repairs_only_from_early_view(self):
        docs = fixture()
        docs['car_data_63_laps3to9.json'][8]['throttle'] = 200.
        docs['car_data_63_laps3to9.json'][9]['throttle'] = 0.
        with patch.object(data, 'read_json', lambda p: docs[p.name]):
            _, _, arrays = data.load_role_source('unused', 'calibration')
        self.assertFalse(arrays['throttle_original_valid'][-1])
        self.assertEqual(arrays['throttle'][-1], 1.)
        self.assertEqual(arrays['signal_audit']['throttle_invalid_samples'], 1)


if __name__ == '__main__': unittest.main()
