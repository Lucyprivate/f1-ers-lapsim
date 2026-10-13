"""Independent model-fitting, accuracy and sustainable-control gates.

``None`` is an unmeasured or inapplicable gate, never evidence of passing it.
Replay has no electrical-state or public-ERS certification. A complete physical
autonomous cold lap may enter calibration without a periodic boundary, while
control qualification still requires the actual six-lap carried-state check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GateReport:
    mode: str
    data_coverage: bool | None
    complete: bool
    physical: bool
    public_rules: bool | None
    accuracy: bool | None = None
    per_lap_energy: bool | None = None
    endpoint_repeatability: bool | None = None
    six_carry_continuity: bool | None = None

    def __post_init__(self):
        if self.mode not in ('replay', 'autonomous'):
            raise ValueError('Gate mode must be replay or autonomous')
        for name in ('data_coverage', 'complete', 'physical', 'public_rules',
                     'accuracy', 'per_lap_energy', 'endpoint_repeatability',
                     'six_carry_continuity'):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError('Gate values must be bool or None: '+name)
        if self.mode == 'replay' and any(getattr(self, name) is not None for name in
                ('public_rules', 'per_lap_energy', 'endpoint_repeatability', 'six_carry_continuity')):
            raise ValueError('Replay ERS and sustainable-control gates are not applicable')

    @property
    def model_fit_eligible(self):
        coverage = self.data_coverage is True if self.mode == 'replay' else self.data_coverage is not False
        rules = self.public_rules is True if self.mode == 'autonomous' else self.public_rules is None
        return bool(coverage and self.complete and self.physical and rules)

    @property
    def single_lap_control_qualified(self):
        return bool(self.mode == 'autonomous' and self.model_fit_eligible
                    and self.per_lap_energy is True and self.endpoint_repeatability is True)

    @property
    def control_qualified(self):
        return bool(self.single_lap_control_qualified and self.six_carry_continuity is True)

    def to_dict(self):
        return {**asdict(self), 'model_fit_eligible': self.model_fit_eligible,
                'single_lap_control_qualified': self.single_lap_control_qualified,
                'control_qualified': self.control_qualified,
                'scope': 'declared public rule subset; model closure is an engineering gate'}

    @classmethod
    def from_result(cls, result, score_report=None, *, six_carry_continuity=None):
        summary = result['summary']
        mode = summary.get('mode', result.get('mode'))
        complete = bool(summary.get('completed_lap', False))
        physical = bool(summary.get('execution_feasible', False))
        measured = complete and physical
        accuracy = (bool(score_report['accuracy_passed'])
                    if measured and score_report is not None else None)
        if mode == 'replay':
            return cls(mode, bool(summary.get('input_coverage_passed',
                summary.get('status') != 'input_exhausted')), complete, physical,
                None, accuracy)
        return cls(mode, None, complete, physical,
            bool(summary.get('public_rules_passed', False)), accuracy,
            bool(summary.get('sustainable', False)) if measured else None,
            bool(all(summary.get(name, False) for name in
                ('periodic_speed_closed', 'dc_boundary_closed', 'rule_state_closed')))
                if measured else None,
            six_carry_continuity)


def model_fit_eligible(record):
    """Aggregate trial eligibility never consults cold-lap SOC/closure flags."""
    return bool(record.get('complete') and record.get('physical')
                and record.get('public_rules_passed') is True
                and record.get('data_coverage_passed', True))
