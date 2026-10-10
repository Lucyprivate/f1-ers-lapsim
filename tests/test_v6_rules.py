import copy
import math
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

from f1ers_v4.vehicle import IntegrationState
from f1ers_v6.rules import RuleConfig, RuleShield, RuleState


def fixture(**settings):
    geo = SimpleNamespace(length_m=4326.)
    rules = {"ers_max_power_kw":350., "soc_window_mj":4., "harvest_max_mj_per_lap":8.,
             "power_reduction_rate_kw_per_s":100., "fia_centreline_length_m":4326.,
             "alternative_power_zones_route_m":[[2170.,2960.],[3720.,3920.]],
             "power_reduction_exceptions_route":[{"start_m":2700.,"end_m":3000.,"initial_reduction_max_kw":350.},
                                                    {"start_m":3750.,"end_m":3900.,"initial_reduction_max_kw":350.}]}
    shield = RuleShield(rules, geo, RuleConfig(**settings))
    state = IntegrationState(100.,300/3.6,0.,3.5,0.,0.)
    shield.reset(state)
    return shield, state, {"wheel_force_n":1000.,"brake_fraction":0.,"power_kw":400.,"target_acceleration_mps2":None}


class RuleTests(unittest.TestCase):
    def test_locked_subset_configuration(self):
        for args in [{"energy_floor_mj":0.},{"initial_hold_s":.9},{"mechanical_reduction_rate_kw_per_s":101.},
                     {"crankshaft_lower_bound_rpm":10000.},{"total_mechanical_reduction_kw":701.}]:
            with self.assertRaises(ValueError):RuleConfig(**args)
        with self.assertRaises(ValueError):RuleShield({"boost_enabled":True},SimpleNamespace(length_m=4326.))

    def test_piecewise_dc_cap_and_half_open_alternative_zones(self):
        shield, _, _ = fixture()
        self.assertEqual(shield._cap(100.,280/3.6),350.)
        self.assertEqual(shield._cap(100.,300/3.6),300.)
        self.assertEqual(shield._cap(2170.,300/3.6),250.)
        self.assertEqual(shield._cap(2960.,300/3.6),300.)
        self.assertEqual(shield._cap(100.,340/3.6),100.)
        self.assertEqual(shield._cap(100.,345/3.6),0.)

    def test_negative_acceleration_is_not_negative_wheel_demand(self):
        shield, state, command = fixture()
        command["target_acceleration_mps2"] = -1.
        info = shield.decide(state,command,-1.)
        self.assertEqual(info["requested_dc_kw"],0.)
        self.assertIn("positive_wheel_superclip_disabled",info["guard_reasons"])

    def test_braking_only_regeneration_and_mode_aware_mapping(self):
        shield, state, command = fixture()
        command["wheel_force_n"] = -3000.
        shield.decide(state,command,-1.)
        self.assertEqual(shield.query(state,command,shield.geometry,.02)["requested_dc_kw"],-350.)
        shield.decide(state,command,1.,"mode_aware")
        self.assertEqual(shield.request_kw,-350.)
        shield.decide(state,command,1.,"signed")
        self.assertEqual(shield.request_kw,0.)

    def test_no_deployment_at_initial_energy_floor(self):
        shield, state, command = fixture()
        state = replace(state,energy_mj=2.)
        shield.decide(state,command,1.)
        self.assertEqual(shield.query(state,command,shield.geometry,.02)["requested_dc_kw"],0.)

    def test_one_second_hold_plus_slew_reserve_before_deployment(self):
        shield, state, command = fixture()
        state = replace(state,energy_mj=2.5)
        shield.decide(state,command,1.)
        dt = .02
        power = shield.query(state,command,shield.geometry,dt)["requested_dc_kw"]
        used = power*dt/1000.
        self.assertLess(power,350.)
        self.assertLessEqual(used+power/1000.+power*power/(2000.*100.),.5+1e-12)

    def test_query_proposals_and_rejections_are_pure(self):
        shield, state, command = fixture()
        shield.state = replace(shield.state,reference_mechanical_kw=332.5,previous_dc_kw=350.)
        state = replace(state,previous_dc_kw=350.)
        shield.decide(state,command,0.)
        before = copy.deepcopy(shield.__dict__)
        a = shield.query(state,command,shield.geometry,.02)
        shield.query(state,command,shield.geometry,.01)
        b = shield.query(state,command,shield.geometry,.02)
        self.assertEqual(a,b)
        self.assertEqual(shield.state,before["state"])
        self.assertEqual(shield.decisions,before["decisions"])
        self.assertEqual(shield.violations,before["violations"])

    def test_high_speed_zero_initial_reduction_respects_both_slew_rules(self):
        shield, state, command = fixture()
        shield.state = replace(shield.state,reference_mechanical_kw=332.5,previous_dc_kw=350.)
        state = replace(state,previous_dc_kw=350.,speed_mps=280/3.6)
        shield.decide(state,command,0.)
        self.assertEqual(shield.state.phase,"HOLD")
        self.assertEqual(shield.state.reduction_mechanical_kw,0.)
        self.assertEqual(shield.next_time_event(0.),1.)
        early = shield.query(replace(state,elapsed_time_s=.5),command,shield.geometry,.02)
        self.assertAlmostEqual(early["requested_dc_kw"],350.)
        self.assertEqual(shield._reduction_at(.99),0.)

    def test_low_speed_initial_mechanical_reduction_is_150(self):
        shield, state, command = fixture()
        shield.state = replace(shield.state,reference_mechanical_kw=332.5)
        shield.decide(replace(state,speed_mps=200/3.6),command,0.)
        self.assertEqual(shield.state.reduction_mechanical_kw,150.)

    def test_special_window_initial_reduction_is_350(self):
        shield, state, command = fixture()
        shield.state = replace(shield.state,reference_mechanical_kw=332.5)
        shield.decide(replace(state,s_m=2750.,speed_mps=200/3.6),command,0.)
        self.assertEqual(shield.state.reduction_mechanical_kw,332.5)
        self.assertEqual(shield._initial_limit(3000.),150.)

    def test_limited_reduction_cannot_increase_demand(self):
        shield, state, command = fixture()
        shield.state = replace(shield.state,reference_mechanical_kw=332.5)
        shield.decide(state,command,0.)
        shield.accept(replace(state,elapsed_time_s=1.5,previous_dc_kw=140.),[])
        self.assertEqual(shield.state.phase,"LIMITED")
        self.assertAlmostEqual(shield.state.reduction_mechanical_kw,47.5)
        shield.decide(replace(state,elapsed_time_s=1.5,previous_dc_kw=140.),command,1.)
        proposed = shield.query(replace(state,elapsed_time_s=1.5,previous_dc_kw=140.),command,shield.geometry,.02)
        self.assertLessEqual(proposed["requested_dc_kw"],285./.95+1e-12)

    def test_real_negative_demand_ends_declared_period(self):
        shield, state, command = fixture()
        shield.state = replace(shield.state,phase="HOLD",reference_mechanical_kw=332.5,reduction_mechanical_kw=150.,hold_until_s=1.)
        shield.decide(state,{**command,"wheel_force_n":-1.},-1.)
        self.assertEqual(shield.state.phase,"NORMAL")
        self.assertEqual(shield.state.reference_mechanical_kw,0.)
        self.assertEqual(shield.state.reduction_mechanical_kw,0.)

    def test_crossing_preserves_dc_and_normalizes_hold_timer(self):
        shield, state, _ = fixture()
        carry = RuleState(phase="HOLD",elapsed_time_s=70.,hold_until_s=70.6,reduction_mechanical_kw=120.,
                          reference_mechanical_kw=300.,previous_dc_kw=200.,target_reduction_mechanical_kw=220.,requested_dc_kw=180.)
        new = replace(state,elapsed_time_s=0.,previous_dc_kw=200.,recharge_mj=0.)
        restored = shield.reset(new,asdict(carry))
        self.assertAlmostEqual(restored.hold_until_s,.6)
        self.assertEqual(restored.phase,"HOLD")
        self.assertEqual(restored.reduction_mechanical_kw,120.)
        self.assertEqual(restored.reference_mechanical_kw,300.)
        self.assertEqual(restored.previous_dc_kw,200.)
        self.assertEqual(restored.requested_dc_kw,180.)
        with self.assertRaises(ValueError):shield.reset(replace(new,previous_dc_kw=0.),carry)

    def test_theoretical_dc_slew_violation_is_detected(self):
        shield, state, _ = fixture()
        shield.state = replace(shield.state,previous_dc_kw=300.)
        rec = {"status":"accepted","dc_kw":0.,"dt_s":.1,"s_start_m":100.,"s_end_m":108.,
               "speed_start_mps":300/3.6,"speed_next_mps":300/3.6,"wheel_force_n":1000.,"time_end_s":.1}
        shield.accept(replace(state,elapsed_time_s=.1),[rec])
        self.assertFalse(shield.certificate()["public_subset_passed"])
        self.assertIn("electrical_reduction_slew",shield.violations[0]["reasons"])

    def test_engineering_and_policing_torque_are_both_checked(self):
        shield, state, _ = fixture()
        omega = 7500.*2*math.pi/60.
        self.assertLess(350/.90*1000/omega,500.)
        self.assertLess(350/.97*1000/omega,500.)
        rec = {"status":"accepted","dc_kw":-360.,"dt_s":.02,"wheel_force_n":-1000.}
        shield.accept(replace(state,elapsed_time_s=.02,previous_dc_kw=-360.),[rec])
        self.assertIn("engineering_mechanical_torque",shield.violations[0]["reasons"])
        cert = shield.certificate()
        self.assertFalse(cert["full_fia_verified"])
        self.assertIn("unverified",cert)

    def test_window_and_recharge_failure_are_retained(self):
        shield, state, _ = fixture()
        rec = {"status":"accepted","dc_kw":0.,"dt_s":.02,"energy_next_mj":1.9,"recharge_next_mj":8.1}
        shield.accept(replace(state,elapsed_time_s=.02),[rec])
        self.assertEqual(set(shield.violations[0]["reasons"]),{"declared_energy_window","lap_recharge"})

    def test_tail_mask_is_experimental_and_does_not_abruptly_reset(self):
        shield, state, command = fixture()
        tail = replace(state,s_m=4000.)
        info = shield.decide(tail,command,1.)
        self.assertEqual(info["requested_dc_kw"],0.)
        self.assertEqual(shield.features(tail,command)["deployment_tail_mask"],1.)
        shield.state = replace(shield.state,reference_mechanical_kw=332.5,previous_dc_kw=350.)
        shield.decide(replace(tail,previous_dc_kw=350.),command,1.)
        self.assertEqual(shield.state.phase,"HOLD")
        power = shield.query(replace(tail,previous_dc_kw=350.),command,shield.geometry,.02)["requested_dc_kw"]
        self.assertGreater(power,0.)
        self.assertIn("not a FIA rule",shield.certificate()["assumptions"]["tail_guard"])

    def test_boundary_closure_checks_fsm_and_dc(self):
        shield, _, _ = fixture()
        self.assertTrue(shield.certificate()["rule_state_closed"])
        shield.state = replace(shield.state,phase="LIMITED",reduction_mechanical_kw=20.)
        self.assertFalse(shield.certificate()["rule_state_closed"])

    def test_auxiliary_draw_is_sensitivity_only(self):
        shield, state, _ = fixture()
        shield.accept(replace(state,elapsed_time_s=80.),[])
        self.assertAlmostEqual(shield.certificate()["auxiliary_energy_sensitivity_mj"],.004)
        self.assertEqual(shield.config.energy_floor_mj,2.)
        self.assertIn("not measured",shield.certificate()["assumptions"]["auxiliary"])


if __name__ == "__main__":unittest.main()
