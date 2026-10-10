"""Independent invariants for the distance-domain physical model."""
import unittest
import numpy as np
from f1ers.model import Simulator, VehicleParams, build_observations


def test_track(n=120, length=3000.0, straight=False):
    phase = np.arange(n) * 2 * np.pi / n
    curvature = np.full(n, 0.00001) if straight else 0.0005 + 0.018 * (np.sin(4 * phase) > 0.88)
    return dict(name="synthetic-invariant-track", s_m=np.arange(n) * length / n,
                x_m=400 * np.cos(phase), y_m=400 * np.sin(phase),
                curvature_1pm=curvature, length_m=length)


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Simulator(test_track(), {"soc_window_mj": 4.0, "harvest_max_mj_per_lap": 8.0}, VehicleParams())
        neutral = cls.sim.simulate(np.zeros(cls.sim.n_nodes), sustainable=False)
        cls.actions = np.where(neutral["wheel_power_kw"] < 0, -1.0, 0.8)
        cls.result = cls.sim.simulate(cls.actions, sustainable=True)

    def test_signed_dc_energy_conservation(self):
        r = self.result
        change = np.diff(r["energy_mj"])
        np.testing.assert_allclose(change, -r["mguk_dc_kw"] * r["dt_s"] / 1000.0, atol=2e-14)
        self.assertLess(abs(r["summary"]["energy_balance_residual_mj"]), 1e-12)
        self.assertAlmostEqual(r["summary"]["deploy_mj"] - r["summary"]["recharge_mj"],
                               r["energy_mj"][0] - r["energy_mj"][-1], places=11)

    def test_mechanical_work_matches_kinetic_change_and_drag_distance(self):
        sim, r, p = self.sim, self.result, self.sim.vehicle
        speed = r["speed_mps"]
        next_speed = np.roll(speed, -1)
        mean_speed = 0.5 * (speed + next_speed)
        drag = 0.5 * p.air_density_kgpm3 * sim.drag_area * mean_speed**2 + p.rolling_coefficient * p.mass_kg * 9.81
        expected = 0.5 * p.mass_kg * (next_speed**2 - speed**2) + drag * sim.ds_m
        np.testing.assert_allclose(r["wheel_power_kw"] * 1000.0 * r["dt_s"], expected, atol=2e-9)
        self.assertLess(r["summary"]["mechanical_work_max_segment_residual_mj"], 1e-12)
        np.testing.assert_allclose(r["ice_mechanical_kw"] + r["mguk_mechanical_kw"] - r["friction_brake_kw"],
                                   r["wheel_power_kw"], atol=1e-10)

    def test_nearly_zero_wheel_power_superclip_uses_ice(self):
        result = self.sim._coupled(np.full(self.sim.n_nodes, -0.2), None, 1.0)
        i = int(np.argmax(result["mguk_dc_kw"] < -1))
        result["wheel_power_kw"][i] = -3e-14
        result["mguk_dc_kw"][i] = -300.0
        finished = self.sim._finish(result, 1.0, float(result["energy_mj"][-1]), False)
        self.assertAlmostEqual(finished["ice_mechanical_kw"][i], 300.0 / self.sim.vehicle.harvest_efficiency)
        self.assertEqual(finished["friction_brake_kw"][i], 0.0)

    def test_dc_mechanical_efficiencies_have_correct_direction(self):
        r, p = self.result, self.sim.vehicle
        pos, neg = r["mguk_dc_kw"] > 0, r["mguk_dc_kw"] < 0
        self.assertTrue(pos.any() and neg.any())
        np.testing.assert_allclose(r["mguk_mechanical_kw"][pos], r["mguk_dc_kw"][pos] * p.deploy_efficiency)
        np.testing.assert_allclose(r["mguk_mechanical_kw"][neg] * p.harvest_efficiency, r["mguk_dc_kw"][neg])

    def test_soc_limits_and_no_fake_terminal_recharge(self):
        sim = Simulator(test_track(straight=True), {}, VehicleParams())
        r = sim.simulate(np.ones(sim.n_nodes), sustainable=False)
        self.assertGreaterEqual(r["energy_mj"].min(), -1e-12)
        self.assertLessEqual(r["energy_mj"].max(), 4 + 1e-12)
        self.assertFalse(r["summary"]["sustainable"])
        self.assertLess(r["energy_mj"][-1], r["energy_mj"][0])
        self.assertEqual(r["summary"]["projection_alpha"], 1.0)

    def test_sustainability_projection_is_disclosed_and_changes_dynamics(self):
        projected = self.result
        self.assertTrue(projected["summary"]["sustainable"])
        self.assertLess(projected["summary"]["projection_alpha"], 1.0)
        self.assertGreater(projected["summary"]["action_projection_mean_abs"], 0.0)
        self.assertGreaterEqual(projected["energy_mj"][-1], projected["energy_mj"][0] - 1e-6)
        unprojected = self.sim.simulate(self.actions, sustainable=False)
        self.assertGreater(np.max(abs(projected["speed_mps"] - unprojected["speed_mps"])), 0.01)

    def test_recharge_budget_truncates_at_limit(self):
        dc, _ = self.sim._constrain_dc(0, -350.0, 1.0, 70.0, -400.0, 1.0, 7.95, 0.0, 350.0)
        self.assertAlmostEqual(dc, -50.0, places=9)
        dc, _ = self.sim._constrain_dc(0, -350.0, 1.0, 70.0, -400.0, 1.0, 8.0, 0.0, 350.0)
        self.assertEqual(dc, 0.0)

    def test_standard_and_alternative_power_curves(self):
        sim = Simulator(test_track(), {"alternative_power_zones_m": [[0, 1000]]}, VehicleParams())
        for kph, expected in [(300, 300), (339, 105), (342, 60), (345, 0), (360, 0)]:
            caps = sim.deployment_limits_kw(np.full(sim.n_nodes, kph / 3.6))
            self.assertAlmostEqual(caps[-1], expected, places=9)
        caps = sim.deployment_limits_kw(np.full(sim.n_nodes, 300 / 3.6))
        self.assertEqual(caps[0], 250)
        self.assertEqual(caps[-1], 300)
        self.assertLessEqual(np.max(abs(self.result["mguk_dc_kw"])), 350 + 1e-9)

    def test_superclip_reduces_straight_speed_and_recovers_real_energy(self):
        sim = Simulator(test_track(straight=True), {}, VehicleParams())
        neutral = sim.simulate(np.zeros(sim.n_nodes), sustainable=False)
        harvesting = sim.simulate(np.full(sim.n_nodes, -0.4), sustainable=False)
        self.assertTrue(harvesting["summary"]["feasible"])
        self.assertGreater(harvesting["summary"]["lap_time_s"], neutral["summary"]["lap_time_s"] + 0.1)
        self.assertGreater(harvesting["summary"]["recharge_mj"], 0.1)
        self.assertLessEqual(harvesting["ice_mechanical_kw"].max(), sim.vehicle.ice_power_kw + 0.8)
        self.assertIn("superclip", harvesting["operating_mode"])

    def test_brake_regen_and_friction_share_the_same_wheel_demand(self):
        r = self.result
        mask = r["wheel_power_kw"] < 0
        regen_kw = -np.minimum(0.0, r["mguk_mechanical_kw"])
        np.testing.assert_allclose(regen_kw[mask] + r["friction_brake_kw"][mask],
                                   -r["wheel_power_kw"][mask], atol=1e-8)
        self.assertGreater(r["friction_brake_kw"][mask].max(), 0)
        self.assertLess(r["summary"]["violations"]["tire_force_excess_n"], 8)

    def test_periodic_velocity_and_closed_lap_kinematics(self):
        r = self.result
        expected_last_ax = (r["speed_mps"][0]**2 - r["speed_mps"][-1]**2) / (2 * self.sim.ds_m)
        self.assertAlmostEqual(r["ax_mps2"][-1], expected_last_ax, places=10)
        self.assertEqual(len(r["time_s"]), self.sim.n_nodes + 1)
        self.assertAlmostEqual(r["time_s"][-1], r["summary"]["lap_time_s"])
        self.assertEqual(r["summary"]["periodic_speed_error_mps"], 0)
        self.assertTrue(r["summary"]["coupling_converged"])

    def test_deployment_slew_public_subset_and_exception(self):
        # High-speed reduction to a target >=100kW is limited; lower target is
        # exempt under public C5.12.7, so we do not slew all sign transitions.
        dc, _ = self.sim._constrain_dc(0, 150, 0.2, 70, 600, 3, 0, 300, 350)
        self.assertAlmostEqual(dc, 280)
        dc, _ = self.sim._constrain_dc(0, 90, 0.2, 70, 600, 3, 0, 300, 350)
        self.assertEqual(dc, 90)

    def test_policy_observation_state_and_environment_energy(self):
        obs = build_observations(self.sim, self.result)
        self.assertEqual(obs.shape, (self.sim.n_nodes, 12))
        np.testing.assert_allclose(obs[:, 5], self.result["soc_fraction"][:-1])
        env = self.sim.create_env()
        state = env.reset()
        energy = env.energy
        state, reward, done, info = env.step(0.3)
        self.assertAlmostEqual(env.energy - energy, -info["mguk_dc_kw"] * info["dt_s"] / 1000, places=12)
        self.assertTrue(np.isfinite(state).all())
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        self.assertLess(abs(info["mechanical_work_residual_mj"]), 1e-12)
        env.reset(start_speed_mps=self.result["speed_mps"][0])
        self.assertAlmostEqual(env.start_speed, self.result["speed_mps"][0])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.sim.simulate(np.full(self.sim.n_nodes, 1.1))
        with self.assertRaises(ValueError):
            self.sim.simulate(np.full(self.sim.n_nodes, np.nan))
        with self.assertRaises(ValueError):
            self.sim.simulate(policy=lambda _: np.inf)


if __name__ == "__main__":
    unittest.main()
