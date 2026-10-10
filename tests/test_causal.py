"""v2 invariants: identical causal trajectories and reserve/slew certificates."""
import unittest
from dataclasses import replace
import numpy as np

from f1ers.causal import CausalSimulator,CausalState,CausalExecutionConfig,PROTOCOL
from f1ers.model import VehicleParams


def synthetic_track(n=80,length=2400.0):
    phase=np.arange(n)*2*np.pi/n
    return dict(s_m=np.arange(n)*length/n,length_m=length,
        x_m=400*np.cos(phase),y_m=400*np.sin(phase),
        curvature_1pm=0.0005+0.018*(np.sin(phase*4)>0.88))


class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim=CausalSimulator(synthetic_track(),{},VehicleParams())
        zero=cls.sim.zero_result
        cls.actions=np.where(zero["wheel_power_kw"]<0,-1.0,0.9)
        cls.result=cls.sim.simulate(cls.actions)

    def test_common_boundary_is_actual_causal_noK_fixed_point(self):
        sim=self.sim
        self.assertTrue(sim.boundary_report["converged"])
        self.assertLessEqual(sim.boundary_report["cycles"],20)
        self.assertLessEqual(sim.boundary_report["residual_mps"],1e-4)
        self.assertTrue(sim.zero_result["summary"]["feasible"])
        self.assertEqual(sim.reference_speed_mps.shape,(sim.n_nodes+1,))
        self.assertLess(abs(sim.reference_speed_mps[-1]-sim.boundary_speed_mps),0.02)

    def test_train_and_evaluation_use_identical_transition(self):
        env=self.sim.create_env()
        env.reset(seed=4)
        done=False
        while not done:
            obs,reward,done,info=env.step(self.actions[env.i])
        replay=env.result()
        for key in ("speed_mps","speed_next_mps","mguk_dc_kw","energy_mj","recharge_mj",
                    "dt_s","requested_actions","executed_actions","observations"):
            np.testing.assert_array_equal(replay[key],self.result[key])

    def test_floor_and_reserve_are_certified_every_step(self):
        r,c=self.result,self.sim.config
        self.assertTrue(r["summary"]["feasible"])
        self.assertGreaterEqual(r["energy_mj"].min(),c.energy_floor_mj)
        reserve=np.maximum(r["mguk_dc_kw"],0)**2/(2000*c.ramp_rate_kw_per_s)
        np.testing.assert_allclose(reserve,r["reserve_mj"],atol=1e-14)
        self.assertTrue(np.all(r["energy_mj"][1:]-c.energy_floor_mj>=reserve-1e-12))
        self.assertTrue(r["summary"]["sustainable"])
        self.assertFalse(r["summary"]["whole_lap_scaling_used"])
        self.assertEqual(r["summary"]["projection_alpha"],1.0)

    def test_no_initial_credit_cannot_deploy(self):
        sim=self.sim
        initial=CausalState(0,sim.boundary_speed_mps,2.0)
        next_state,info=sim.advance(initial,1.0)
        self.assertTrue(info["step_certified"])
        self.assertEqual(info["mguk_dc_kw"],0.0)
        self.assertEqual(next_state.energy_mj,2.0)
        self.assertEqual(info["deployable_credit_mj"],0.0)
        self.assertGreater(info["requested_dc_kw"],0.0)

    def test_only_actual_harvest_releases_credit(self):
        sim=self.sim
        i=int(np.argmax(sim.zero_result["wheel_power_kw"]<0))
        state=CausalState(i,float(sim.reference_speed_mps[i]),2.0)
        next_state,info=sim.advance(state,-1.0)
        self.assertTrue(info["step_certified"])
        self.assertGreater(next_state.energy_mj,2.0)
        self.assertGreater(sim.observation(next_state)[12],0.0)
        self.assertGreater(next_state.recharge_mj,0.0)

    def test_dc_and_mechanical_work_are_independently_conserved(self):
        r=self.result
        np.testing.assert_allclose(np.diff(r["energy_mj"]),-r["mguk_dc_kw"]*r["dt_s"]/1000,atol=5e-16)
        np.testing.assert_allclose(r["ice_mechanical_kw"]+r["mguk_mechanical_kw"]-r["friction_brake_kw"],
                                   r["wheel_power_kw"],atol=1e-10)
        self.assertLess(r["summary"]["mechanical_work_max_segment_residual_mj"],1e-12)
        self.assertLess(abs(r["summary"]["energy_balance_residual_mj"]),1e-12)

    def test_slew_and_reserve_no_posthoc_energy_clip(self):
        # Search a power-on segment with enough speed and a standard 350kW cap.
        sim=self.sim
        found=None
        for i in range(sim.n_nodes):
            state=CausalState(i,65.0,3.0,0.0,300.0)
            physical=sim._physical(state,0.0)
            if sim._slew_exempt(state,physical)=="none":
                found=state;break
        self.assertIsNotNone(found)
        next_state,info=sim.advance(found,0.0)
        self.assertTrue(info["step_certified"])
        self.assertGreater(info["mguk_dc_kw"],0.0)
        self.assertGreaterEqual(info["mguk_dc_kw"]+100*info["dt_s"],300-1e-4)
        self.assertGreaterEqual(next_state.energy_mj-2.0,sim.reserve_mj(info["mguk_dc_kw"])-1e-12)
        self.assertIn("conservative_slew_hold",info["guard_reasons"])

    def test_empty_intersection_stops_without_fake_state_update(self):
        sim=self.sim
        found=None
        for i in range(sim.n_nodes):
            state=CausalState(i,65.0,2.0,0.0,300.0)
            if sim._slew_exempt(state,sim._physical(state,0.0))=="none":
                found=state;break
        self.assertIsNotNone(found)
        next_state,info=sim.advance(found,0.0)
        self.assertFalse(info["step_certified"])
        self.assertEqual(next_state,found)
        self.assertIn("empty_power_intersection",info["guard_reasons"])
        self.assertTrue(info["execution_stopped"])

    def test_cap_window_and_recharge_limit(self):
        sim=self.sim
        i=int(np.argmax(sim.zero_result["wheel_power_kw"]<0))
        for energy,recharge in ((4.0,0.0),(2.0,8.0)):
            state=CausalState(i,float(sim.reference_speed_mps[i]),energy,recharge)
            next_state,info=sim.advance(state,-1.0)
            self.assertTrue(info["step_certified"])
            self.assertEqual(info["mguk_dc_kw"],0.0)
            self.assertEqual(next_state.energy_mj,energy)
            self.assertEqual(next_state.recharge_mj,recharge)
        self.assertLessEqual(np.max(abs(self.result["mguk_dc_kw"])),350)
        self.assertTrue(np.all(self.result["mguk_dc_kw"]<=self.result["deployment_cap_kw"]+1e-8))

    def test_finish_speed_is_actually_capped_and_failure_is_not_scored_valid(self):
        sim=self.sim
        self.assertLessEqual(self.result["speed_next_mps"][-1],sim.boundary_speed_mps+1e-8)
        r=sim.simulate(np.full(sim.n_nodes,-1.0))
        self.assertEqual(r["summary"]["boundary_speed_mps"],sim.boundary_speed_mps)
        if r["summary"]["periodic_speed_error_mps"]>.02:
            self.assertFalse(r["summary"]["feasible"])
        with self.assertRaises(ValueError):
            sim.create_env().reset(start_speed_mps=sim.boundary_speed_mps+1)

    def test_observation_schema_and_reward_components(self):
        env=self.sim.create_env();obs=env.reset()
        self.assertEqual(obs.shape,(13,))
        self.assertEqual(obs[-1],0)
        _,reward,done,info=env.step(1.0)
        self.assertAlmostEqual(reward,info["reward_time_s"]+info["reward_terminal_energy_s"]+
                               info["reward_terminal_speed_s"]+info["reward_failure_s"])
        self.assertIn("requested_action",info)
        self.assertIn("executed_action",info)
        self.assertEqual(self.result["summary"]["protocol"],PROTOCOL)

    def test_reset_does_not_lock_next_lap_floor_to_previous_end(self):
        env=self.sim.create_env();env.reset()
        while not env.done:
            env.step(-1.0)
        env.reset()
        self.assertEqual(env.energy,2.0)
        self.assertEqual(env.recharge,0.0)
        self.assertEqual(env.state.previous_dc_kw,0.0)

    def test_invalid_protocol_and_action_are_rejected(self):
        with self.assertRaises(ValueError):
            CausalSimulator(synthetic_track(),{},VehicleParams(),CausalExecutionConfig(energy_floor_mj=1.5))
        with self.assertRaises(ValueError):
            self.sim.simulate(np.full(self.sim.n_nodes,np.nan))
        with self.assertRaises(ValueError):
            self.sim.simulate(policy=lambda _:1.01)
        with self.assertRaises(ValueError):
            self.sim.simulate(sustainable=False)


if __name__=="__main__":
    unittest.main()
