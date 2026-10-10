import tempfile
import unittest
from pathlib import Path
import numpy as np
from f1ers_v3.vehicle import VehicleParamsV3,VehicleState,WheelCommand,SegmentGeometry,vehicle_step
from f1ers_v3.simulator import SimulatorV3
from f1ers_v3.learning import save_policy_v3,load_policy_v3,train_bc_dagger_v3
from f1ers.learning import GaussianPolicy


def straight_track(n=20,length=1000.):
    return dict(s_m=np.arange(n)*length/n,curvature_1pm=np.zeros(n),length_m=length,
                x_m=np.arange(n)*length/n,y_m=np.zeros(n),fia_s_m=np.arange(n)*length/n)


class VehicleTests(unittest.TestCase):
    def test_signed_wheel_work_conserved(self):
        p=VehicleParamsV3()
        for force in (2500.,-2500.):
            state,info=vehicle_step(VehicleState(50),WheelCommand(wheel_force_n=force),SegmentGeometry(5),p)
            self.assertTrue(info['step_certified'])
            self.assertAlmostEqual(info['wheel_force_n'],force)
            self.assertLess(abs(info['mechanical_work_residual_mj']),1e-8)
            self.assertGreater(state.elapsed_time_s,0.)
    def test_brake_ratio_only_changes_negative_capacity(self):
        g=SegmentGeometry(5,.001)
        p=VehicleParamsV3();q=VehicleParamsV3(brake_ratio=.6)
        _,a=vehicle_step(VehicleState(50),WheelCommand(brake_fraction=1),g,p)
        _,b=vehicle_step(VehicleState(50),WheelCommand(brake_fraction=1),g,q)
        self.assertAlmostEqual(b['tire_drive_capacity_n'],a['tire_drive_capacity_n'])
        self.assertAlmostEqual(b['tire_brake_capacity_n'],.6*a['tire_brake_capacity_n'])
    def test_invalid_lateral_step_does_not_reset_speed(self):
        state=VehicleState(100)
        after,info=vehicle_step(state,WheelCommand(power_kw=400),SegmentGeometry(5,.03),VehicleParamsV3())
        self.assertEqual(after,state);self.assertFalse(info['step_certified'])
        self.assertIn('lateral_tire_force',info['guard_reasons'])
    def test_accelerating_end_lateral_overshoot_rejected(self):
        state=VehicleState(33.)
        after,info=vehicle_step(state,WheelCommand(power_kw=1000),SegmentGeometry(50,.02),VehicleParamsV3())
        self.assertEqual(after,state)
        self.assertIn('lateral_tire_force',info['guard_reasons'])
        self.assertGreater(info['speed_next_mps'],state.speed_mps)
    def test_replay_has_no_hidden_brake_or_electrical_claim(self):
        sim=SimulatorV3(straight_track(),{},VehicleParamsV3())
        trace=dict(time_s=np.array([0,5,50.]),speed_mps=np.array([30.,2.,2.]),
                   throttle=np.array([1.,1.,1.]),brake=np.zeros(3))
        a=sim.simulate('replay',driver_trace=trace)
        trace['speed_mps'][1:]=100
        b=sim.simulate('replay',driver_trace=trace)
        self.assertTrue(a['summary']['feasible']);self.assertFalse(a['ers_validated'])
        self.assertIsNone(a['energy_mj']);self.assertIsNone(a['mguk_dc_kw'])
        np.testing.assert_array_equal(a['speed_mps'],b['speed_mps'])
        self.assertTrue(np.all(a['wheel_force_n']>=0))
        self.assertGreater(a['speed_mps'][-1],30.)
    def test_replay_brake_is_previous_hold_not_interpolated(self):
        sim=SimulatorV3(straight_track(length=500),{},VehicleParamsV3())
        trace=dict(time_s=np.array([0,1.,2.,50.]),speed_mps=np.full(4,60.),
                   throttle=np.ones(4),brake=np.array([0,1,0,0]))
        result=sim.simulate('replay',driver_trace=trace)
        times=result['observed_input_time_s'];brake=result['brake']
        np.testing.assert_array_equal(brake,((times>=1)&(times<2)).astype(float))
        self.assertTrue(np.all(np.isin(brake,[0.,1.])))
    def test_replay_failure_is_retained_as_partial(self):
        sim=SimulatorV3(straight_track(),{},VehicleParamsV3())
        trace=dict(time_s=np.array([0,.01]),speed_mps=np.array([30.,30.]),
                   throttle=np.ones(2),brake=np.zeros(2))
        result=sim.simulate('replay',driver_trace=trace)
        self.assertFalse(result['summary']['completed_lap'])
        self.assertLess(len(result['s_m']),sim.n_nodes)
        self.assertEqual(result['summary']['guard_failures'][0]['guard_reasons'],['driver_trace_exhausted'])
    def test_autonomous_causal_floor_and_shared_kernel(self):
        sim=SimulatorV3(straight_track(),{},VehicleParamsV3())
        result=sim.simulate(controller=np.r_[-np.ones(10)*.3,np.ones(10)])
        self.assertTrue(result['summary']['completed_lap'])
        self.assertTrue(result['summary']['execution_feasible'])
        self.assertGreaterEqual(np.min(result['energy_mj']),2.-1e-8)
        self.assertTrue(result['summary']['shared_vehicle_step'])
        self.assertFalse(result['summary']['whole_lap_scaling_used'])
        for row in result['guard_diagnostics']:
            if row['wheel_power_kw']<0:
                self.assertLessEqual(row['friction_brake_kw']+max(0.,-row['mguk_mechanical_kw']),
                    max(0.,-row['wheel_power_kw'])+1e-5)
    def test_autonomous_rejects_observed_trace(self):
        sim=SimulatorV3(straight_track(),{},VehicleParamsV3())
        with self.assertRaises(ValueError):sim.simulate(driver_trace={})
    def test_actual_brake_regen_and_friction_share_budget(self):
        track=straight_track();track['curvature_1pm'][8:12]=.01
        sim=SimulatorV3(track,{},VehicleParamsV3())
        result=sim.simulate(controller=-np.ones(sim.n_nodes))
        braking=[r for r in result['guard_diagnostics'] if r['wheel_power_kw']<-1e-6]
        self.assertTrue(braking)
        for row in braking:
            self.assertAlmostEqual(row['friction_brake_kw']+max(0.,-row['mguk_mechanical_kw']),-row['wheel_power_kw'],places=5)
            self.assertLessEqual(-row['wheel_force_n'],row['tire_brake_capacity_n']+1e-5)
    def test_mesh_profile_freezes_driver_input(self):
        sim=SimulatorV3(straight_track(),{},VehicleParamsV3())
        profile={'s_m':np.array([0.,1000.]),'drive_fraction':np.ones(2),'brake_fraction':np.zeros(2)}
        fine=SimulatorV3(straight_track(n=40),{},VehicleParamsV3())
        result=fine.simulate(controller={'driver_profile':profile,'actions':np.zeros(40)},initial_speed_mps=sim.boundary_speed_mps)
        self.assertTrue(result['summary']['driver_profile_fixed'])
        self.assertTrue(result['summary']['execution_feasible'])
        self.assertAlmostEqual(result['summary']['speed_start_mps'],sim.boundary_speed_mps)
        self.assertAlmostEqual(result['distance_completed_m'],1000.)
    def test_fixed_profile_does_not_reintroduce_future_braking_cap(self):
        sim=SimulatorV3(straight_track(),{},VehicleParamsV3())
        profile={'s_m':np.array([0.,1000.]),'drive_fraction':np.ones(2),'brake_fraction':np.zeros(2)}
        sim.set_fixed_driver_profile(profile)
        # Deliberately reduce only the future geometry-driver envelope. Actual
        # straight-line wheel, tire and electrical capabilities are unchanged.
        sim.safe_speed_mps[:]=20.
        result=sim.simulate(controller=np.zeros(sim.n_nodes))
        self.assertTrue(result['summary']['completed_lap'])
        self.assertTrue(result['summary']['execution_feasible'])
        self.assertFalse(result['summary']['future_braking_envelope_gate'])
        self.assertTrue(any(r['future_braking_envelope_excess_mps']>0 for r in result['guard_diagnostics']))
        self.assertTrue(all('next_speed_cap_mps' not in r['guard_reasons'] for r in result['guard_diagnostics']))
    def test_fixed_profile_preserves_actual_lateral_and_energy_guards(self):
        track=straight_track();track['curvature_1pm'][10:]=.02
        sim=SimulatorV3(track,{},VehicleParamsV3())
        profile={'s_m':np.array([0.,1000.]),'drive_fraction':np.ones(2),'brake_fraction':np.zeros(2)}
        result=sim.simulate(controller={'driver_profile':profile,'actions':np.ones(sim.n_nodes)})
        self.assertFalse(result['summary']['completed_lap'])
        failures=result['summary']['guard_failures']
        self.assertTrue(any('lateral_tire_force' in reasons for reasons in failures))
        self.assertGreaterEqual(np.min(result['energy_mj']),2.-1e-8)
    def test_bc_dagger_real_executor_checkpoint_roundtrip(self):
        sim=SimulatorV3(straight_track(),{},VehicleParamsV3())
        teacher=sim.simulate()
        with tempfile.TemporaryDirectory() as folder:
            gates={key:True for key in ('convergence','physical','grid','development','new_test')}
            result=train_bc_dagger_v3(sim,teacher,folder,gates,seeds=(0,),bc_epochs=2,dagger_rounds=1,dagger_epochs=2)
            self.assertFalse(result['skipped'])
            self.assertTrue(result['training_passed'])
            self.assertEqual(len(result['controllers']),1)
            self.assertEqual(len(result['records'][0]['candidates']),2)
    def test_v2_policy_schema_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'v3.npz';policy=GaussianPolicy(13,np.zeros(13),np.ones(13),(8,8))
            save_policy_v3(policy,path);loaded=load_policy_v3(path)
            self.assertEqual(loaded.observation_dim,13)
            with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
            arrays['schema_version']=np.asarray(2);np.savez(path,**arrays)
            with self.assertRaises(ValueError):load_policy_v3(path)
    def test_training_gate_cannot_be_skipped(self):
        result=train_bc_dagger_v3(None,None,'unused',{'physical':True})
        self.assertTrue(result['skipped']);self.assertFalse(result['sac_expanded'])


if __name__=='__main__':unittest.main()
