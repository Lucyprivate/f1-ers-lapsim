"""Bounded action-only proposals, qualified by six actual carried executions."""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from .._core.learning_base import _write


@dataclass
class TeacherPlan:
    actions: list
    blocks: int
    qualification: dict
    search_evaluations: int
    proposal_status: list
    global_optimality_proven: bool = False


class BlockController:
    def __init__(self, actions):
        self.actions = np.asarray(actions, float).copy()
        if self.actions.ndim != 1 or not np.isfinite(self.actions).all() or (abs(self.actions)>1).any():
            raise ValueError('Finite action blocks in [-1,1] required')
    def __call__(self, observation):
        i = min(len(self.actions)-1, max(0, int(float(observation[0])*len(self.actions))))
        return float(self.actions[i])


def deployment_coefficients(deploy_mj, baseline_actions):
    fractions=(np.asarray(baseline_actions,float)+1.)/2.
    deploy=np.asarray(deploy_mj,float)
    coefficients=np.divide(deploy,fractions,out=np.zeros_like(deploy),where=fractions>1e-12)
    if np.any((fractions<=1e-12)&(deploy>1e-10)):
        raise ValueError('Zero requested deployment cannot identify a positive block coefficient')
    return fractions,coefficients


def qualify(sim, controller, output, initial_speed, *, initial_energy=2., initial_rule=None):
    """Carry all physical terminal fields; never shoot a policy-specific boundary."""
    from .environment import proxy_audit
    output = Path(output)
    speed, energy, carry = float(initial_speed), float(initial_energy), initial_rule
    laps, rows, proxies = [], [], []
    for index in range(6):
        result = sim.simulate(controller, initial_speed_mps=speed,
            initial_energy_mj=energy, carried_rule_state=carry, action_space='mode_aware')
        _write(output/f'lap_{index+1}.json.gz', result)
        laps.append(result)
        row = result['summary']
        rows.append(row)
        proxy = proxy_audit(sim, result)
        proxies.append(proxy)
        _write(output/f'lap_{index+1}_proxy.json', proxy)
        if not row['completed_lap'] or not row['execution_feasible'] or not row['feasible'] or not proxy['passed']:
            break
        speed, energy = result['terminal_speed_mps'], row['energy_end_mj']
        carry = result['rule_certificate']['terminal_boundary_state']
    crossings = [{'speed_error_mps': b['summary']['speed_start_mps']-a['terminal_speed_mps'],
        'energy_error_mj': b['summary']['energy_start_mj']-a['summary']['energy_end_mj'],
        'dc_error_kw': b['summary']['initial_dc_kw']-a['summary']['terminal_dc_kw'],
        'FSM_equal': b['summary']['initial_rule_state'] == a['rule_certificate']['terminal_boundary_state']}
        for a,b in zip(laps,laps[1:])]
    full = len(laps)==6 and all(r['completed_lap'] and r['execution_feasible'] for r in rows)
    continuity = full and all(abs(c['speed_error_mps'])<1e-10 and abs(c['energy_error_mj'])<1e-10
        and abs(c['dc_error_kw'])<1e-10 and c['FSM_equal'] for c in crossings)
    passed = bool(full and continuity and all(r['feasible'] for r in rows) and all(p['passed'] for p in proxies))
    report = {'passed':passed, 'physical_laps':len(laps), 'crossings':crossings,
        'physical_control_line_continuity':bool(continuity), 'lap_summaries':rows,
        'formal_objective_s':float(np.mean([r['lap_time_s'] for r in rows[-2:]])) if passed else None,
        'within_lap_velocity_or_energy_resets':0, 'policy_specific_boundary_shooting':False,
        'global_optimality_proven':False, 'infinite_horizon_proven':False}
    _write(output/'qualification.json', report)
    return report, laps


def casadi_proposal(sim, baseline, initial_state, blocks, initial_actions):
    """Fixed mode multiple shooting is a proposal, never an execution certificate.

    Mixed blocks use fixed braking duration and measured baseline aero drag;
    full time-domain integration/FSM is required for every resulting action.
    """
    try:
        import casadi as ca
    except ImportError:
        return None, {'status':'optional_dependency_missing','solver':'casadi'}
    trace = baseline['internal_trace']
    ss, tt = np.asarray(trace['s_m']), np.asarray(trace['time_s'])
    edges = np.linspace(0.,sim.length_m,blocks+1)
    dt = np.diff(np.interp(edges,ss,tt))
    ds = np.diff(edges)
    vref = np.interp(edges,ss,trace['speed_mps'])
    mids = .5*(edges[:-1]+edges[1:])
    g = sim.geometry.evaluate(mids)
    drag = .5*sim.vehicle.air_density_kgpm3*np.asarray(g['drag_area_m2'])*(.5*(vref[:-1]+vref[1:]))**2
    dct = np.asarray(trace['dc_kw'])
    rt = np.asarray(trace['dt_s'])
    deploy = np.r_[0.,np.cumsum(np.maximum(dct,0.)*rt/1000.)]
    harvest = np.r_[0.,np.cumsum(np.maximum(-dct,0.)*rt/1000.)]
    dep = np.diff(np.interp(edges,ss,deploy))
    har = np.diff(np.interp(edges,ss,harvest))
    reference_fraction,coefficients=deployment_coefficients(dep,initial_actions)
    # Fixed conditional schedule: blocks intersecting recorded HOLD/LIMITED
    # keep the baseline command. NORMAL blocks have a free proposal action.
    # This conservative block approximation does not predict FSM transitions.
    audit = baseline['energy_viability_audit']
    frozen_phases = []
    for a,b in zip(edges[:-1],edges[1:]):
        phases = sorted({r['state_after_decide']['phase'] for r in audit if a<=r['s_m']<b})
        frozen_phases.append(phases)
    opt = ca.Opti()
    u,v,e = opt.variable(blocks),opt.variable(blocks+1),opt.variable(blocks+1)
    opt.subject_to(opt.bounded(-1,u,1))
    opt.subject_to(opt.bounded(1,v,120))
    opt.subject_to(opt.bounded(2,e,4))
    opt.subject_to(v[0]==initial_state.speed_mps)
    opt.subject_to(e[0]==initial_state.energy_mj)
    time_terms=[]
    for i in range(blocks):
        if any(phase!='NORMAL' for phase in frozen_phases[i]):
            opt.subject_to(u[i]==float(initial_actions[i]))
        fraction=(u[i]+1)/2
        # The requested reference fraction is start-specific. Regeneration is
        # fixed; it is not an independent force source in this surrogate.
        power_delta=coefficients[i]/max(dt[i],1e-12)*1000*(fraction-reference_fraction[i])*sim.vehicle.deploy_efficiency
        opt.subject_to(v[i+1]**2==vref[i+1]**2+(v[i]**2-vref[i]**2)+2*power_delta*ds[i]/(sim.vehicle.mass_kg*max(.5*(vref[i]+vref[i+1]),1.)))
        opt.subject_to(e[i+1]==e[i]-coefficients[i]*fraction+har[i])
        time_terms.append(2*ds[i]/(v[i]+v[i+1]))
    opt.subject_to(e[-1]>=initial_state.energy_mj)
    opt.subject_to(v[-1]==initial_state.speed_mps)
    opt.minimize(ca.sum1(ca.vertcat(*time_terms)))
    opt.set_initial(u,initial_actions);opt.set_initial(v,vref)
    opt.set_initial(e,np.interp(edges,ss,trace['energy_mj']))
    opt.solver('ipopt',{'print_time':False},{'print_level':0,'max_iter':300,'tol':1e-8})
    try:
        solution=opt.solve()
        return np.clip(np.asarray(solution.value(u)).reshape(-1),-1,1), {
            'status':'proposal_only','solver':'casadi_ipopt','fixed_FSM_mode':True,
            'frozen_phase_schedule':frozen_phases,'HOLD_LIMITED_blocks_pinned_to_baseline':True,
            'FSM_transitions_optimized':False,'exact_FSM_in_solver':False,
            'solver_iterations':solution.stats().get('iter_count')}
    except RuntimeError as exc:
        return None, {'status':'solver_infeasible_or_nonconverged','reason':str(exc)[-800:],
                      'execution_state_overwritten':False}


def plan_ers(initial_state, rule_state, geometry, control_grid, budget):
    """geometry is the qualified research simulator; budget owns output/counts."""
    sim = geometry
    output = Path(budget['output'])
    cap = min(600,int(budget.get('maximum',600)))
    blocks = len(control_grid)-1
    if blocks != 16 or initial_state.s_m != 0. or initial_state.energy_mj != 2. or rule_state is not None:
        raise ValueError('Fixed cold boundary and 16 equal-length blocks required')
    count, proposals, best, best_laps, best_actions = 0, [], None, None, None
    def evaluate(actions):
        nonlocal count,best,best_laps,best_actions
        if count>=cap: return None,None
        count+=1
        target=output/f'candidate_{count:04d}'
        _write(target/'started.json',{'exact_search_evaluation':count,'actions':np.asarray(actions).tolist()})
        row,laps=qualify(sim,BlockController(actions),target,initial_state.speed_mps)
        if row['passed'] and (best is None or row['formal_objective_s']<best['formal_objective_s']-1e-8):
            best,best_laps,best_actions=row,laps,np.asarray(actions).copy()
        return row,laps
    # Each start, including failure, spends one exact-search evaluation.
    first_laps=None
    for start in (0.,-.25,.25):
        row,laps=evaluate(np.full(blocks,start))
        if start==0.: first_laps=laps
        if laps and laps[0]['summary']['completed_lap']:
            proposal,status=casadi_proposal(sim,laps[0],initial_state,blocks,np.full(blocks,start))
            proposals.append(status)
            if proposal is not None: evaluate(proposal)
    if best is not None:
        # Two fixed pattern-search scales, then exactly one 32-block sweep.
        # Stop at the global cap. Solver states never enter qualification.
        for step in (.1,.025):
            for index in range(blocks):
                for sign in (-1,1):
                    candidate=best_actions.copy();candidate[index]=np.clip(candidate[index]+sign*step,-1,1)
                    evaluate(candidate)
        best_actions=np.repeat(best_actions,2)
        for index in range(32):
            for sign in (-1,1):
                candidate=best_actions.copy();candidate[index]=np.clip(candidate[index]+sign*.025,-1,1)
                evaluate(candidate)
    report=TeacherPlan(best_actions.tolist() if best_actions is not None else [],
        len(best_actions) if best_actions is not None else blocks,
        best or {'passed':False,'reason':'No qualified exact candidate'},count,proposals)
    _write(output/'plan.json',asdict(report))
    return report
