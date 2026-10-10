"""Finite causal control family; robust search across shield plateaus."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from f1ers_v4.artifacts import write_json
from f1ers_v4.calibration import write_compressed
from .metrics import legal_time


class BlockController:
    def __init__(self, values, action_space='signed'):
        self.values = np.asarray(values,float)
        self.action_space = action_space
    def __call__(self, obs):
        block = min(len(self.values)-1, int(max(0.,obs[0])*len(self.values)))
        amount = float(np.clip(self.values[block],0.,1.))
        # Actual negative wheel demand is available causally in feature 11.
        if self.action_space == 'signed':
            return -1. if obs[11]>.5 else amount
        return 1. if obs[11]>.5 else 2*amount-1.
    def to_dict(self):
        return {'kind':'phase_blocks_with_actual_braking','values':self.values.tolist(),'action_space':self.action_space}


def search(sim, output, *, blocks=6, evaluations=24, seed=2026, action_space='signed', initial=None, progress=print):
    """Bounded coarse/coordinate search, no smooth-gradient claim at clipping."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    best=None; best_result=None; best_score=float('inf'); history=[]
    initial = np.full(blocks,.6) if initial is None else np.asarray(initial,float)
    if len(initial)!=blocks:
        initial=np.interp((np.arange(blocks)+.5)/blocks,(np.arange(len(initial))+.5)/len(initial),initial)
    for i in range(evaluations):
        if i<7:
            values = np.full(blocks,[0.,.25,.5,.7,.9,1.][i-1]) if i else initial.copy()
        elif best is None:
            values=rng.uniform(.05,1.,blocks)
        elif i%4==0:
            values=np.clip(best+rng.normal(0,.18,blocks),0,1)
        else:
            values=best.copy(); j=(i-6)%blocks
            values[j]=np.clip(values[j]+(.12 if (i//blocks)%2==0 else -.12),0.,1.)
        controller=BlockController(values,action_space)
        result=sim.periodic_simulate(controller,action_space=action_space,max_cycles=4)
        cost=legal_time(result); improved=cost<best_score-1e-8
        if improved: best,best_result,best_score=values.copy(),result,cost
        history.append({'evaluation':i,'values':values.tolist(),'summary':result['summary'],
            'selected':improved,'objective_s':cost if np.isfinite(cost) else None,
            'failure_or_rejection':None if np.isfinite(cost) else result.get('guard_diagnostics',[])})
        # Every proposal retained, including complete but rule-boundary failures.
        write_compressed(output/f'trial_{i:03d}.json.gz',result)
        write_json(output/'search_history.json',history)
        progress(f'{action_space} teacher {i+1}/{evaluations}: '+(f'{cost:.6f}s' if np.isfinite(cost) else 'rejected'),flush=True)
    report={'algorithm':'seeded finite coarse/coordinate search','seed':seed,'budget':evaluations,
        'evaluations':len(history),'complete_feasible_count':sum(h['objective_s'] is not None for h in history),
        'best':None if best is None else BlockController(best,action_space).to_dict(),
        'best_time_s':best_score if np.isfinite(best_score) else None,'history':history,
        'global_optimality_proven':False,'future_observed_speed_used':False}
    write_json(output/'teacher.json',report)
    return {'controller':None if best is None else BlockController(best,action_space),'result':best_result,'report':report}
