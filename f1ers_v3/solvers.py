"""Bounded geometry-map heuristic and numerical teacher for the v3 executor."""
from f1ers.solvers_v2 import valid_result
from f1ers.solvers_v2 import solve_heuristic_v2, solve_optimal_v2


def solve_heuristic_v3(sim):
    result=solve_heuristic_v2(sim)
    result['solver_status']['execution_protocol']='signed_vehicle_causal_v3'
    return result


def solve_optimal_v3(sim,initial=None,maxiter=40,n_blocks=6):
    result=solve_optimal_v2(sim,initial=initial,maxiter=maxiter,n_blocks=n_blocks)
    result['solver_status']['execution_protocol']='signed_vehicle_causal_v3'
    return result
