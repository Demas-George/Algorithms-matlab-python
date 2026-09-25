"""
Ivy Algorithm (IVYA)

Derived from:
    Sources(MATLAB)/IVY/IVYA_code/IVY.m
    Authors: Mojtaba Ghasemi, Mohsen Zare, Pavel Trojovsky, Ravipudi Venkata Rao,
             Eva Trojovska, Venkatachalam Kandasamy (2024)
    Paper: "Optimization based on the smart behavior of plants with its engineering applications:
           Ivy Algorithm (IVYA)"
           Knowledge-Based Systems.
           DOI: 10.1016/j.knosys.2024.111850

Original MATLAB Function Signature:
    [Destination_fitness, Destination_position, Convergence_curve] = IVY(N, Max_iteration, lb, ub, dim, fobj)

Python Entry Point:
    ivy(N, Max_iteration, lb, ub, dim, fobj)

Parameters:
    N                    : int, population size (number of plants)
    Max_iteration        : int, maximum number of iterations
    lb                   : float or numpy.ndarray of shape (dim,), lower boundaries
    ub                   : float or numpy.ndarray of shape (dim,), upper boundaries
    dim                  : int, problem dimension
    fobj                 : callable, objective function f(x) returning a scalar float

Returns:
    Destination_fitness  : float, optimal fitness score found
    Destination_position : numpy.ndarray of shape (dim,), optimal decision vector
    Convergence_curve    : numpy.ndarray of shape (Max_iteration,), convergence history
"""

import numpy as np


class _Plant:
    __slots__ = ('Position', 'Cost', 'GV')

    def __init__(self, position, cost, gv):
        self.Position = position
        self.Cost = cost
        self.GV = gv


def ivy(N, Max_iteration, lb, ub, dim, fobj):
    """
    Execute the Ivy Algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    range_diff = np.where(ub - lb == 0, 1e-15, ub - lb)

    pop = []
    for _ in range(N):
        pos = lb + np.random.rand(dim) * (ub - lb)
        gv = pos / range_diff
        cost = fobj(pos)
        pop.append(_Plant(pos, cost, gv))

    Convergence_curve = np.zeros(Max_iteration)

    for it in range(1, Max_iteration + 1):
        newpop = []
        n_current = len(pop)

        for i in range(n_current):
            ii = (i + 1) if (i + 1) < n_current else 0

            beta_1 = 1.0 + (np.random.rand() / 2.0)

            if pop[i].Cost < beta_1 * pop[0].Cost:
                new_pos = (pop[i].Position
                           + np.abs(np.random.randn(dim)) * (pop[ii].Position - pop[i].Position)
                           + np.random.randn(dim) * pop[i].GV)
            else:
                new_pos = pop[0].Position * (np.random.rand() + np.random.randn(dim) * pop[i].GV)

            pop[i].GV = pop[i].GV * ((np.random.rand() ** 2) * np.random.randn(dim))

            new_pos = np.clip(new_pos, lb, ub)
            new_gv = new_pos / range_diff
            new_cost = fobj(new_pos)

            newpop.append(_Plant(new_pos, new_cost, new_gv))

        pop.extend(newpop)
        pop.sort(key=lambda p: p.Cost)

        if len(pop) > N:
            pop = pop[:N]

        Convergence_curve[it - 1] = pop[0].Cost

    return pop[0].Cost, pop[0].Position.copy(), Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="IVY",
        algo_func=ivy,
        full_name="Ivy Algorithm (IVYA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

