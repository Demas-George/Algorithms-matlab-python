"""
Sine Cosine Algorithm (SCA)
Source: SCA: A Sine Cosine Algorithm for solving optimization problems
        Knowledge-Based Systems, 2016. DOI: 10.1016/j.knosys.2015.12.022
Author: Seyedali Mirjalili
"""

import numpy as np


def initialization(SearchAgents_no, dim, ub, lb):
    ub = np.array(ub, dtype=float)
    lb = np.array(lb, dtype=float)
    if ub.size == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    return Positions


def sca(N, Max_iteration, lb, ub, dim, fobj):
    """
    Sine Cosine Algorithm (SCA)

    Parameters:
        N             : Number of search agents
        Max_iteration : Maximum number of iterations
        lb            : Lower bounds (scalar or vector)
        ub            : Upper bounds (scalar or vector)
        dim           : Dimensionality of problem
        fobj          : Objective function handle f(x) -> float

    Returns:
        Destination_fitness  : Global minimum objective found
        Destination_position : Global best position vector
        Convergence_curve    : History of best fitness per iteration
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    X = initialization(N, dim, ub_arr, lb_arr)

    Destination_position = np.zeros(dim)
    Destination_fitness = np.inf

    Convergence_curve = np.zeros(Max_iteration)

    for i in range(N):
        fit = fobj(X[i, :])
        if fit < Destination_fitness:
            Destination_fitness = fit
            Destination_position = X[i, :].copy()

    Convergence_curve[0] = Destination_fitness
    a = 2.0

    for t in range(2, Max_iteration + 1):
        r1 = a - t * (a / Max_iteration)

        for i in range(N):
            for j in range(dim):
                r2 = (2.0 * np.pi) * np.random.rand()
                r3 = 2.0 * np.random.rand()
                r4 = np.random.rand()

                if r4 < 0.5:
                    X[i, j] = X[i, j] + (r1 * np.sin(r2) * abs(r3 * Destination_position[j] - X[i, j]))
                else:
                    X[i, j] = X[i, j] + (r1 * np.cos(r2) * abs(r3 * Destination_position[j] - X[i, j]))

            # Boundary clipping
            X[i, :] = np.clip(X[i, :], lb_arr, ub_arr)

            obj_val = fobj(X[i, :])
            if obj_val < Destination_fitness:
                Destination_position = X[i, :].copy()
                Destination_fitness = obj_val

        Convergence_curve[t - 1] = Destination_fitness

    return Destination_fitness, Destination_position, Convergence_curve


SCA = sca


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="SCA",
        algo_func=sca,
        full_name="Sine Cosine Algorithm (SCA)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
