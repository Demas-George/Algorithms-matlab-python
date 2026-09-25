"""
Whale Optimization Algorithm (WOA)

Derived from:
    Sources(MATLAB)/WOA/WOA-master/WOA.m
    Authors: Seyedali Mirjalili, Andrew Lewis (2016)
    Paper: "The Whale Optimization Algorithm"
           Advances in Engineering Software, 95, pp. 51-67.
           DOI: 10.1016/j.advengsoft.2016.01.008

Original MATLAB Function Signature:
    [Leader_score, Leader_pos, Convergence_curve] = WOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

Python Entry Point:
    woa(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

Parameters:
    SearchAgents_no   : int, number of search agents (whales)
    Max_iter          : int, maximum number of iterations
    lb                : float or numpy.ndarray of shape (dim,), lower boundaries
    ub                : float or numpy.ndarray of shape (dim,), upper boundaries
    dim               : int, dimensionality of the problem
    fobj              : callable, objective function f(x) returning a scalar float

Returns:
    Leader_score      : float, best fitness score found
    Leader_pos        : numpy.ndarray of shape (dim,), optimal decision vector found
    Convergence_curve : numpy.ndarray of shape (Max_iter,), convergence history
"""

import numpy as np


def woa(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    Execute the Whale Optimization Algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Leader_pos = np.zeros(dim)
    Leader_score = np.inf

    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.rand(SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(Max_iter)

    for t in range(Max_iter):
        for i in range(SearchAgents_no):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            fitness = fobj(Positions[i, :])
            if fitness < Leader_score:
                Leader_score = fitness
                Leader_pos = Positions[i, :].copy()

        a = 2.0 - t * (2.0 / Max_iter)
        a2 = -1.0 + t * (-1.0 / Max_iter)

        for i in range(SearchAgents_no):
            r1 = np.random.rand()
            r2 = np.random.rand()

            A = 2.0 * a * r1 - a
            C = 2.0 * r2

            b = 1.0
            l = (a2 - 1.0) * np.random.rand() + 1.0
            p = np.random.rand()

            for j in range(dim):
                if p < 0.5:
                    if abs(A) >= 1.0:
                        rand_leader_index = np.random.randint(0, SearchAgents_no)
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand
                    else:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader
                else:
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    Positions[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2.0 * np.pi) + Leader_pos[j]

        Convergence_curve[t] = Leader_score

    return Leader_score, Leader_pos, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="WOA",
        algo_func=woa,
        full_name="Whale Optimization Algorithm (WOA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

