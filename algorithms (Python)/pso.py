"""
Particle Swarm Optimization (PSO)

Derived from:
    Sources(MATLAB)/PSO/PSO.m
    Author: Seyedali Mirjalili

Original MATLAB Function Signature:
    [GBEST, cgCurve] = PSO(noP, maxIter, problem, dataVis)

Python Entry Point:
    pso(noP, maxIter, lb, ub, dim, fobj)

Parameters:
    noP     : int, number of particles
    maxIter : int, maximum number of iterations
    lb      : float or numpy.ndarray of shape (dim,), lower boundaries
    ub      : float or numpy.ndarray of shape (dim,), upper boundaries
    dim     : int, problem dimension
    fobj    : callable, objective function f(x) returning a scalar float

Returns:
    gbest_score : float, best objective value found (GBEST.O)
    gbest_pos   : numpy.ndarray of shape (dim,), best position vector found (GBEST.X)
    cgCurve     : numpy.ndarray of shape (maxIter,), convergence curve
"""

import numpy as np


def pso(noP, maxIter, lb, ub, dim, fobj):
    """
    Execute the Particle Swarm Optimization algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    wMax = 0.9
    wMin = 0.2
    c1 = 2.0
    c2 = 2.0
    vMax = (ub - lb) * 0.2
    vMin = -vMax

    # Swarm initialization
    X = np.zeros((noP, dim))
    V = np.zeros((noP, dim))
    pbest_X = np.zeros((noP, dim))
    pbest_O = np.full(noP, np.inf)

    gbest_X = np.zeros(dim)
    gbest_O = np.inf

    for k in range(noP):
        X[k, :] = (ub - lb) * np.random.rand(dim) + lb
        V[k, :] = np.zeros(dim)

    cgCurve = np.zeros(maxIter)

    for t in range(1, maxIter + 1):
        for k in range(noP):
            currentX = X[k, :]
            score = fobj(currentX)

            if score < pbest_O[k]:
                pbest_X[k, :] = currentX.copy()
                pbest_O[k] = score

            if score < gbest_O:
                gbest_X = currentX.copy()
                gbest_O = score

        w = wMax - t * ((wMax - wMin) / maxIter)

        for k in range(noP):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)
            V[k, :] = (w * V[k, :]
                       + c1 * r1 * (pbest_X[k, :] - X[k, :])
                       + c2 * r2 * (gbest_X - X[k, :]))

            V[k, :] = np.clip(V[k, :], vMin, vMax)
            X[k, :] = X[k, :] + V[k, :]
            X[k, :] = np.clip(X[k, :], lb, ub)

        cgCurve[t - 1] = gbest_O

    return gbest_O, gbest_X, cgCurve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="PSO",
        algo_func=pso,
        full_name="Particle Swarm Optimization (PSO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

