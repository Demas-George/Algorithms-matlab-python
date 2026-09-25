"""
Zebra Optimization Algorithm (ZOA)

Derived from:
    Sources(MATLAB)/ZOA/ZOA.m
    Authors: Eva Trojovska, Mohammad Dehghani, Pavel Trojovsky (2022)
    Paper: "Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm
           for Solving Optimization Problems"
           IEEE Access, 10, pp. 49445-49473.
           DOI: 10.1109/ACCESS.2022.3172789

Original MATLAB Function Signature:
    [Best_score, Best_pos, ZOA_curve] = ZOA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

Python Entry Point:
    zoa(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

Parameters:
    SearchAgents   : int, population size of zebras
    Max_iterations : int, maximum number of iterations
    lowerbound     : float or numpy.ndarray of shape (dimension,), lower boundaries
    upperbound     : float or numpy.ndarray of shape (dimension,), upper boundaries
    dimension      : int, dimensionality of the problem
    fitness        : callable, objective function f(x) returning a scalar float

Returns:
    Best_score     : float, optimal fitness score found
    Best_pos       : numpy.ndarray of shape (dimension,), optimal solution vector found
    ZOA_curve      : numpy.ndarray of shape (Max_iterations,), convergence history
"""

import numpy as np


def zoa(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """
    Execute the Zebra Optimization Algorithm.
    """
    lb = np.full(dimension, lowerbound, dtype=float) if np.isscalar(lowerbound) else np.asarray(lowerbound, dtype=float)
    ub = np.full(dimension, upperbound, dtype=float) if np.isscalar(upperbound) else np.asarray(upperbound, dtype=float)

    X = np.zeros((SearchAgents, dimension))
    fit = np.zeros(SearchAgents)

    for i in range(dimension):
        X[:, i] = lb[i] + np.random.rand(SearchAgents) * (ub[i] - lb[i])

    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :])

    best_idx = np.argmin(fit)
    fbest = fit[best_idx]
    PZ = X[best_idx, :].copy()

    ZOA_curve = np.zeros(Max_iterations)

    for t in range(1, Max_iterations + 1):
        cur_min_idx = np.argmin(fit)
        if fit[cur_min_idx] < fbest:
            fbest = fit[cur_min_idx]
            PZ = X[cur_min_idx, :].copy()

        # Phase 1: Foraging Behaviour
        for i in range(SearchAgents):
            I = int(round(1.0 + np.random.rand()))
            X_newP1 = X[i, :] + np.random.rand(dimension) * (PZ - I * X[i, :])
            X_newP1 = np.clip(X_newP1, lb, ub)

            f_newP1 = fitness(X_newP1)
            if f_newP1 <= fit[i]:
                X[i, :] = X_newP1
                fit[i] = f_newP1

        # Phase 2: Defense strategies against predators
        Ps = np.random.rand()
        k = np.random.randint(0, SearchAgents)
        AZ = X[k, :].copy()  # attacked zebra

        for i in range(SearchAgents):
            if Ps < 0.5:
                # S1: lion attacks zebra, escape strategy
                R = 0.1
                X_newP2 = X[i, :] + R * (2.0 * np.random.rand(dimension) - 1.0) * (1.0 - t / Max_iterations) * X[i, :]
            else:
                # S2: other predators attack zebra, offensive strategy
                I = int(round(1.0 + np.random.rand()))
                X_newP2 = X[i, :] + np.random.rand(dimension) * (AZ - I * X[i, :])

            X_newP2 = np.clip(X_newP2, lb, ub)
            f_newP2 = fitness(X_newP2)
            if f_newP2 <= fit[i]:
                X[i, :] = X_newP2
                fit[i] = f_newP2

        ZOA_curve[t - 1] = fbest

    return fbest, PZ, ZOA_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="ZOA",
        algo_func=zoa,
        full_name="Zebra Optimization Algorithm (ZOA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

