"""
Osprey Optimization Algorithm (OOA)
Source: Osprey Optimization Algorithm: A New Bio-Inspired Metaheuristic Algorithm
        Frontiers in Mechanical Engineering, 2023. DOI: 10.3389/fmech.2022.1126450
Authors: Pavel Trojovský, Mohammad Dehghani
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


def ooa(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """
    Osprey Optimization Algorithm (OOA)

    Parameters:
        SearchAgents   : Number of search agents (ospreys)
        Max_iterations : Maximum number of iterations
        lowerbound     : Lower bounds (scalar or vector)
        upperbound     : Upper bounds (scalar or vector)
        dimension      : Problem dimension
        fitness        : Objective function f(x) -> float

    Returns:
        Best_score : Global best score
        Best_pos   : Global best position
        OOA_curve  : Convergence history
    """
    lb = np.ones(dimension) * lowerbound if np.isscalar(lowerbound) else np.asarray(lowerbound, dtype=float)
    ub = np.ones(dimension) * upperbound if np.isscalar(upperbound) else np.asarray(upperbound, dtype=float)

    X = initialization(SearchAgents, dimension, ub, lb)
    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :])

    best_idx = np.argmin(fit)
    fbest = float(fit[best_idx])
    xbest = X[best_idx, :].copy()

    OOA_curve = np.zeros(Max_iterations)

    for t in range(1, Max_iterations + 1):
        # Update best solution
        best_idx = np.argmin(fit)
        if fit[best_idx] < fbest:
            fbest = float(fit[best_idx])
            xbest = X[best_idx, :].copy()

        for i in range(SearchAgents):
            # Phase 1: Position identification and hunting the fish (Exploration)
            fish_positions = np.where(fit < fit[i])[0]
            if len(fish_positions) == 0:
                selected_fish = xbest
            else:
                if np.random.rand() < 0.5:
                    selected_fish = xbest
                else:
                    k = np.random.choice(fish_positions)
                    selected_fish = X[k, :]

            I = round(1 + np.random.rand())
            X_new_P1 = X[i, :] + np.random.rand() * (selected_fish - I * X[i, :])
            X_new_P1 = np.clip(X_new_P1, lb, ub)

            fit_new_P1 = fitness(X_new_P1)
            if fit_new_P1 < fit[i]:
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1

            # Phase 2: Carrying the fish to the suitable position (Exploitation)
            X_new_P2 = X[i, :] + (lb + np.random.rand() * (ub - lb)) / t
            X_new_P2 = np.clip(X_new_P2, lb, ub)

            fit_new_P2 = fitness(X_new_P2)
            if fit_new_P2 < fit[i]:
                X[i, :] = X_new_P2
                fit[i] = fit_new_P2

        best_idx = np.argmin(fit)
        if fit[best_idx] < fbest:
            fbest = float(fit[best_idx])
            xbest = X[best_idx, :].copy()

        OOA_curve[t - 1] = fbest

    Best_score = fbest
    Best_pos = xbest
    return Best_score, Best_pos, OOA_curve


OOA = ooa


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="OOA",
        algo_func=ooa,
        full_name="Osprey Optimization Algorithm (OOA)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
