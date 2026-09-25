"""
Hippopotamus Optimization (HO)

Derived from:
    Sources(MATLAB)/HO/HO/HO.m
    Authors: Mohammad Hussien Amiri, Nastaran Mehrabi Hashjin (2024)
    Paper: "Hippopotamus Optimization Algorithm: A novel nature-inspired metaheuristic"
           Scientific Reports.

Original MATLAB Function Signature:
    [Best_score, Best_pos, HO_curve] = HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

Python Entry Point:
    ho(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)

Parameters:
    SearchAgents   : int, population size
    Max_iterations : int, maximum number of iterations
    lowerbound     : float or numpy.ndarray of shape (dimension,), lower search bounds
    upperbound     : float or numpy.ndarray of shape (dimension,), upper search bounds
    dimension      : int, dimensionality of the problem
    fitness        : callable, objective function f(x) returning a scalar float

Returns:
    Best_score     : float, best fitness score found
    Best_pos       : numpy.ndarray of shape (dimension,), optimal solution vector
    HO_curve       : numpy.ndarray of shape (Max_iterations,), convergence history
"""

import math
import numpy as np


def _levy(n, m, beta=1.5):
    num = math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = np.random.normal(0.0, sigma_u, size=(n, m))
    v = np.random.normal(0.0, 1.0, size=(n, m))
    return u / (np.abs(v) ** (1.0 / beta))


def ho(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """
    Execute the Hippopotamus Optimization algorithm.
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
    Xbest = X[best_idx, :].copy()

    HO_curve = np.zeros(Max_iterations)

    half_agents = int(SearchAgents / 2)

    for t in range(1, Max_iterations + 1):
        cur_min_idx = np.argmin(fit)
        if fit[cur_min_idx] < fbest:
            fbest = fit[cur_min_idx]
            Xbest = X[cur_min_idx, :].copy()

        # Phase 1: Exploration in river or pond
        for i in range(half_agents):
            Dominant_hippopotamus = Xbest
            I1 = np.random.randint(1, 3)
            I2 = np.random.randint(1, 3)
            Ip1 = np.random.randint(0, 2, size=2)
            RandGroupNumber = np.random.randint(1, SearchAgents + 1)
            RandGroup = np.random.choice(SearchAgents, size=RandGroupNumber, replace=False)

            if len(RandGroup) != 1:
                MeanGroup = np.mean(X[RandGroup, :], axis=0)
            else:
                MeanGroup = X[RandGroup[0], :].copy()

            alfa_choices = [
                I2 * np.random.rand(dimension) + (1.0 - Ip1[0]),
                2.0 * np.random.rand(dimension) - 1.0,
                np.random.rand(dimension),
                I1 * np.random.rand(dimension) + (1.0 - Ip1[1]),
                np.random.rand()
            ]
            A = alfa_choices[np.random.randint(0, 5)]
            B = alfa_choices[np.random.randint(0, 5)]

            X_P1 = X[i, :] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i, :])
            X_P1 = np.clip(X_P1, lb, ub)

            T_factor = np.exp(-t / Max_iterations)
            if T_factor > 0.6:
                X_P2 = X[i, :] + A * (Dominant_hippopotamus - I2 * MeanGroup)
            else:
                if np.random.rand() > 0.5:
                    X_P2 = X[i, :] + B * (MeanGroup - Dominant_hippopotamus)
                else:
                    X_P2 = (ub - lb) * np.random.rand() + lb

            X_P2 = np.clip(X_P2, lb, ub)

            F_P1 = fitness(X_P1)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

            F_P2 = fitness(X_P2)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        # Phase 2: Defense against predators (Exploration)
        RL = 0.05 * _levy(SearchAgents, dimension, 1.5)
        for i in range(half_agents, SearchAgents):
            predator = lb + np.random.rand(dimension) * (ub - lb)
            F_HL = fitness(predator)
            distance2Leader = np.abs(predator - X[i, :]) + 1e-300
            b = np.random.uniform(2.0, 4.0)
            c = np.random.uniform(1.0, 1.5)
            d = np.random.uniform(2.0, 3.0)
            l = np.random.uniform(-2.0 * np.pi, 2.0 * np.pi)

            factor = b / (c - d * np.cos(l))

            if fit[i] > F_HL:
                X_P3 = RL[i, :] * predator + factor * (1.0 / distance2Leader)
            else:
                X_P3 = RL[i, :] * predator + factor * (1.0 / (2.0 * distance2Leader + np.random.rand(dimension)))

            X_P3 = np.clip(X_P3, lb, ub)
            F_P3 = fitness(X_P3)
            if F_P3 < fit[i]:
                X[i, :] = X_P3
                fit[i] = F_P3

        # Phase 3: Escaping from predator (Exploitation)
        LO_LOCAL = lb / t
        HI_LOCAL = ub / t
        for i in range(SearchAgents):
            alfa3_choices = [
                2.0 * np.random.rand(dimension) - 1.0,
                np.random.rand(),
                np.random.randn()
            ]
            D = alfa3_choices[np.random.randint(0, 3)]
            X_P4 = X[i, :] + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))
            X_P4 = np.clip(X_P4, lb, ub)

            F_P4 = fitness(X_P4)
            if F_P4 < fit[i]:
                X[i, :] = X_P4
                fit[i] = F_P4

        cur_min_idx = np.argmin(fit)
        if fit[cur_min_idx] < fbest:
            fbest = fit[cur_min_idx]
            Xbest = X[cur_min_idx, :].copy()

        HO_curve[t - 1] = fbest

    return fbest, Xbest, HO_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="HO",
        algo_func=ho,
        full_name="Hippopotamus Optimization (HO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

