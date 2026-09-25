"""
Dandelion Optimizer (DO)

Derived from:
    Sources(MATLAB)/Dandelion-Optimizer (DO)/Dandelion-Optimizer/DO.m
    Authors: Shijie Zhao, Tianran Zhang, Shilin Ma, Miao Chen (2022)
    Paper: "Dandelion Optimizer: A nature-inspired metaheuristic algorithm
           for engineering applications"
           Engineering Applications of Artificial Intelligence, 114, 105075.
           DOI: 10.1016/j.engappai.2022.105075

Original MATLAB Function Signature:
    [Best_fitness, Best_position, Convergence_curve] = DO(Popsize, Maxiteration, LB, UB, Dim, Fobj)

Python Entry Point:
    do(Popsize, Maxiteration, LB, UB, Dim, Fobj)

Parameters:
    Popsize           : int, number of dandelion seeds in the population
    Maxiteration      : int, maximum number of iterations
    LB                : float or numpy.ndarray of shape (Dim,), lower boundaries
    UB                : float or numpy.ndarray of shape (Dim,), upper boundaries
    Dim               : int, dimensionality of the problem
    Fobj              : callable, objective function f(x) returning a scalar float

Returns:
    Best_fitness      : float, global minimum fitness score
    Best_position     : numpy.ndarray of shape (Dim,), optimal solution vector
    Convergence_curve : numpy.ndarray of shape (Maxiteration,), convergence history
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


def _lognpdf(x, mu=0.0, sigma=1.0):
    """
    Computes log-normal probability density function matching MATLAB lognpdf.
    """
    x_safe = np.maximum(x, 1e-300)
    return (1.0 / (x_safe * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((np.log(x_safe) - mu) / sigma) ** 2)


def do(Popsize, Maxiteration, LB, UB, Dim, Fobj):
    """
    Execute the Dandelion Optimizer algorithm.
    """
    LB = np.full(Dim, LB, dtype=float) if np.isscalar(LB) else np.asarray(LB, dtype=float)
    UB = np.full(Dim, UB, dtype=float) if np.isscalar(UB) else np.asarray(UB, dtype=float)

    dandelions = np.zeros((Popsize, Dim))
    for i in range(Dim):
        dandelions[:, i] = np.random.rand(Popsize) * (UB[i] - LB[i]) + LB[i]

    dandelionsFitness = np.zeros(Popsize)
    for i in range(Popsize):
        dandelionsFitness[i] = Fobj(dandelions[i, :])

    sorted_indexes = np.argsort(dandelionsFitness)
    Best_position = dandelions[sorted_indexes[0], :].copy()
    Best_fitness = dandelionsFitness[sorted_indexes[0]]

    Convergence_curve = np.zeros(Maxiteration)
    Convergence_curve[0] = Best_fitness

    denom_a = (Maxiteration ** 2 - 2.0 * Maxiteration + 1.0)
    a = 1.0 / denom_a if denom_a != 0 else 1.0
    b = -2.0 * a
    c = 1.0 - a - b

    for t in range(2, Maxiteration + 1):
        # Rising stage
        beta = np.random.randn(Popsize, Dim)
        alpha = np.random.rand() * ((1.0 / Maxiteration ** 2) * t ** 2 - 2.0 / Maxiteration * t + 1.0)
        k = 1.0 - np.random.rand() * (c + a * t ** 2 + b * t)

        dandelions_1 = np.zeros((Popsize, Dim))
        if np.random.randn() < 1.5:
            for i in range(Popsize):
                lamb = np.abs(np.random.randn(Dim))
                theta = (2.0 * np.random.rand() - 1.0) * np.pi
                row = 1.0 / np.exp(theta)
                vx = row * np.cos(theta)
                vy = row * np.sin(theta)
                NEW = np.random.rand(Dim) * (UB - LB) + LB
                dandelions_1[i, :] = (dandelions[i, :]
                                      + alpha * vx * vy * _lognpdf(lamb, 0.0, 1.0) * (NEW - dandelions[i, :]))
        else:
            for i in range(Popsize):
                dandelions_1[i, :] = dandelions[i, :] * k

        dandelions = np.clip(dandelions_1, LB, UB)

        # Decline stage
        dandelions_mean = np.mean(dandelions, axis=0)
        dandelions_2 = np.zeros((Popsize, Dim))
        for i in range(Popsize):
            dandelions_2[i, :] = (dandelions[i, :]
                                  - beta[i, :] * alpha * (dandelions_mean - beta[i, :] * alpha * dandelions[i, :]))
        dandelions = np.clip(dandelions_2, LB, UB)

        # Landing stage
        Step_length = _levy(Popsize, Dim, 1.5)
        Elite = np.tile(Best_position, (Popsize, 1))
        dandelions_3 = np.zeros((Popsize, Dim))
        for i in range(Popsize):
            dandelions_3[i, :] = (Elite[i, :]
                                  + Step_length[i, :] * alpha * (Elite[i, :] - dandelions[i, :] * (2.0 * t / Maxiteration)))
        dandelions = np.clip(dandelions_3, LB, UB)

        # Evaluate fitness
        for i in range(Popsize):
            dandelionsFitness[i] = Fobj(dandelions[i, :])

        sorted_indexes = np.argsort(dandelionsFitness)
        dandelions = dandelions[sorted_indexes, :]
        dandelionsFitness = dandelionsFitness[sorted_indexes]

        if dandelionsFitness[0] < Best_fitness:
            Best_position = dandelions[0, :].copy()
            Best_fitness = dandelionsFitness[0]

        Convergence_curve[t - 1] = Best_fitness

    return Best_fitness, Best_position, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="DO",
        algo_func=do,
        full_name="Dandelion Optimizer (DO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

