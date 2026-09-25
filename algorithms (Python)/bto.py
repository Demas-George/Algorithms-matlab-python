"""
Barrel Theory-Based Optimizer (BTO)

Derived from:
    Sources(MATLAB)/BTO/BTO_v1.m
    Authors: Van Tai Tran, Quynh T.T. Nhu, Chitsutha Soomlek, Punyaphol Horata, Khamron Sunat (2025)
    Paper: "BTO: A Barrel Theory-Based Optimizer for Engineering Design Problems"
           Applied Engineering Letters, 10(4), DOI: 10.46793/aeletters.2025.10.4.2

Original MATLAB Function Signature:
    [Best_fit, Best_sol, Convergence_curve] = BTO(N, MaxIter, lb, ub, dim, fobj)

Python Entry Point:
    bto(N, MaxIter, lb, ub, dim, fobj)

Parameters:
    N                 : int, population size (number of planks)
    MaxIter           : int, maximum number of iterations
    lb                : float or numpy.ndarray of shape (dim,), lower boundaries
    ub                : float or numpy.ndarray of shape (dim,), upper boundaries
    dim               : int, dimensionality of the problem
    fobj              : callable, objective function f(x) returning a scalar float

Returns:
    Best_fit          : float, best objective value found
    Best_sol          : numpy.ndarray of shape (dim,), optimal decision vector
    Convergence_curve : numpy.ndarray of shape (MaxIter,), convergence history
"""

import numpy as np


def _handle_boundary(val, original, lb, ub):
    if val < lb:
        return (original + lb) / 2.0
    elif val > ub:
        return (original + ub) / 2.0
    return val


def _barrel_adjustment_probability(obj_vals, it, MaxIter, D):
    min_val = np.min(obj_vals)
    max_val = np.max(obj_vals)
    if max_val == min_val:
        norm_fitness = np.zeros_like(obj_vals, dtype=float)
    else:
        norm_fitness = (obj_vals - min_val) / (max_val - min_val)

    progress = it / MaxIter
    lower = 0.3 * (1.0 - np.sqrt(progress))
    upper_end = 0.3 * (1.0 - D / 100.0)
    upper = 1.0 - (1.0 - upper_end) * np.sqrt(progress)
    return lower + (upper - lower) * norm_fitness


def bto(N, MaxIter, lb, ub, dim, fobj):
    """
    Execute the Barrel Theory-Based Optimizer algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Planks = np.zeros((N, dim))
    for i in range(dim):
        Planks[:, i] = np.random.rand(N) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(MaxIter)
    Fitness = np.zeros(N)
    epsilon = 1e-8

    for i in range(N):
        Fitness[i] = fobj(Planks[i, :])

    best_idx = np.argmin(Fitness)
    Best_fit = Fitness[best_idx]
    Best_sol = Planks[best_idx, :].copy()

    for it in range(1, MaxIter + 1):
        E = np.sqrt(it / MaxIter)
        normalized_fitness = _barrel_adjustment_probability(Fitness, it, MaxIter, dim)

        K = max(1, int(np.ceil(10.0 * (1.0 - E))))
        topK_idx = np.argsort(Fitness)[:K]

        for i in range(N):
            base = Planks[i, :].copy()
            jrand = np.random.randint(0, dim)

            F = (2.0 * (np.random.rand(dim) - 0.5)
                 * np.sin(2.0 * np.random.rand(dim) * np.pi * it / (MaxIter / 10.0))
                 * (1.0 - it / MaxIter))

            elite_idx = topK_idx[np.random.randint(0, K)]

            for j in range(dim):
                r1 = np.random.rand()
                r2 = np.random.rand()

                if r1 < normalized_fitness[i] or j == jrand:
                    if r2 < E:
                        sel = np.random.randint(0, N)
                        while sel == elite_idx:
                            sel = np.random.randint(0, N)

                        base[j] = (Planks[sel, j]
                                   + F[j] * (Planks[elite_idx, j] - Planks[sel, j])
                                   + F[j] * 0.25 * (1.0 - E) * np.random.randn() * (ub[j] - lb[j]))
                    else:
                        d = Planks[elite_idx, j] - Planks[i, j]
                        sign_term = d + (1.0 if d == 0.0 else 0.0) * (2.0 * (1.0 if np.random.rand() < E else 0.0) - 1.0)
                        direction = -np.sign(sign_term)
                        step = abs(F[j] * (ub[j] - lb[j]) * 0.3
                                   + (1.0 - abs(F[j])) * E * Planks[i, j] * np.random.randn()
                                   + epsilon)
                        base[j] = Planks[i, j] + direction * step

            for j in range(dim):
                base[j] = _handle_boundary(base[j], Planks[i, j], lb[j], ub[j])

            fitTrial = fobj(base)
            if fitTrial < Fitness[i]:
                Fitness[i] = fitTrial
                Planks[i, :] = base
                if fitTrial < Best_fit:
                    Best_fit = fitTrial
                    Best_sol = base.copy()

        Convergence_curve[it - 1] = Best_fit

    return Best_fit, Best_sol, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="BTO",
        algo_func=bto,
        full_name="Barrel Tumbleweed Optimizer (BTO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

