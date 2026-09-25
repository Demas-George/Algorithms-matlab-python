"""
Flower Pollination Algorithm (FPA)

Derived from:
    Sources(MATLAB)/FPA/fpa_demo.m
    Author: Xin-She Yang (2012)
    Paper: "Flower pollination algorithm for global optimization"
           Unconventional Computation and Natural Computation,
           Lecture Notes in Computer Science, Vol. 7445, pp. 240-249 (2012).

Original MATLAB Function Signature:
    [best, fmin, N_iter] = fpa_demo(para)

Python Entry Point:
    fpa(n, N_iter, lb, ub, dim, fobj, p=0.8)

Parameters:
    n                 : int, population size (typically 10 to 25)
    N_iter            : int, total number of iterations
    lb                : float or numpy.ndarray of shape (dim,), lower search boundaries
    ub                : float or numpy.ndarray of shape (dim,), upper search boundaries
    dim               : int, dimensionality of search space
    fobj              : callable, objective function f(x) returning a scalar float
    p                 : float, switch probability between global and local pollination (default 0.8)

Returns:
    best              : numpy.ndarray of shape (dim,), optimal solution vector
    fmin              : float, global minimum objective value
    Convergence_curve : numpy.ndarray of shape (N_iter,), convergence history
"""

import math
import numpy as np


def _levy(d, beta=1.5):
    """
    Generate step from Levy distribution matching Yang's formula.
    """
    sigma = (math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0) /
             (math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))) ** (1.0 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    step = u / (np.abs(v) ** (1.0 / beta))
    return 0.01 * step


def fpa(n, N_iter, lb, ub, dim, fobj, p=0.8):
    """
    Execute the Flower Pollination Algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Sol = np.zeros((n, dim))
    Fitness = np.zeros(n)

    for i in range(n):
        Sol[i, :] = lb + (ub - lb) * np.random.rand(dim)
        Fitness[i] = fobj(Sol[i, :])

    best_idx = np.argmin(Fitness)
    fmin = Fitness[best_idx]
    best = Sol[best_idx, :].copy()

    Convergence_curve = np.zeros(N_iter)

    for t in range(1, N_iter + 1):
        for i in range(n):
            if np.random.rand() > p:
                # Global pollination via Levy flights
                L = _levy(dim, 1.5)
                dS = L * (Sol[i, :] - best)
                S = Sol[i, :] + dS
            else:
                # Local pollination
                epsilon = np.random.rand()
                JK = np.random.choice(n, size=2, replace=False)
                S = Sol[i, :] + epsilon * (Sol[JK[0], :] - Sol[JK[1], :])

            # Simple bounds check
            S = np.clip(S, lb, ub)

            Fnew = fobj(S)
            if Fnew <= Fitness[i]:
                Sol[i, :] = S
                Fitness[i] = Fnew

            if Fnew <= fmin:
                best = S.copy()
                fmin = Fnew

        Convergence_curve[t - 1] = fmin

    return best, fmin, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def fpa_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        pos, score, curve = fpa(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
        return score, pos, curve

    run_experiment(
        algo_name="FPA",
        algo_func=fpa_adapter,
        full_name="Flower Pollination Algorithm (FPA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

