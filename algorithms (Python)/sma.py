"""
Slime Mould Algorithm (SMA)

Derived from:
    Sources(MATLAB)/Slime mould algorithm (SMA)/Slime mould algorithm (SMA)-2020/SMA.m
    Authors: Shimin Li, Huiling Chen, Mingjing Wang, Ali Asghar Heidari, Seyedali Mirjalili (2020)
    Paper: "Slime Mould Algorithm: A New Method for Stochastic Optimization"
           Future Generation Computer Systems, 111, pp. 300-323.
           DOI: 10.1016/j.future.2020.03.055

Original MATLAB Function Signature:
    [Destination_fitness, bestPositions, Convergence_curve] = SMA(N, Max_iter, lb, ub, dim, fobj)

Python Entry Point:
    sma(N, Max_iter, lb, ub, dim, fobj)

Parameters:
    N                   : int, population size of slime mould
    Max_iter            : int, maximum number of iterations
    lb                  : float or numpy.ndarray of shape (dim,), lower search boundaries
    ub                  : float or numpy.ndarray of shape (dim,), upper search boundaries
    dim                 : int, dimensionality of the search space
    fobj                : callable, objective function f(x) returning a scalar float

Returns:
    Destination_fitness : float, optimal objective value found
    bestPositions       : numpy.ndarray of shape (dim,), optimal solution vector
    Convergence_curve   : numpy.ndarray of shape (Max_iter,), convergence history
"""

import numpy as np


def sma(N, Max_iter, lb, ub, dim, fobj):
    """
    Execute the Slime Mould Algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    bestPositions = np.zeros(dim)
    Destination_fitness = np.inf
    AllFitness = np.full(N, np.inf)
    weight = np.ones((N, dim))

    X = np.zeros((N, dim))
    for i in range(dim):
        X[:, i] = np.random.rand(N) * (ub[i] - lb[i]) + lb[i]

    Convergence_curve = np.zeros(Max_iter)
    eps = 1e-15
    z = 0.03

    it = 1
    while it <= Max_iter:
        X = np.clip(X, lb, ub)
        for i in range(N):
            AllFitness[i] = fobj(X[i, :])

        SmellIndex = np.argsort(AllFitness)
        SmellOrder = AllFitness[SmellIndex]
        worstFitness = SmellOrder[-1]
        bestFitness = SmellOrder[0]

        S = bestFitness - worstFitness + eps

        for i in range(N):
            for j in range(dim):
                if (i + 1) <= (N / 2):
                    weight[SmellIndex[i], j] = 1.0 + np.random.rand() * np.log10((bestFitness - SmellOrder[i]) / S + 1.0)
                else:
                    weight[SmellIndex[i], j] = 1.0 - np.random.rand() * np.log10((bestFitness - SmellOrder[i]) / S + 1.0)

        if bestFitness < Destination_fitness:
            bestPositions = X[SmellIndex[0], :].copy()
            Destination_fitness = bestFitness

        # a = atanh(-(it/Max_iter) + 1); bound argument away from 1 and -1
        tanh_arg = -(it / Max_iter) + 1.0
        tanh_arg = max(-0.9999999, min(0.9999999, tanh_arg))
        a = np.arctanh(tanh_arg)
        b = 1.0 - it / Max_iter

        for i in range(N):
            if np.random.rand() < z:
                X[i, :] = (ub - lb) * np.random.rand() + lb
            else:
                p = np.tanh(abs(AllFitness[i] - Destination_fitness))
                vb = np.random.uniform(-a, a, size=dim)
                vc = np.random.uniform(-b, b, size=dim)
                for j in range(dim):
                    r = np.random.rand()
                    A = np.random.randint(0, N)
                    B = np.random.randint(0, N)
                    if r < p:
                        X[i, j] = bestPositions[j] + vb[j] * (weight[i, j] * X[A, j] - X[B, j])
                    else:
                        X[i, j] = vc[j] * X[i, j]

        Convergence_curve[it - 1] = Destination_fitness
        it += 1

    return Destination_fitness, bestPositions, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="SMA",
        algo_func=sma,
        full_name="Slime Mould Algorithm (SMA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

