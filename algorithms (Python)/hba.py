"""
Honey Badger Algorithm (HBA)
Source: Honey Badger Algorithm: New Metaheuristic Algorithm for Solving Optimization Problems
        Mathematics and Computers in Simulation, 2021.
Authors: Fatma A. Hashim, Essam H. Houssein, Kashif Hussain, Mai S. Mabrouk, Walid Al-Atabany
"""

import numpy as np


def initialization(N, dim, ub, lb):
    ub = np.array(ub, dtype=float)
    lb = np.array(lb, dtype=float)
    if ub.size == 1:
        Positions = np.random.rand(N, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((N, dim))
        for i in range(dim):
            Positions[:, i] = np.random.rand(N) * (ub[i] - lb[i]) + lb[i]
    return Positions


def _intensity(N, Xprey, X):
    eps = 1e-16
    di = np.zeros(N)
    S = np.zeros(N)
    for i in range(N - 1):
        di[i] = (np.linalg.norm(X[i, :] - Xprey + eps)) ** 2
        S[i]  = (np.linalg.norm(X[i, :] - X[i + 1, :] + eps)) ** 2
    di[N - 1] = (np.linalg.norm(X[N - 1, :] - Xprey + eps)) ** 2
    S[N - 1]  = (np.linalg.norm(X[N - 1, :] - X[0, :] + eps)) ** 2

    I = np.zeros(N)
    for i in range(N):
        r2 = np.random.rand()
        I[i] = r2 * S[i] / (4.0 * np.pi * di[i] + eps)
    return I


def hba(N, tmax, lb, ub, dim, fobj):
    """
    Honey Badger Algorithm (HBA)

    Parameters:
        N    : Population size
        tmax : Maximum number of iterations
        lb   : Lower bound(s)
        ub   : Upper bound(s)
        dim  : Dimension of problem
        fobj : Objective function f(x) -> float

    Returns:
        Food_Score : Global best fitness
        Xprey      : Global best position
        CNVG       : Convergence curve history
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    beta = 6.0
    C = 2.0
    vec_flag = np.array([1, -1])

    X = initialization(N, dim, ub_arr, lb_arr)
    fitness = np.zeros(N)
    for i in range(N):
        fitness[i] = fobj(X[i, :])

    gbest = np.argmin(fitness)
    GYbest = float(fitness[gbest])
    Xprey = X[gbest, :].copy()

    CNVG = np.zeros(tmax)
    Xnew = np.zeros((N, dim))

    for t in range(1, tmax + 1):
        alpha = C * np.exp(-t / tmax)
        I = _intensity(N, Xprey, X)

        for i in range(N):
            r = np.random.rand()
            F = vec_flag[np.random.randint(0, 2)]

            for j in range(dim):
                di = Xprey[j] - X[i, j]
                if r < 0.5:
                    r3 = np.random.rand()
                    r4 = np.random.rand()
                    r5 = np.random.rand()
                    Xnew[i, j] = (
                        Xprey[j]
                        + F * beta * I[i] * Xprey[j]
                        + F * r3 * alpha * di * np.abs(np.cos(2.0 * np.pi * r4) * (1.0 - np.cos(2.0 * np.pi * r5)))
                    )
                else:
                    r7 = np.random.rand()
                    Xnew[i, j] = Xprey[j] + F * r7 * alpha * di

            Xnew[i, :] = np.clip(Xnew[i, :], lb_arr, ub_arr)
            tempFitness = fobj(Xnew[i, :])
            if tempFitness < fitness[i]:
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :].copy()

        X = np.clip(X, lb_arr, ub_arr)
        best_idx = np.argmin(fitness)
        Ybest = float(fitness[best_idx])
        CNVG[t - 1] = Ybest

        if Ybest < GYbest:
            GYbest = Ybest
            Xprey = X[best_idx, :].copy()

    Food_Score = GYbest
    return Food_Score, Xprey, CNVG


HBA = hba


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="HBA",
        algo_func=hba,
        full_name="Honey Badger Algorithm (HBA)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
