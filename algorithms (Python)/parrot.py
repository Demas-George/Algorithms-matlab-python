"""
Parrot Optimizer (PO)
Source: Parrot optimizer: Algorithm and applications to medical problems
        Computers in Biology and Medicine, Elsevier, 2024.
Authors: Junbo Lian, Guohua Hui, Ling Ma, Ting Zhu, Xincan Wu, Ali Asghar Heidari, Yi Chen, Huiling Chen
"""

import math
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


def _levy(d):
    beta = 1.5
    num = math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma = (num / den) ** (1.0 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    denom = np.abs(v) ** (1.0 / beta)
    denom[denom == 0] = 1e-16
    return u / denom


def parrot(N, Max_iter, lb, ub, dim, fobj):
    """
    Parrot Optimizer (PO)

    Parameters:
        N        : Population size
        Max_iter : Maximum number of iterations
        lb       : Lower bound(s)
        ub       : Upper bound(s)
        dim      : Problem dimensionality
        fobj     : Objective function handle f(x) -> float

    Returns:
        Best_score : Global best objective value
        Best_pos   : Global best position
        curve      : Convergence curve over iterations
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    X = initialization(N, dim, ub_arr, lb_arr)
    fitness = np.zeros(N)
    for i in range(N):
        fitness[i] = fobj(X[i, :])

    idx = np.argsort(fitness)
    fitness = fitness[idx]
    X = X[idx, :]

    GBestF = float(fitness[0])
    GBestX = X[0, :].copy()

    curve = np.zeros(Max_iter)
    X_new = X.copy()

    for i in range(1, Max_iter + 1):
        alpha = np.random.rand() / 5.0
        theta = np.random.rand() * np.pi
        mean_X = np.mean(X, axis=0)

        for j in range(N):
            St = np.random.randint(1, 5)

            # 1. Foraging behavior
            if St == 1:
                decay = (1.0 - i / Max_iter) ** (2.0 * i / Max_iter)
                X_new[j, :] = (X[j, :] - GBestX) * _levy(dim) + np.random.rand() * mean_X * decay

            # 2. Staying behavior
            elif St == 2:
                X_new[j, :] = X[j, :] + GBestX * _levy(dim) + np.random.randn() * (1.0 - i / Max_iter) * np.ones(dim)

            # 3. Communicating behavior
            elif St == 3:
                H = np.random.rand()
                if H < 0.5:
                    X_new[j, :] = X[j, :] + alpha * (1.0 - i / Max_iter) * (X[j, :] - mean_X)
                else:
                    rand_val = max(np.random.rand(), 1e-12)
                    X_new[j, :] = X[j, :] + alpha * (1.0 - i / Max_iter) * np.exp(-(j + 1) / (rand_val * Max_iter))

            # 4. Fear of strangers' behavior
            else:
                ratio = (i / Max_iter) ** (2.0 / Max_iter)
                X_new[j, :] = (
                    X[j, :]
                    + np.random.rand() * np.cos((np.pi * i) / (2.0 * Max_iter)) * (GBestX - X[j, :])
                    - np.cos(theta) * ratio * (X[j, :] - GBestX)
                )

            X_new[j, :] = np.clip(X_new[j, :], lb_arr, ub_arr)
            fit_j = fobj(X_new[j, :])
            if fit_j < GBestF:
                GBestF = fit_j
                GBestX = X_new[j, :].copy()

        fitness_new = np.zeros(N)
        for s in range(N):
            fitness_new[s] = fobj(X_new[s, :])
            if fitness_new[s] < GBestF:
                GBestF = fitness_new[s]
                GBestX = X_new[s, :].copy()

        X = X_new.copy()
        fitness = fitness_new

        idx = np.argsort(fitness)
        fitness = fitness[idx]
        X = X[idx, :]

        curve[i - 1] = GBestF

    Best_score = GBestF
    Best_pos = GBestX
    return Best_score, Best_pos, curve


PO_PARROT = parrot
PARROT = parrot


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="PARROT",
        algo_func=parrot,
        full_name="Parrot Optimizer (PO)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
