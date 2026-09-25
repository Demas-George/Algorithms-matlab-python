"""
Harris Hawks Optimization (HHO)

Derived from:
    Sources(MATLAB)/Harris-Hawks-Optimization-Algorithm (HHO)/Harris-Hawks-Optimization-Algorithm-and-Applications-master/HHO.m
    Authors: Ali Asghar Heidari, Seyedali Mirjalili, Hossam Faris, Ibrahim Aljarah, Majdi Mafarja, Huiling Chen (2019)
    Paper: "Harris hawks optimization: Algorithm and applications"
           Future Generation Computer Systems, 97, pp. 849-872.
           DOI: 10.1016/j.future.2019.02.028

Original MATLAB Function Signature:
    [Rabbit_Energy, Rabbit_Location, CNVG] = HHO(N, T, lb, ub, dim, fobj)

Python Entry Point:
    hho(N, T, lb, ub, dim, fobj)

Parameters:
    N               : int, population size (number of hawks)
    T               : int, maximum number of iterations
    lb              : float or numpy.ndarray of shape (dim,), lower search boundaries
    ub              : float or numpy.ndarray of shape (dim,), upper search boundaries
    dim             : int, problem dimension
    fobj            : callable, objective function f(x) returning a scalar float

Returns:
    Rabbit_Energy   : float, best objective value (minimum energy of rabbit)
    Rabbit_Location : numpy.ndarray of shape (dim,), optimal position vector
    CNVG            : numpy.ndarray of shape (T,), convergence history
"""

import math
import numpy as np


def _levy(d, beta=1.5):
    sigma = (math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0) /
             (math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))) ** (1.0 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    return u / (np.abs(v) ** (1.0 / beta))


def hho(N, T, lb, ub, dim, fobj):
    """
    Execute the Harris Hawks Optimization algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = np.inf

    X = np.zeros((N, dim))
    for i in range(dim):
        X[:, i] = np.random.rand(N) * (ub[i] - lb[i]) + lb[i]

    CNVG = np.zeros(T)

    for t in range(T):
        for i in range(N):
            X[i, :] = np.clip(X[i, :], lb, ub)
            fitness = fobj(X[i, :])
            if fitness < Rabbit_Energy:
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        E1 = 2.0 * (1.0 - (t / T))

        for i in range(N):
            E0 = 2.0 * np.random.rand() - 1.0
            Escaping_Energy = E1 * E0

            if abs(Escaping_Energy) >= 1.0:
                # Exploration
                q = np.random.rand()
                rand_Hawk_index = np.random.randint(0, N)
                X_rand = X[rand_Hawk_index, :]

                if q < 0.5:
                    X[i, :] = X_rand - np.random.rand() * np.abs(X_rand - 2.0 * np.random.rand() * X[i, :])
                else:
                    X[i, :] = ((Rabbit_Location - np.mean(X, axis=0))
                               - np.random.rand() * ((ub - lb) * np.random.rand() + lb))
            else:
                # Exploitation
                r = np.random.rand()

                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    # Hard besiege
                    X[i, :] = Rabbit_Location - Escaping_Energy * np.abs(Rabbit_Location - X[i, :])

                elif r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    # Soft besiege
                    Jump_strength = 2.0 * (1.0 - np.random.rand())
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])

                elif r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    # Soft besiege with rapid dives
                    Jump_strength = 2.0 * (1.0 - np.random.rand())
                    X1 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])

                    if fobj(X1) < fobj(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = (Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - X[i, :])
                              + np.random.rand(dim) * _levy(dim))
                        if fobj(X2) < fobj(X[i, :]):
                            X[i, :] = X2

                elif r < 0.5 and abs(Escaping_Energy) < 0.5:
                    # Hard besiege with rapid dives
                    Jump_strength = 2.0 * (1.0 - np.random.rand())
                    X1 = Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - np.mean(X, axis=0))

                    if fobj(X1) < fobj(X[i, :]):
                        X[i, :] = X1
                    else:
                        X2 = (Rabbit_Location - Escaping_Energy * np.abs(Jump_strength * Rabbit_Location - np.mean(X, axis=0))
                              + np.random.rand(dim) * _levy(dim))
                        if fobj(X2) < fobj(X[i, :]):
                            X[i, :] = X2

        CNVG[t] = Rabbit_Energy

    return Rabbit_Energy, Rabbit_Location, CNVG


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="HHO",
        algo_func=hho,
        full_name="Harris Hawks Optimization (HHO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

