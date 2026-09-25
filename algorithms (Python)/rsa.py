"""
Reptile Search Algorithm (RSA)

Derived from:
    Sources(MATLAB)/Reptile-Search-Algorithm (RSA)/Reptile-Search-Algorithm-RSA-A-nature-inspired-optimizer-main/RSA.m
    Author: Laith Abualigah (2021)
    Paper: "Reptile Search Algorithm (RSA): A novel nature-inspired metaheuristic algorithm"
           Expert Systems with Applications, 191, 116158.
           DOI: 10.1016/j.eswa.2021.116158

Original MATLAB Function Signature:
    [Best_F, Best_P, Conv] = RSA(N, T, LB, UB, Dim, F_obj)

Python Entry Point:
    rsa(N, T, LB, UB, Dim, F_obj)

Parameters:
    N     : int, population size (number of crocodiles)
    T     : int, maximum number of iterations
    LB    : float or numpy.ndarray of shape (Dim,), lower boundaries
    UB    : float or numpy.ndarray of shape (Dim,), upper boundaries
    Dim   : int, dimensionality of the problem
    F_obj : callable, objective function f(x) returning a scalar float

Returns:
    Best_F : float, best objective value (fitness score) found
    Best_P : numpy.ndarray of shape (Dim,), optimal position vector
    Conv   : numpy.ndarray of shape (T,), convergence history
"""

import numpy as np


def rsa(N, T, LB, UB, Dim, F_obj):
    """
    Execute the Reptile Search Algorithm.
    """
    LB = np.full(Dim, LB, dtype=float) if np.isscalar(LB) else np.asarray(LB, dtype=float)
    UB = np.full(Dim, UB, dtype=float) if np.isscalar(UB) else np.asarray(UB, dtype=float)

    Best_P = np.zeros(Dim)
    Best_F = np.inf

    X = np.zeros((N, Dim))
    for i in range(Dim):
        X[:, i] = np.random.rand(N) * (UB[i] - LB[i]) + LB[i]

    Xnew = np.zeros((N, Dim))
    Conv = np.zeros(T)

    Alpha = 0.1
    Beta = 0.005
    eps = 1e-15

    Ffun = np.zeros(N)
    for i in range(N):
        Ffun[i] = F_obj(X[i, :])
        if Ffun[i] < Best_F:
            Best_F = Ffun[i]
            Best_P = X[i, :].copy()

    for t in range(1, T + 1):
        ES = 2.0 * np.random.choice([-1, 0, 1]) * (1.0 - (t / T))

        # In MATLAB, loop runs from agent 2 to N (1-based), preserving agent 1
        for i in range(1, N):
            x_mean = np.mean(X[i, :])
            for j in range(Dim):
                rand_agent = np.random.randint(0, N)
                R = Best_P[j] - X[rand_agent, j] / (Best_P[j] + eps)
                P = Alpha + (X[i, j] - x_mean) / (Best_P[j] * (UB[j] - LB[j]) + eps)
                Eta = Best_P[j] * P

                if t < T / 4.0:
                    Xnew[i, j] = Best_P[j] - Eta * Beta - R * np.random.rand()
                elif t < 2.0 * T / 4.0:
                    rand_agent_2 = np.random.randint(0, N)
                    Xnew[i, j] = Best_P[j] * X[rand_agent_2, j] * ES * np.random.rand()
                elif t < 3.0 * T / 4.0:
                    Xnew[i, j] = Best_P[j] * P * np.random.rand()
                else:
                    Xnew[i, j] = Best_P[j] - Eta * eps - R * np.random.rand()

            Xnew[i, :] = np.clip(Xnew[i, :], LB, UB)
            Ffun_new = F_obj(Xnew[i, :])

            if Ffun_new < Ffun[i]:
                X[i, :] = Xnew[i, :].copy()
                Ffun[i] = Ffun_new

            if Ffun[i] < Best_F:
                Best_F = Ffun[i]
                Best_P = X[i, :].copy()

        Conv[t - 1] = Best_F

    return Best_F, Best_P, Conv


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="RSA",
        algo_func=rsa,
        full_name="Reptile Search Algorithm (RSA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

