"""
Snake Optimizer (SO)

Derived from:
    Sources(MATLAB)/Snake Optimizer (SO)/SO.m
    Authors: Fatma Hashim, Abdelazim G. Hussien (2022)
    Paper: "Snake Optimizer: A novel meta-heuristic optimization algorithm"
           Knowledge-Based Systems, 242, 108320.
           DOI: 10.1016/j.knosys.2022.108320

Original MATLAB Function Signature:
    [Xfood, fval, gbest_t] = SO(N, T, fobj, dim, lb, ub)

Python Entry Point:
    so(N, T, fobj, dim, lb, ub)

Parameters:
    N       : int, population size (total snakes)
    T       : int, maximum number of iterations
    fobj    : callable, objective function f(x) returning a scalar float
    dim     : int, dimensionality of the problem
    lb      : float or numpy.ndarray of shape (dim,), lower search boundaries
    ub      : float or numpy.ndarray of shape (dim,), upper search boundaries

Returns:
    Xfood   : numpy.ndarray of shape (dim,), optimal solution vector found
    fval    : float, global minimum objective value found
    gbest_t : numpy.ndarray of shape (T,), convergence history of best fitness
"""

import numpy as np


def so(N, T, fobj, dim, lb, ub):
    """
    Execute the Snake Optimizer algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    vec_flag = np.array([1.0, -1.0])
    Threshold = 0.25
    Threshold2 = 0.6
    C1 = 0.5
    C2 = 0.05
    C3 = 2.0
    eps = 1e-15

    X = lb + np.random.rand(N, dim) * (ub - lb)
    fitness = np.zeros(N)
    for i in range(N):
        fitness[i] = fobj(X[i, :])

    gbest = np.argmin(fitness)
    GYbest = fitness[gbest]
    Xfood = X[gbest, :].copy()

    Nm = int(round(N / 2.0))
    Nf = N - Nm
    Xm = X[:Nm, :].copy()
    Xf = X[Nm:N, :].copy()
    fitness_m = fitness[:Nm].copy()
    fitness_f = fitness[Nm:N].copy()

    gbest1 = np.argmin(fitness_m)
    fitnessBest_m = fitness_m[gbest1]
    Xbest_m = Xm[gbest1, :].copy()

    gbest2 = np.argmin(fitness_f)
    fitnessBest_f = fitness_f[gbest2]
    Xbest_f = Xf[gbest2, :].copy()

    gbest_t = np.zeros(T)

    for t in range(1, T + 1):
        Temp = np.exp(-t / T)
        Q = C1 * np.exp((t - T) / T)
        if Q > 1.0:
            Q = 1.0

        Xnewm = np.zeros((Nm, dim))
        Xnewf = np.zeros((Nf, dim))

        if Q < Threshold:
            # Exploration phase
            for i in range(Nm):
                rand_leader_idx = np.random.randint(0, Nm)
                X_randm = Xm[rand_leader_idx, :]
                Flag = vec_flag[np.random.randint(0, 2)]
                Am = np.exp(-fitness_m[rand_leader_idx] / (fitness_m[i] + eps))
                for j in range(dim):
                    Xnewm[i, j] = X_randm[j] + Flag * C2 * Am * ((ub[j] - lb[j]) * np.random.rand() + lb[j])

            for i in range(Nf):
                rand_leader_idx = np.random.randint(0, Nf)
                X_randf = Xf[rand_leader_idx, :]
                Flag = vec_flag[np.random.randint(0, 2)]
                Af = np.exp(-fitness_f[rand_leader_idx] / (fitness_f[i] + eps))
                for j in range(dim):
                    Xnewf[i, j] = X_randf[j] + Flag * C2 * Af * ((ub[j] - lb[j]) * np.random.rand() + lb[j])
        else:
            # Exploitation phase
            if Temp > Threshold2:
                # Hot
                for i in range(Nm):
                    Flag = vec_flag[np.random.randint(0, 2)]
                    for j in range(dim):
                        Xnewm[i, j] = Xfood[j] + C3 * Flag * Temp * np.random.rand() * (Xfood[j] - Xm[i, j])

                for i in range(Nf):
                    Flag = vec_flag[np.random.randint(0, 2)]
                    for j in range(dim):
                        Xnewf[i, j] = Xfood[j] + Flag * C3 * Temp * np.random.rand() * (Xfood[j] - Xf[i, j])
            else:
                # Cold
                if np.random.rand() > 0.6:
                    # Fight
                    for i in range(Nm):
                        FM = np.exp(-fitnessBest_f / (fitness_m[i] + eps))
                        for j in range(dim):
                            Xnewm[i, j] = Xm[i, j] + C3 * FM * np.random.rand() * (Q * Xbest_f[j] - Xm[i, j])

                    for i in range(Nf):
                        FF = np.exp(-fitnessBest_m / (fitness_f[i] + eps))
                        for j in range(dim):
                            Xnewf[i, j] = Xf[i, j] + C3 * FF * np.random.rand() * (Q * Xbest_m[j] - Xf[i, j])
                else:
                    # Mating
                    for i in range(Nm):
                        Mm_val = np.exp(-fitness_f[i % Nf] / (fitness_m[i] + eps))
                        for j in range(dim):
                            Xnewm[i, j] = Xm[i, j] + C3 * np.random.rand() * Mm_val * (Q * Xf[i % Nf, j] - Xm[i, j])

                    for i in range(Nf):
                        Mf_val = np.exp(-fitness_m[i % Nm] / (fitness_f[i] + eps))
                        for j in range(dim):
                            Xnewf[i, j] = Xf[i, j] + C3 * np.random.rand() * Mf_val * (Q * Xm[i % Nm, j] - Xf[i, j])

                    egg = vec_flag[np.random.randint(0, 2)]
                    if egg == 1.0:
                        gworst_m = np.argmax(fitness_m)
                        Xnewm[gworst_m, :] = lb + np.random.rand(dim) * (ub - lb)
                        gworst_f = np.argmax(fitness_f)
                        Xnewf[gworst_f, :] = lb + np.random.rand(dim) * (ub - lb)

        # Boundary checks and evaluations
        for j in range(Nm):
            Xnewm[j, :] = np.clip(Xnewm[j, :], lb, ub)
            y = fobj(Xnewm[j, :])
            if y < fitness_m[j]:
                fitness_m[j] = y
                Xm[j, :] = Xnewm[j, :].copy()

        Ybest1 = np.min(fitness_m)
        gbest1 = np.argmin(fitness_m)

        for j in range(Nf):
            Xnewf[j, :] = np.clip(Xnewf[j, :], lb, ub)
            y = fobj(Xnewf[j, :])
            if y < fitness_f[j]:
                fitness_f[j] = y
                Xf[j, :] = Xnewf[j, :].copy()

        Ybest2 = np.min(fitness_f)
        gbest2 = np.argmin(fitness_f)

        if Ybest1 < fitnessBest_m:
            Xbest_m = Xm[gbest1, :].copy()
            fitnessBest_m = Ybest1

        if Ybest2 < fitnessBest_f:
            Xbest_f = Xf[gbest2, :].copy()
            fitnessBest_f = Ybest2

        if Ybest1 < Ybest2:
            gbest_t[t - 1] = Ybest1
        else:
            gbest_t[t - 1] = Ybest2

        if fitnessBest_m < fitnessBest_f:
            GYbest = fitnessBest_m
            Xfood = Xbest_m.copy()
        else:
            GYbest = fitnessBest_f
            Xfood = Xbest_f.copy()

    fval = GYbest
    return Xfood, fval, gbest_t


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def so_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        pos, score, curve = so(SearchAgents_no, Max_iter, fobj, dim, lb, ub)
        return score, pos, curve

    run_experiment(
        algo_name="SO",
        algo_func=so_adapter,
        full_name="Snake Optimizer (SO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

