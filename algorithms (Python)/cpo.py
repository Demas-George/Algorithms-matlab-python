"""
Crested Porcupine Optimizer (CPO)

Derived from:
    Sources(MATLAB)/CPO/CPO/CPO.m
    Authors: Reda Mohamed, Mohamed Abdel-Basset (2024)
    Paper: "Crested Porcupine Optimizer: A new nature-inspired metaheuristic"
           Knowledge-Based Systems, 284, 111257.
           DOI: 10.1016/j.knosys.2023.111257

Original MATLAB Function Signature:
    [Gb_Fit, Gb_Sol, Conv_curve] = CPO(Pop_size, Tmax, ub, lb, dim, fobj, fhd)

Python Entry Point:
    cpo(Pop_size, Tmax, ub, lb, dim, fobj, fhd=None)

Parameters:
    Pop_size   : int, initial population size of crested porcupines
    Tmax       : int, maximum number of function evaluations
    ub         : float or numpy.ndarray of shape (dim,), upper search boundaries
    lb         : float or numpy.ndarray of shape (dim,), lower search boundaries
    dim        : int, dimensionality of the search space
    fobj       : callable or function ID, objective function to minimize (f(x) -> float)
    fhd        : callable, optional benchmark suite evaluator (e.g. cec_func(x, fobj))

Returns:
    Gb_Fit     : float, best-so-far fitness score
    Gb_Sol     : numpy.ndarray of shape (dim,), best-so-far position vector
    Conv_curve : numpy.ndarray of shape (Tmax,), convergence curve recording Gb_Fit at each evaluation
"""

import numpy as np


def cpo(Pop_size, Tmax, ub, lb, dim, fobj, fhd=None):
    """
    Execute the Crested Porcupine Optimizer algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    def _eval(x):
        if fhd is not None:
            return float(fhd(x.reshape(-1, 1), fobj))
        return float(fobj(x))

    N = Pop_size
    N_min = min(Pop_size, 120)
    T = 2  # Number of cycles
    alpha = 0.2  # Convergence rate
    Tf = 0.8  # Tradeoff percentage between 3rd and 4th defense
    eps = 1e-15

    # Initialization
    X = np.zeros((Pop_size, dim))
    for i in range(dim):
        X[:, i] = np.random.rand(Pop_size) * (ub[i] - lb[i]) + lb[i]

    fitness = np.zeros(Pop_size)
    for i in range(Pop_size):
        fitness[i] = _eval(X[i, :])

    index = np.argmin(fitness)
    Gb_Fit = fitness[index]
    Gb_Sol = X[index, :].copy()
    Xp = X.copy()

    Conv_curve = np.zeros(Tmax)
    t = 0
    cur_pop = Pop_size

    while t < Tmax:
        r2 = np.random.rand()
        for i in range(cur_pop):
            U1 = (np.random.rand(dim) > np.random.rand()).astype(float)

            if np.random.rand() < np.random.rand():
                # Exploration phase
                if np.random.rand() < np.random.rand():
                    # First defense mechanism
                    rand_idx = np.random.randint(0, cur_pop)
                    y = (X[i, :] + X[rand_idx, :]) / 2.0
                    X[i, :] = X[i, :] + np.random.randn() * np.abs(2.0 * np.random.rand() * Gb_Sol - y)
                else:
                    # Second defense mechanism
                    rand_idx = np.random.randint(0, cur_pop)
                    y = (X[i, :] + X[rand_idx, :]) / 2.0
                    r_a = np.random.randint(0, cur_pop)
                    r_b = np.random.randint(0, cur_pop)
                    X[i, :] = U1 * X[i, :] + (1.0 - U1) * (y + np.random.rand() * (X[r_a, :] - X[r_b, :]))
            else:
                # Exploitation phase
                t_ratio = t / Tmax
                Yt = 2.0 * np.random.rand() * ((1.0 - t_ratio) ** t_ratio)
                U2 = (np.random.rand(dim) < (0.5 * 2.0 - 1.0)).astype(float)
                S = np.random.rand() * U2

                fit_sum = np.sum(fitness[:cur_pop]) + eps
                if np.random.rand() < Tf:
                    # Third defense mechanism
                    St = np.exp(fitness[i] / fit_sum)
                    S = S * Yt * St
                    r_a = np.random.randint(0, cur_pop)
                    r_b = np.random.randint(0, cur_pop)
                    r_c = np.random.randint(0, cur_pop)
                    X[i, :] = ((1.0 - U1) * X[i, :]
                               + U1 * (X[r_a, :] + St * (X[r_b, :] - X[r_c, :]) - S))
                else:
                    # Fourth defense mechanism
                    Mt = np.exp(fitness[i] / fit_sum)
                    vt = X[i, :]
                    Vtp = X[np.random.randint(0, cur_pop), :]
                    Ft = np.random.rand(dim) * (Mt * (-vt + Vtp))
                    S = S * Yt * Ft
                    X[i, :] = (Gb_Sol + (alpha * (1.0 - r2) + r2) * (U2 * Gb_Sol - X[i, :])) - S

            # Boundary check
            out_bounds = (X[i, :] > ub) | (X[i, :] < lb)
            if np.any(out_bounds):
                rand_reset = lb + np.random.rand(dim) * (ub - lb)
                X[i, :] = np.where(out_bounds, rand_reset, X[i, :])

            nF = _eval(X[i, :])

            if fitness[i] < nF:
                X[i, :] = Xp[i, :].copy()
            else:
                Xp[i, :] = X[i, :].copy()
                fitness[i] = nF
                if fitness[i] <= Gb_Fit:
                    Gb_Sol = X[i, :].copy()
                    Gb_Fit = fitness[i]

            Conv_curve[t] = Gb_Fit
            t += 1
            if t >= Tmax:
                break

        cycle_period = Tmax / T
        cur_pop = int(np.fix(N_min + (N - N_min) * (1.0 - ((t % cycle_period) / cycle_period))))
        cur_pop = max(1, min(cur_pop, Pop_size))

    return Gb_Fit, Gb_Sol, Conv_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def cpo_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        return cpo(SearchAgents_no, Max_iter, ub, lb, dim, fobj)

    run_experiment(
        algo_name="CPO",
        algo_func=cpo_adapter,
        full_name="Crested Porcupine Optimizer (CPO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

