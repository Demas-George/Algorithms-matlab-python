"""
Artemisinin Optimizer (AO)
Source: Artemisinin Optimization based on Malaria Therapy: Algorithm and
        Applications to Medical Image Segmentation (Displays, Elsevier, 2024)
Authors: Chong Yuan, Dong Zhao, Ali Asghar Heidari, Lei Liu, Yi Chen, Zongda Wu, Huiling Chen
"""

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


def _mutation(z, x, b, dim):
    z_out = z.copy()
    for j in range(dim):
        if np.random.rand() < 0.05:
            z_out[j] = x[j]
        if np.random.rand() < 0.2:
            z_out[j] = b[j]
    return z_out


def _transborder_reset(z, ub, lb, dim, best):
    z_out = z.copy()
    for j in range(dim):
        high = ub if np.isscalar(ub) else ub[j]
        low = lb if np.isscalar(lb) else lb[j]
        if z_out[j] > high or z_out[j] < low:
            z_out[j] = best[j]
            if z_out[j] > high:
                z_out[j] = high
            elif z_out[j] < low:
                z_out[j] = low
    return z_out


def ao(N, Max_iter, lb, ub, dim, fobj):
    """
    Artemisinin Optimizer (AO)

    Parameters:
        N        : Population size (SearchAgents_no)
        Max_iter : Maximum number of iterations
        lb       : Lower bounds (scalar or array of shape (dim,))
        ub       : Upper bounds (scalar or array of shape (dim,))
        dim      : Problem dimensionality
        fobj     : Objective function handle f(x) -> float

    Returns:
        bestfitness       : Minimum objective value found
        Leader_pos        : Position vector of global optimum
        Convergence_curve : Best fitness history across iterations
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    MaxFEs = N * Max_iter
    FEs = 0
    it = 0

    pop = initialization(N, dim, ub_arr, lb_arr)
    Fitness = np.zeros(N)
    for i in range(N):
        Fitness[i] = fobj(pop[i, :])
        FEs += 1

    x_min = np.argmin(Fitness)
    bestfitness = float(Fitness[x_min])
    best = pop[x_min, :].copy()

    New_pop = np.zeros((N, dim))
    Fitnorm = np.zeros(N)
    Convergence_curve = np.zeros(Max_iter)

    while it < Max_iter and FEs <= MaxFEs:
        K = 1.0 - ((max(FEs, 1)) ** (1.0 / 6.0) / (MaxFEs ** (1.0 / 6.0)))
        E = 1.0 * np.exp(-4.0 * (FEs / MaxFEs))

        fit_min = np.min(Fitness)
        fit_max = np.max(Fitness)
        denom = (fit_max - fit_min) if (fit_max - fit_min) > 1e-16 else 1.0

        for i in range(N):
            Fitnorm[i] = (Fitness[i] - fit_min) / denom
            for j in range(dim):
                sign_val = (-1.0) ** (FEs % 2)
                if np.random.rand() < K:
                    if np.random.rand() < 0.5:
                        New_pop[i, j] = pop[i, j] + E * pop[i, j] * sign_val
                    else:
                        New_pop[i, j] = pop[i, j] + E * best[j] * sign_val
                else:
                    New_pop[i, j] = pop[i, j]

                if np.random.rand() < Fitnorm[i]:
                    A = np.random.permutation(N)
                    beta = (np.random.rand() / 2.0) + 0.1
                    New_pop[i, j] = pop[A[2], j] + beta * (pop[A[0], j] - pop[A[1], j])

            New_pop[i, :] = _mutation(New_pop[i, :], pop[i, :], best, dim)
            New_pop[i, :] = _transborder_reset(New_pop[i, :], ub_arr, lb_arr, dim, best)

            tFitness = fobj(New_pop[i, :])
            FEs += 1
            if tFitness < Fitness[i]:
                pop[i, :] = New_pop[i, :].copy()
                Fitness[i] = tFitness

        x_min = np.argmin(Fitness)
        if Fitness[x_min] < bestfitness:
            best = pop[x_min, :].copy()
            bestfitness = float(Fitness[x_min])

        Convergence_curve[it] = bestfitness
        it += 1

    # In case loop finished slightly early due to FEs cap
    if it < Max_iter:
        Convergence_curve[it:] = bestfitness

    Leader_pos = best
    return bestfitness, Leader_pos, Convergence_curve


AO = ao


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="AO",
        algo_func=ao,
        full_name="Artemisinin Optimizer (AO)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
