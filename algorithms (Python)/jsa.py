"""
Jellyfish Search Algorithm (JSA)
Source: A Novel Metaheuristic Optimizer Inspired By Behavior of Jellyfish in Ocean
        Applied Mathematics and Computation, 2021.
Authors: Jui-Sheng Chou, Dinh-Nhat Truong
"""

import numpy as np


def _logistic_initialization(num_pop, nd, ub, lb):
    x = np.zeros((num_pop, nd))
    x[0, :] = np.random.rand(nd)
    a = 4.0
    for i in range(num_pop - 1):
        x[i + 1, :] = a * x[i, :] * (1.0 - x[i, :])
    pop = lb + x * (ub - lb)
    return pop


def _simplebounds(s, lb, ub):
    ns_tmp = s.copy()
    max_loops = 10
    loops = 0
    I = ns_tmp < lb
    while np.any(I) and loops < max_loops:
        ns_tmp[I] = ub[I] + (ns_tmp[I] - lb[I])
        I = ns_tmp < lb
        loops += 1

    loops = 0
    J = ns_tmp > ub
    while np.any(J) and loops < max_loops:
        ns_tmp[J] = lb[J] + (ns_tmp[J] - ub[J])
        J = ns_tmp > ub
        loops += 1

    return np.clip(ns_tmp, lb, ub)


def jsa(nPop, MaxIt, lb, ub, dim, fobj):
    """
    Jellyfish Search Optimizer (JS / JSA)

    Parameters:
        nPop  : Population size
        MaxIt : Maximum number of iterations
        lb    : Lower bound(s)
        ub    : Upper bound(s)
        dim   : Dimensionality
        fobj  : Objective function handle f(x) -> float

    Returns:
        BestCost : Optimal objective value found
        BestSol  : Optimal position vector
        fbestvl  : Best fitness convergence curve
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    popi = _logistic_initialization(nPop, dim, ub_arr, lb_arr)
    popCost = np.zeros(nPop)
    for i in range(nPop):
        popCost[i] = fobj(popi[i, :])

    best_idx = np.argmin(popCost)
    BestSol = popi[best_idx, :].copy()
    BestCost = float(popCost[best_idx])

    fbestvl = np.zeros(MaxIt)

    for it in range(1, MaxIt + 1):
        Meanvl = np.mean(popi, axis=0)
        best_idx = np.argmin(popCost)
        BestSol = popi[best_idx, :].copy()
        BestCost = float(popCost[best_idx])

        for i in range(nPop):
            Ar = (1.0 - it / MaxIt) * (2.0 * np.random.rand() - 1.0)
            if abs(Ar) >= 0.5:
                # Ocean current motion
                newsol = popi[i, :] + np.random.rand(dim) * (BestSol - 3.0 * np.random.rand() * Meanvl)
            else:
                # Moving inside swarm
                if np.random.rand() <= (1.0 - Ar):
                    # Active motion (Type B)
                    j = i
                    while j == i:
                        j = np.random.randint(0, nPop)
                    Step = popi[i, :] - popi[j, :]
                    if popCost[j] < popCost[i]:
                        Step = -Step
                    newsol = popi[i, :] + np.random.rand(dim) * Step
                else:
                    # Passive motion (Type A)
                    newsol = popi[i, :] + 0.1 * (ub_arr - lb_arr) * np.random.rand()

            newsol = _simplebounds(newsol, lb_arr, ub_arr)
            newsolCost = fobj(newsol)

            if newsolCost < popCost[i]:
                popi[i, :] = newsol.copy()
                popCost[i] = newsolCost
                if popCost[i] < BestCost:
                    BestCost = float(popCost[i])
                    BestSol = popi[i, :].copy()

        fbestvl[it - 1] = BestCost

    return BestCost, BestSol, fbestvl


JSA = jsa
JS = jsa


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="JSA",
        algo_func=jsa,
        full_name="Jellyfish Search Algorithm (JSA)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
