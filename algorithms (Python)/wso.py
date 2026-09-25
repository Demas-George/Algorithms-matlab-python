"""
White Shark Optimizer (WSO)
Source: White Shark Optimizer: A novel bio-inspired meta-heuristic algorithm
        for global optimization problems (Knowledge-Based Systems, 2022)
Authors: Malik Braik, Abdelaziz Hammouri, Jaffar Atwan, Mohammed Azmi Al-Betar, Mohammed A. Awadallah
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


def wso(whiteSharks, itemax, lb, ub, dim, fobj):
    """
    White Shark Optimizer (WSO)

    Parameters:
        whiteSharks : Number of white sharks (population size)
        itemax      : Maximum iterations
        lb          : Lower bound(s)
        ub          : Upper bound(s)
        dim         : Problem dimensionality
        fobj        : Objective function f(x) -> float

    Returns:
        fmin0  : Global best fitness
        gbest  : Global best position
        ccurve : Convergence curve history
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    ccurve = np.zeros(itemax)
    WSO_Positions = initialization(whiteSharks, dim, ub_arr, lb_arr)
    v = np.zeros((whiteSharks, dim))

    fit = np.zeros(whiteSharks)
    for i in range(whiteSharks):
        fit[i] = fobj(WSO_Positions[i, :])

    fitness = fit.copy()
    index = np.argmin(fit)
    fmin0 = float(fit[index])

    wbest = WSO_Positions.copy()
    gbest = WSO_Positions[index, :].copy()

    fmax = 0.75
    fmin = 0.07
    tau = 4.11
    sqrt_term = np.sqrt(max(0.0, tau ** 2 - 4.0 * tau))
    mu = 2.0 / abs(2.0 - tau - sqrt_term)

    a0 = 6.250
    a1 = 100.0
    a2 = 0.0005

    for ite in range(1, itemax + 1):
        mv = 1.0 / (a0 + np.exp((itemax / 2.0 - ite) / a1))
        s_s = abs(1.0 - np.exp(-a2 * ite / itemax))

        nu = np.random.randint(0, whiteSharks, size=whiteSharks)

        # ── Update speed ─────────────────────────────────────────
        for i in range(whiteSharks):
            rmin, rmax = 1.0, 3.0
            rr = rmin + np.random.rand() * (rmax - rmin)
            wr = abs(((2.0 * np.random.rand()) - (np.random.rand() + np.random.rand())) / rr)
            v[i, :] = mu * v[i, :] + wr * (wbest[nu[i], :] - WSO_Positions[i, :])

        # ── Update white shark position ──────────────────────────
        f = fmin + (fmax - fmin) / (fmax + fmin)
        for i in range(whiteSharks):
            a_flag = WSO_Positions[i, :] > ub_arr
            b_flag = WSO_Positions[i, :] < lb_arr
            wo = np.logical_xor(a_flag, b_flag)

            if np.random.rand() < mv:
                WSO_Positions[i, :] = WSO_Positions[i, :] * (~wo) + (ub_arr * a_flag + lb_arr * b_flag)
            else:
                WSO_Positions[i, :] = WSO_Positions[i, :] + v[i, :] / f

        # ── Update position considering school ───────────────────
        for i in range(whiteSharks):
            for j in range(dim):
                if np.random.rand() < s_s:
                    dist = abs(np.random.rand() * (gbest[j] - WSO_Positions[i, j]))
                    sign_rnd = np.sign(np.random.rand() - 0.5)
                    if i == 0:
                        WSO_Positions[i, j] = gbest[j] + np.random.rand() * dist * sign_rnd
                    else:
                        wso_pos_ij = gbest[j] + np.random.rand() * dist * sign_rnd
                        WSO_Positions[i, j] = (wso_pos_ij + WSO_Positions[i - 1, j]) / 2.0 * np.random.rand()

        # ── Evaluation and best update ───────────────────────────
        WSO_Positions = np.clip(WSO_Positions, lb_arr, ub_arr)
        for i in range(whiteSharks):
            fit_i = fobj(WSO_Positions[i, :])
            if fit_i < fitness[i]:
                wbest[i, :] = WSO_Positions[i, :].copy()
                fitness[i] = fit_i

            if fitness[i] < fmin0:
                fmin0 = float(fitness[i])
                gbest = wbest[i, :].copy()

        ccurve[ite - 1] = fmin0

    return fmin0, gbest, ccurve


WSO = wso


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="WSO",
        algo_func=wso,
        full_name="White Shark Optimizer (WSO)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
