"""
Kepler Optimization Algorithm (KOA)

Derived from:
    Sources(MATLAB)/Kepler-Optimization-Algorithm (KOA)/Kepler-Optimization-Algorithm-main/KOA.m
    Authors: Reda Mohamed, Mohamed Abdel-Basset (2023)
    Paper: "Kepler optimization algorithm: A new metaheuristic algorithm inspired by
           Kepler's laws of planetary motion"
           Knowledge-Based Systems, 268, 110454.
           DOI: 10.1016/j.knosys.2023.110454

Original MATLAB Function Signature:
    [Sun_Score, Sun_Pos, Convergence_curve] = KOA(SearchAgents_no, Tmax, ub, lb, dim, fobj, fhd)

Python Entry Point:
    koa(SearchAgents_no, Tmax, ub, lb, dim, fobj, fhd=None)

Parameters:
    SearchAgents_no   : int, number of planets (search agents)
    Tmax              : int, maximum number of function evaluations
    ub                : float or numpy.ndarray of shape (dim,), upper boundaries
    lb                : float or numpy.ndarray of shape (dim,), lower boundaries
    dim               : int, dimensionality of the problem
    fobj              : callable or function ID, objective function to minimize (f(x) -> float)
    fhd               : callable, optional benchmark evaluator suite

Returns:
    Sun_Score         : float, best fitness score found (representing the Sun)
    Sun_Pos           : numpy.ndarray of shape (dim,), optimal decision vector
    Convergence_curve : numpy.ndarray of shape (Tmax,), best score at each evaluation
"""

import numpy as np


def koa(SearchAgents_no, Tmax, ub, lb, dim, fobj, fhd=None):
    """
    Execute the Kepler Optimization Algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    def _eval(x):
        if fhd is not None:
            return float(fhd(x.reshape(-1, 1), fobj))
        return float(fobj(x))

    Tc = 3.0
    M0 = 0.1
    lambda_param = 15.0
    eps = 1e-15

    orbital = np.random.rand(SearchAgents_no)
    T = np.abs(np.random.randn(SearchAgents_no))

    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = np.random.rand(SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    PL_Fit = np.zeros(SearchAgents_no)
    Sun_Score = np.inf
    Sun_Pos = np.zeros(dim)

    for i in range(SearchAgents_no):
        PL_Fit[i] = _eval(Positions[i, :])
        if PL_Fit[i] < Sun_Score:
            Sun_Score = PL_Fit[i]
            Sun_Pos = Positions[i, :].copy()

    Convergence_curve = np.zeros(Tmax)
    t = 0

    while t < Tmax:
        sorted_fit = np.sort(PL_Fit)
        worstFitness = sorted_fit[-1]
        M = M0 * np.exp(-lambda_param * (t / Tmax))

        R = np.zeros(SearchAgents_no)
        for i in range(SearchAgents_no):
            R[i] = np.sqrt(np.sum((Sun_Pos - Positions[i, :]) ** 2))

        denom_sum = np.sum(PL_Fit - worstFitness) + eps
        MS = np.zeros(SearchAgents_no)
        m = np.zeros(SearchAgents_no)
        for i in range(SearchAgents_no):
            MS[i] = np.random.rand() * (Sun_Score - worstFitness) / denom_sum
            m[i] = (PL_Fit[i] - worstFitness) / denom_sum

        R_range = (np.max(R) - np.min(R)) + eps
        MS_range = (np.max(MS) - np.min(MS)) + eps
        m_range = (np.max(m) - np.min(m)) + eps

        Rnorm = (R - np.min(R)) / R_range
        MSnorm = (MS - np.min(MS)) / MS_range
        Mnorm = (m - np.min(m)) / m_range

        Fg = orbital * M * ((MSnorm * Mnorm) / (Rnorm * Rnorm + eps)) + np.random.rand(SearchAgents_no)

        a1 = np.zeros(SearchAgents_no)
        for i in range(SearchAgents_no):
            term = T[i] ** 2 * (M * (MS[i] + m[i]) / (4.0 * np.pi * np.pi))
            a1[i] = np.random.rand() * (term ** (1.0 / 3.0))

        for i in range(SearchAgents_no):
            cycle_denom = Tmax / Tc
            a2 = -1.0 + -1.0 * ((t % cycle_denom) / cycle_denom)
            n = (a2 - 1.0) * np.random.rand() + 1.0
            a_idx = np.random.randint(0, SearchAgents_no)
            b_idx = np.random.randint(0, SearchAgents_no)
            rd = np.random.rand(dim)
            r = np.random.rand()
            U1 = (rd < r).astype(float)
            O_P = Positions[i, :].copy()

            if np.random.rand() < np.random.rand():
                h = 1.0 / np.exp(n * np.random.randn())
                Xm = (Positions[b_idx, :] + Sun_Pos + Positions[i, :]) / 3.0
                Positions[i, :] = Positions[i, :] * U1 + (Xm + h * (Xm - Positions[a_idx, :])) * (1.0 - U1)
            else:
                f_dir = 1.0 if np.random.rand() < 0.5 else -1.0
                v_term = (2.0 / (R[i] + eps)) - (1.0 / (a1[i] + eps))
                L = (M * (MS[i] + m[i]) * abs(v_term)) ** 0.5
                U = (rd > np.random.rand(dim)).astype(float)

                if Rnorm[i] < 0.5:
                    M_rand = np.random.rand() * (1.0 - r) + r
                    l_val = L * M_rand * U
                    Mv = np.random.rand() * (1.0 - rd) + rd
                    l1_val = L * Mv * (1.0 - U)
                    V_i = (l_val * (2.0 * np.random.rand() * Positions[i, :] - Positions[a_idx, :])
                           + l1_val * (Positions[b_idx, :] - Positions[a_idx, :])
                           + (1.0 - Rnorm[i]) * f_dir * U1 * np.random.rand(dim) * (ub - lb))
                else:
                    U2 = float(np.random.rand() > np.random.rand())
                    V_i = (np.random.rand() * L * (Positions[a_idx, :] - Positions[i, :])
                           + (1.0 - Rnorm[i]) * f_dir * U2 * np.random.rand(dim) * (np.random.rand() * ub - lb))

                f_dir = 1.0 if np.random.rand() < 0.5 else -1.0
                Positions[i, :] = ((Positions[i, :] + V_i * f_dir)
                                   + (Fg[i] + abs(np.random.randn())) * U * (Sun_Pos - Positions[i, :]))

            if np.random.rand() < np.random.rand():
                out_bounds = (Positions[i, :] > ub) | (Positions[i, :] < lb)
                if np.any(out_bounds):
                    rand_reset = lb + np.random.rand(dim) * (ub - lb)
                    Positions[i, :] = np.where(out_bounds, rand_reset, Positions[i, :])
            else:
                Positions[i, :] = np.clip(Positions[i, :], lb, ub)

            PL_Fit1 = _eval(Positions[i, :])
            if PL_Fit1 < PL_Fit[i]:
                PL_Fit[i] = PL_Fit1
                if PL_Fit[i] < Sun_Score:
                    Sun_Score = PL_Fit[i]
                    Sun_Pos = Positions[i, :].copy()
            else:
                Positions[i, :] = O_P

            Convergence_curve[t] = Sun_Score
            t += 1
            if t >= Tmax:
                break

    return Sun_Score, Sun_Pos, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def koa_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        return koa(SearchAgents_no, Max_iter, ub, lb, dim, fobj)

    run_experiment(
        algo_name="KOA",
        algo_func=koa_adapter,
        full_name="Kepler Optimization Algorithm (KOA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

