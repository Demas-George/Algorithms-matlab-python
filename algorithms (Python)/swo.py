"""
Spider Wasp Optimizer (SWO)
Source: Spider Wasp Optimizer: A Novel Meta-Heuristic Optimization Algorithm
        Artificial Intelligence Review, 2023.
Authors: Mohamed Abdel-Basset, Reda Mohamed
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
    return 0.05 * (u / denom)


def swo(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    Spider Wasp Optimizer (SWO)

    Parameters:
        SearchAgents_no : Initial number of spider wasps
        Max_iter        : Maximum number of iterations
        lb              : Lower bounds (scalar or array)
        ub              : Upper bounds (scalar or array)
        dim             : Dimensionality
        fobj            : Objective function handle f(x) -> float

    Returns:
        Best_score        : Minimum objective value found
        Best_SW           : Best position vector found
        Convergence_curve : History of best fitness per iteration
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    TR = 0.3
    Cr = 0.2
    N_min = min(20, SearchAgents_no)
    N_orig = SearchAgents_no

    Positions = initialization(SearchAgents_no, dim, ub_arr, lb_arr)
    SW_Fit = np.zeros(SearchAgents_no)
    Best_score = np.inf
    Best_SW = np.zeros(dim)

    for i in range(SearchAgents_no):
        SW_Fit[i] = fobj(Positions[i, :])
        if SW_Fit[i] < Best_score:
            Best_score = SW_Fit[i]
            Best_SW = Positions[i, :].copy()

    Convergence_curve = np.zeros(Max_iter)
    current_N = SearchAgents_no

    for iteration in range(1, Max_iter + 1):
        a = 2.0 - 2.0 * (iteration / Max_iter)
        a2 = -1.0 - 1.0 * (iteration / Max_iter)
        k = 1.0 - iteration / Max_iter

        JK = np.random.permutation(current_N)

        if np.random.rand() < TR:
            # ── 3.2 Hunting and nesting behavior ─────────────────────
            for i in range(current_N):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                p  = np.random.rand()
                C = a * (2.0 * r1 - 1.0)
                l = (a2 - 1.0) * np.random.rand() + 1.0
                L = _levy(1)[0]
                vc = np.random.uniform(-k, k, dim)
                rn1 = np.random.randn()
                O_P = Positions[i, :].copy()

                for j in range(dim):
                    if i < k * current_N:
                        if p < (1.0 - iteration / Max_iter):
                            if r1 < r2:
                                m1 = abs(rn1) * r1
                                Positions[i, j] += m1 * (Positions[JK[0], j] - Positions[JK[1], j])
                            else:
                                B = 1.0 / (1.0 + np.exp(l))
                                m2 = B * np.cos(l * 2.0 * np.pi)
                                Positions[i, j] = Positions[JK[i % current_N], j] + m2 * (lb_arr[j] + np.random.rand() * (ub_arr[j] - lb_arr[j]))
                        else:
                            if r1 < r2:
                                Positions[i, j] += C * abs(2.0 * np.random.rand() * Positions[JK[2 % current_N], j] - Positions[i, j])
                            else:
                                Positions[i, j] = Positions[i, j] * vc[j]
                    else:
                        if r1 < r2:
                            Positions[i, j] = Best_SW[j] + np.cos(2.0 * l * np.pi) * (Best_SW[j] - Positions[i, j])
                        else:
                            Positions[i, j] = (
                                Positions[JK[0], j]
                                + r3 * abs(L) * (Positions[JK[0], j] - Positions[i, j])
                                + (1.0 - r3) * float(np.random.rand() > np.random.rand()) * (Positions[JK[2 % current_N], j] - Positions[JK[1], j])
                            )

                # Return search agents exceeding bounds
                for j in range(dim):
                    if Positions[i, j] > ub_arr[j] or Positions[i, j] < lb_arr[j]:
                        Positions[i, j] = lb_arr[j] + np.random.rand() * (ub_arr[j] - lb_arr[j])

                SW_Fit1 = fobj(Positions[i, :])
                if SW_Fit1 < SW_Fit[i]:
                    SW_Fit[i] = SW_Fit1
                    if SW_Fit[i] < Best_score:
                        Best_score = SW_Fit[i]
                        Best_SW = Positions[i, :].copy()
                else:
                    Positions[i, :] = O_P
        else:
            # ── 3.3 Mating behavior ──────────────────────────────────
            for i in range(current_N):
                l = (a2 - 1.0) * np.random.rand() + 1.0
                SW_m = np.zeros(dim)
                O_P = Positions[i, :].copy()

                idx1 = JK[0]
                idx2 = JK[1 % current_N]
                idx3 = JK[2 % current_N]

                v1 = (Positions[idx1, :] - Positions[i, :]) if SW_Fit[idx1] < SW_Fit[i] else (Positions[i, :] - Positions[idx1, :])
                v2 = (Positions[idx2, :] - Positions[idx3, :]) if SW_Fit[idx2] < SW_Fit[idx3] else (Positions[idx3, :] - Positions[idx2, :])

                rn1 = np.random.randn()
                rn2 = np.random.randn()

                for j in range(dim):
                    SW_m[j] = Positions[i, j] + np.exp(l) * abs(rn1) * v1[j] + (1.0 - np.exp(l)) * abs(rn2) * v2[j]
                    if np.random.rand() < Cr:
                        Positions[i, j] = SW_m[j]

                for j in range(dim):
                    if Positions[i, j] > ub_arr[j] or Positions[i, j] < lb_arr[j]:
                        Positions[i, j] = lb_arr[j] + np.random.rand() * (ub_arr[j] - lb_arr[j])

                SW_Fit1 = fobj(Positions[i, :])
                if SW_Fit1 < SW_Fit[i]:
                    SW_Fit[i] = SW_Fit1
                    if SW_Fit[i] < Best_score:
                        Best_score = SW_Fit[i]
                        Best_SW = Positions[i, :].copy()
                else:
                    Positions[i, :] = O_P

        # Population reduction
        new_N = int(math.floor(N_min + (N_orig - N_min) * ((Max_iter - iteration) / Max_iter)))
        new_N = max(N_min, min(new_N, current_N))
        if new_N < current_N:
            sort_idx = np.argsort(SW_Fit[:current_N])
            Positions = Positions[sort_idx[:new_N], :]
            SW_Fit = SW_Fit[sort_idx[:new_N]]
            current_N = new_N

        Convergence_curve[iteration - 1] = Best_score

    return Best_score, Best_SW, Convergence_curve


SWO = swo


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="SWO",
        algo_func=swo,
        full_name="Spider Wasp Optimizer (SWO)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
