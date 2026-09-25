"""
Enhanced Osprey Optimization Algorithm (EOOA)
Source: Enhanced Osprey Optimization Algorithm for Global Optimization with
        Application to PEM Fuel Cell Parameter Identification
        Biomimetics, 2026. DOI: 10.3390/biomimetics11080545
Authors: Yacine Bouali, Basem Alamri
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


def eooa(SearchAgents, Max_iterations, lb, ub, dim, ObjFun):
    """
    Enhanced Osprey Optimization Algorithm (EOOA)

    Parameters:
        SearchAgents   : Number of search agents
        Max_iterations : Maximum number of iterations
        lb             : Lower bound(s)
        ub             : Upper bound(s)
        dim            : Problem dimension
        ObjFun         : Objective function handle f(x) -> float

    Returns:
        Best_score : Minimum objective value found
        Best_pos   : Optimal position vector
        EOOA_curve : Best fitness convergence history
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    EOOA_curve = np.zeros(Max_iterations)

    # DE Control Parameters
    lambda_param = 3.0
    CR = 0.9
    F_max = 0.9
    F_min = 0.3

    X = initialization(SearchAgents, dim, ub_arr, lb_arr)
    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        fit[i] = ObjFun(X[i, :])

    best_idx = np.argmin(fit)
    fbest = float(fit[best_idx])
    xbest = X[best_idx, :].copy()

    for t in range(1, Max_iterations + 1):
        # Non-linear adaptive parameter
        alpha_t = np.exp(-lambda_param * ((t / Max_iterations) ** 2))
        F_de = F_min + (F_max - F_min) * alpha_t

        # Update global best
        best_idx = np.argmin(fit)
        if fit[best_idx] < fbest:
            fbest = float(fit[best_idx])
            xbest = X[best_idx, :].copy()

        for i in range(SearchAgents):
            # Phase 1: Position identification & hunting the fish (Exploration)
            fish_positions = np.where(fit < fit[i])[0]
            if len(fish_positions) == 0:
                selected_fish = xbest
            else:
                if np.random.rand() < 0.5:
                    selected_fish = xbest
                else:
                    k = np.random.choice(fish_positions)
                    selected_fish = X[k, :]

            I = round(1 + np.random.rand())
            X_new = X[i, :] + np.random.rand() * (selected_fish - I * X[i, :])
            X_new = np.clip(X_new, lb_arr, ub_arr)
            fit_new = ObjFun(X_new)
            if fit_new < fit[i]:
                X[i, :] = X_new
                fit[i] = fit_new

            # Phase 2: Carrying the fish to the suitable position (Exploitation)
            X_new = X[i, :] + (lb_arr + np.random.rand() * (ub_arr - lb_arr)) / t
            X_new = np.clip(X_new, lb_arr, ub_arr)
            fit_new = ObjFun(X_new)
            if fit_new < fit[i]:
                X[i, :] = X_new
                fit[i] = fit_new

            # Phase 3: DE-based mutation and crossover
            candidates = [idx for idx in range(SearchAgents) if idx != i]
            if len(candidates) >= 3:
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = X[r1, :] + F_de * (X[r2, :] - X[r3, :])
                mutant = np.clip(mutant, lb_arr, ub_arr)

                mask = np.random.rand(dim) < CR
                if not np.any(mask):
                    mask[np.random.randint(0, dim)] = True

                trial = X[i, :].copy()
                trial[mask] = mutant[mask]

                fit_trial = ObjFun(trial)
                if fit_trial < fit[i]:
                    X[i, :] = trial
                    fit[i] = fit_trial

        best_idx = np.argmin(fit)
        if fit[best_idx] < fbest:
            fbest = float(fit[best_idx])
            xbest = X[best_idx, :].copy()

        EOOA_curve[t - 1] = fbest

    Best_score = fbest
    Best_pos = xbest
    return Best_score, Best_pos, EOOA_curve


EOOA = eooa


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="EOOA",
        algo_func=eooa,
        full_name="Enhanced Osprey Optimization Algorithm (EOOA)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
