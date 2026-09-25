"""
Gorilla Troops Optimizer (GTO)
Source: Artificial protozoa optimizer / Gorilla Troops Optimizer: A new nature-inspired
        metaheuristic algorithm for global optimization problems (Expert Systems with Applications, 2021)
Authors: Benyamin Abdollahzadeh, Farhad Soleimanian Gharehchopogh, Seyedali Mirjalili
"""

import numpy as np


def initialization(pop_size, dim, ub, lb):
    ub = np.array(ub, dtype=float)
    lb = np.array(lb, dtype=float)
    if ub.size == 1:
        Positions = np.random.rand(pop_size, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((pop_size, dim))
        for i in range(dim):
            Positions[:, i] = np.random.rand(pop_size) * (ub[i] - lb[i]) + lb[i]
    return Positions


def gto(pop_size, max_iter, lower_bound, upper_bound, variables_no, fobj):
    """
    Gorilla Troops Optimizer (GTO)

    Parameters:
        pop_size     : Population size of gorilla troop
        max_iter     : Maximum number of iterations
        lower_bound  : Lower bound(s)
        upper_bound  : Upper bound(s)
        variables_no : Number of decision variables (dimension)
        fobj         : Objective function handle f(x) -> float

    Returns:
        Silverback_Score  : Best fitness achieved
        Silverback        : Best position found
        convergence_curve : Convergence curve over iterations
    """
    lb = np.ones(variables_no) * lower_bound if np.isscalar(lower_bound) else np.asarray(lower_bound, dtype=float)
    ub = np.ones(variables_no) * upper_bound if np.isscalar(upper_bound) else np.asarray(upper_bound, dtype=float)

    Silverback = np.zeros(variables_no)
    Silverback_Score = np.inf

    X = initialization(pop_size, variables_no, ub, lb)
    Pop_Fit = np.zeros(pop_size)
    convergence_curve = np.zeros(max_iter)

    for i in range(pop_size):
        Pop_Fit[i] = fobj(X[i, :])
        if Pop_Fit[i] < Silverback_Score:
            Silverback_Score = Pop_Fit[i]
            Silverback = X[i, :].copy()

    GX = X.copy()
    p = 0.03
    Beta = 3.0
    w = 0.8

    for It in range(1, max_iter + 1):
        a = (np.cos(2.0 * np.random.rand()) + 1.0) * (1.0 - It / max_iter)
        C = a * (2.0 * np.random.rand() - 1.0)

        # ── Exploration ──────────────────────────────────────────
        for i in range(pop_size):
            if np.random.rand() < p:
                GX[i, :] = (ub - lb) * np.random.rand(variables_no) + lb
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(-a, a, variables_no)
                    H = Z * X[i, :]
                    rand_idx = np.random.randint(0, pop_size)
                    GX[i, :] = (np.random.rand() - a) * X[rand_idx, :] + C * H
                else:
                    rand_idx1 = np.random.randint(0, pop_size)
                    rand_idx2 = np.random.randint(0, pop_size)
                    GX[i, :] = X[i, :] - C * (
                        C * (X[i, :] - GX[rand_idx1, :]) + np.random.rand() * (X[i, :] - GX[rand_idx2, :])
                    )

        GX = np.clip(GX, lb, ub)

        # Group formation operation
        for i in range(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :].copy()
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :].copy()

        # ── Exploitation ─────────────────────────────────────────
        for i in range(pop_size):
            if a >= w:
                g = 2.0 ** C
                mean_gx = np.abs(np.mean(GX, axis=0))
                # Protect against overflow in power
                g_safe = np.clip(g, -100.0, 100.0)
                delta = (mean_gx ** g_safe) ** (1.0 / (g_safe if abs(g_safe) > 1e-12 else 1.0))
                GX[i, :] = C * delta * (X[i, :] - Silverback) + X[i, :]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.randn(variables_no)
                else:
                    h = np.random.randn()
                r1 = np.random.rand()
                GX[i, :] = Silverback - (
                    Silverback * (2.0 * r1 - 1.0) - X[i, :] * (2.0 * r1 - 1.0)
                ) * (Beta * h)

        GX = np.clip(GX, lb, ub)

        # Group formation operation
        for i in range(pop_size):
            New_Fit = fobj(GX[i, :])
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :].copy()
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :].copy()

        convergence_curve[It - 1] = Silverback_Score

    return Silverback_Score, Silverback, convergence_curve


GTO = gto


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="GTO",
        algo_func=gto,
        full_name="Gorilla Troops Optimizer (GTO)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
