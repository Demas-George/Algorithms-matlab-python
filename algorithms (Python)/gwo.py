"""
Grey Wolf Optimizer (GWO)
Applied to: Triple Diode Model (TDM) Parameter Identification
Dataset: RTC France Silicon Solar Cell (26 I-V points)
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
#  PART 1 - GWO Algorithm
# ═══════════════════════════════════════════════════════════════

def initialization(SearchAgents_no, dim, ub, lb):
    ub = np.array(ub)
    lb = np.array(lb)
    if ub.size == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    return Positions


def gwo(SearchAgents_no, Max_iter, lb, ub, dim, fobj):

    Alpha_pos   = np.zeros(dim)
    Alpha_score = np.inf

    Beta_pos   = np.zeros(dim)
    Beta_score = np.inf

    Delta_pos   = np.zeros(dim)
    Delta_score = np.inf

    Positions = initialization(SearchAgents_no, dim, ub, lb)
    Convergence_curve = np.zeros(Max_iter)

    l = 0

    while l < Max_iter:

        for i in range(Positions.shape[0]):

            # Boundary check
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))
                                + ub * Flag4ub
                                + lb * Flag4lb)

            fitness = fobj(Positions[i, :])

            if fitness < Alpha_score:
                Alpha_score = fitness
                Alpha_pos   = Positions[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness
                Beta_pos   = Positions[i, :].copy()

            if fitness > Alpha_score and fitness > Beta_score and fitness < Delta_score:
                Delta_score = fitness
                Delta_pos   = Positions[i, :].copy()

        # a decreases linearly from 2 to 0
        a = 2 - l * (2 / Max_iter)

        for i in range(Positions.shape[0]):
            for j in range(Positions.shape[1]):

                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3   # Eq.(3.7)

        l += 1
        Convergence_curve[l - 1] = Alpha_score

        if l % 100 == 0:
            print(f'    Iteration {l:4d}: Best RMSE = {Alpha_score:.10f}')

    return Alpha_score, Alpha_pos, Convergence_curve


GWO = gwo


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="GWO",
        algo_func=gwo,
        full_name="Grey Wolf Optimizer (GWO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )
