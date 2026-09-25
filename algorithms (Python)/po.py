"""
Puma Optimizer (PO)

Derived from:
    Sources(MATLAB)/Puma-Optimizer (PO)/Puma-Optimizer-PO--main/Puma.m
    Authors: Benyamin Abdollahzadeh, Nima Khodadadi, Saeid Barshandeh, Pavel Trojovsky,
             Farhad Soleimanian Gharehchopogh, El-Sayed M. El-kenawy, Laith Abualigah,
             Seyedali Mirjalili (2024)
    Paper: "Puma optimizer (PO): a novel metaheuristic optimization algorithm and its
           application in machine learning"
           Cluster Computing. DOI: 10.1007/s10586-023-04221-5

Original MATLAB Function Signature:
    [Puma_X, Puma_C, Convergence] = Puma(nSol, MaxIter, lb, ub, dim, CostFunction)

Python Entry Point:
    puma(nSol, MaxIter, lb, ub, dim, CostFunction)

Parameters:
    nSol         : int, number of candidate solutions (puma population size, >= 7 recommended)
    MaxIter      : int, maximum number of iterations
    lb           : float or numpy.ndarray of shape (dim,), lower boundaries
    ub           : float or numpy.ndarray of shape (dim,), upper boundaries
    dim          : int, problem dimension
    CostFunction : callable, objective function f(x) returning a scalar float

Returns:
    Puma_X       : numpy.ndarray of shape (dim,), optimal solution vector
    Puma_C       : float, best cost (fitness) found
    Convergence  : numpy.ndarray of shape (MaxIter,), convergence curve
"""

import copy
import numpy as np


class _Sol:
    __slots__ = ('X', 'Cost')

    def __init__(self, X, Cost):
        self.X = X
        self.Cost = Cost


def _exploration(sol_list, lb, ub, dim, nSol, cost_func):
    Sol = copy.deepcopy(sol_list)
    Sol.sort(key=lambda s: s.Cost)
    pCR = 0.20
    PCR = 1.0 - pCR
    p = PCR / nSol

    for i in range(nSol):
        x = Sol[i].X.copy()
        candidates = [k for k in range(nSol) if k != i]
        if len(candidates) < 6:
            # Wrap-around sampling if nSol is small
            a, b, c, d, e, f = np.random.choice(candidates, size=6, replace=True)
        else:
            perm = np.random.choice(candidates, size=6, replace=False)
            a, b, c, d, e, f = perm[:6]

        G = 2.0 * np.random.rand() - 1.0
        if np.random.rand() < 0.5:
            y = np.random.rand(dim) * (ub - lb) + lb
        else:
            diff_ab = Sol[a].X - Sol[b].X
            diff_cd = Sol[c].X - Sol[d].X
            diff_ef = Sol[e].X - Sol[f].X
            y = Sol[a].X + G * diff_ab + G * ((diff_ab - diff_cd) + (diff_cd - diff_ef))

        y = np.clip(y, lb, ub)
        z = np.zeros(dim)
        j0 = np.random.randint(0, dim)
        for j in range(dim):
            if j == j0 or np.random.rand() <= pCR:
                z[j] = y[j]
            else:
                z[j] = x[j]

        new_cost = cost_func(z)
        if new_cost < Sol[i].Cost:
            Sol[i].X = z
            Sol[i].Cost = new_cost
        else:
            pCR += p

    return Sol


def _exploitation(sol_list, lb, ub, dim, nSol, Best, MaxIter, Iter, cost_func):
    Sol = copy.deepcopy(sol_list)
    Q = 0.67
    Beta = 2.0

    # mbest = mean([Sol.X])/nSol matching MATLAB: mean of all scalar coordinates across all individuals
    all_coords = np.concatenate([s.X for s in Sol])
    mbest = np.mean(all_coords) / nSol

    for i in range(nSol):
        beta1 = 2.0 * np.random.rand()
        beta2 = np.random.randn(dim)
        w = np.random.randn(dim)
        v = np.random.randn(dim)
        F1 = np.random.randn(dim) * np.exp(2.0 - Iter * (2.0 / MaxIter))
        F2 = w * (v ** 2) * np.cos((2.0 * np.random.rand()) * w)

        R_1 = 2.0 * np.random.rand() - 1.0
        S1 = (2.0 * np.random.rand() - 1.0) + np.random.randn(dim)
        S2 = F1 * R_1 * Sol[i].X + F2 * (1.0 - R_1) * Best.X

        # MATLAB S2 / S1 (mrdivide of row vectors) produces scalar dot(S2, S1)/dot(S1, S1)
        denom = np.dot(S1, S1)
        VEC = np.dot(S2, S1) / denom if denom != 0 else 0.0

        if np.random.rand() <= 0.5:
            Xatack = VEC
            if np.random.rand() > Q:
                rand_idx = np.random.randint(0, nSol)
                NewX = Best.X + beta1 * np.exp(beta2) * (Sol[rand_idx].X - Sol[i].X)
            else:
                NewX = beta1 * Xatack - Best.X
        else:
            r1 = int(np.round(1.0 + (nSol - 1) * np.random.rand())) - 1
            r1 = max(0, min(r1, nSol - 1))
            sign_val = (-1.0) ** np.random.randint(0, 2)
            NewX = (mbest * Sol[r1].X - sign_val * Sol[i].X) / (1.0 + Beta * np.random.rand())

        NewX = np.clip(NewX, lb, ub)
        new_cost = cost_func(NewX)

        if new_cost < Sol[i].Cost:
            Sol[i].X = NewX
            Sol[i].Cost = new_cost

    return Sol


def puma(nSol, MaxIter, lb, ub, dim, CostFunction):
    """
    Execute the Puma Optimizer algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    UnSelected = np.ones(2)  # 0: Exploration, 1: Exploitation
    F3_Explore = 0.0
    F3_Exploit = 0.0
    Seq_Time_Explore = np.ones(3)
    Seq_Time_Exploit = np.ones(3)
    Seq_Cost_Explore = np.ones(3)
    Seq_Cost_Exploit = np.ones(3)
    PF = [0.5, 0.5, 0.3]
    PF_F3 = []
    Mega_Explor = 0.99
    Mega_Exploit = 0.99
    Convergence = np.zeros(MaxIter)

    # Population Initialization
    Sol = []
    for _ in range(nSol):
        x = lb + np.random.rand(dim) * (ub - lb)
        Sol.append(_Sol(x, CostFunction(x)))

    Sol.sort(key=lambda s: s.Cost)
    Best = copy.deepcopy(Sol[0])
    Initial_Best = copy.deepcopy(Best)

    Costs_Explor = np.zeros(3)
    Costs_Exploit = np.zeros(3)

    # Unexperienced Phase (Iter 1 to 3)
    for Iter in range(1, 4):
        Sol_Explor = _exploration(Sol, lb, ub, dim, nSol, CostFunction)
        Costs_Explor[Iter - 1] = min(s.Cost for s in Sol_Explor)

        Sol_Exploit = _exploitation(Sol, lb, ub, dim, nSol, Best, MaxIter, Iter, CostFunction)
        Costs_Exploit[Iter - 1] = min(s.Cost for s in Sol_Exploit)

        Sol = Sol + Sol_Explor + Sol_Exploit
        Sol.sort(key=lambda s: s.Cost)
        Sol = Sol[:nSol]
        Best = copy.deepcopy(Sol[0])
        Convergence[Iter - 1] = Best.Cost

    Seq_Cost_Explore[0] = abs(Initial_Best.Cost - Costs_Explor[0])
    Seq_Cost_Exploit[0] = abs(Initial_Best.Cost - Costs_Exploit[0])
    Seq_Cost_Explore[1] = abs(Costs_Explor[1] - Costs_Explor[0])
    Seq_Cost_Exploit[1] = abs(Costs_Exploit[1] - Costs_Exploit[0])
    Seq_Cost_Explore[2] = abs(Costs_Explor[2] - Costs_Explor[1])
    Seq_Cost_Exploit[2] = abs(Costs_Exploit[2] - Costs_Exploit[1])

    for i in range(3):
        if Seq_Cost_Explore[i] != 0:
            PF_F3.append(Seq_Cost_Explore[i])
        if Seq_Cost_Exploit[i] != 0:
            PF_F3.append(Seq_Cost_Exploit[i])

    F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
    F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
    F2_Explor = PF[1] * (np.sum(Seq_Cost_Explore) / np.sum(Seq_Time_Explore))
    F2_Exploit = PF[1] * (np.sum(Seq_Cost_Exploit) / np.sum(Seq_Time_Exploit))

    Score_Explore = (PF[0] * F1_Explor) + (PF[1] * F2_Explor)
    Score_Exploit = (PF[0] * F1_Exploit) + (PF[1] * F2_Exploit)

    # Experienced Phase (Iter 4 to MaxIter)
    for Iter in range(4, MaxIter + 1):
        if Score_Explore > Score_Exploit:
            SelectFlag = 1
            Sol = _exploration(Sol, lb, ub, dim, nSol, CostFunction)
            Count_select = UnSelected.copy()
            UnSelected[1] += 1
            UnSelected[0] = 1
            F3_Explore = PF[2]
            F3_Exploit += PF[2]

            min_idx = np.argmin([s.Cost for s in Sol])
            TBest = Sol[min_idx]
            Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
            Seq_Cost_Explore[1] = Seq_Cost_Explore[0]
            Seq_Cost_Explore[0] = abs(Best.Cost - TBest.Cost)
            if Seq_Cost_Explore[0] != 0:
                PF_F3.append(Seq_Cost_Explore[0])
            if TBest.Cost < Best.Cost:
                Best = copy.deepcopy(TBest)
        else:
            SelectFlag = 2
            Sol = _exploitation(Sol, lb, ub, dim, nSol, Best, MaxIter, Iter, CostFunction)
            Count_select = UnSelected.copy()
            UnSelected[0] += 1
            UnSelected[1] = 1
            F3_Exploit = PF[2]
            F3_Explore += PF[2]

            min_idx = np.argmin([s.Cost for s in Sol])
            TBest = Sol[min_idx]
            Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
            Seq_Cost_Exploit[1] = Seq_Cost_Exploit[0]
            Seq_Cost_Exploit[0] = abs(Best.Cost - TBest.Cost)
            if Seq_Cost_Exploit[0] != 0:
                PF_F3.append(Seq_Cost_Exploit[0])
            if TBest.Cost < Best.Cost:
                Best = copy.deepcopy(TBest)

        if SelectFlag == 1:
            Seq_Time_Explore[2] = Seq_Time_Explore[1]
            Seq_Time_Explore[1] = Seq_Time_Explore[0]
            Seq_Time_Explore[0] = Count_select[0]
        else:
            Seq_Time_Exploit[2] = Seq_Time_Exploit[1]
            Seq_Time_Exploit[1] = Seq_Time_Exploit[0]
            Seq_Time_Exploit[0] = Count_select[1]

        F1_Explor = PF[0] * (Seq_Cost_Explore[0] / Seq_Time_Explore[0])
        F1_Exploit = PF[0] * (Seq_Cost_Exploit[0] / Seq_Time_Exploit[0])
        F2_Explor = PF[1] * (np.sum(Seq_Cost_Explore) / np.sum(Seq_Time_Explore))
        F2_Exploit = PF[1] * (np.sum(Seq_Cost_Exploit) / np.sum(Seq_Time_Exploit))

        if Score_Explore < Score_Exploit:
            Mega_Explor = max(Mega_Explor - 0.01, 0.01)
            Mega_Exploit = 0.99
        elif Score_Explore > Score_Exploit:
            Mega_Explor = 0.99
            Mega_Exploit = max(Mega_Exploit - 0.01, 0.01)

        lmn_Explore = 1.0 - Mega_Explor
        lmn_Exploit = 1.0 - Mega_Exploit

        min_pf3 = min(PF_F3) if len(PF_F3) > 0 else 0.0
        Score_Explore = (Mega_Explor * F1_Explor) + (Mega_Explor * F2_Explor) + (lmn_Explore * (min_pf3 * F3_Explore))
        Score_Exploit = (Mega_Exploit * F1_Exploit) + (Mega_Exploit * F2_Exploit) + (lmn_Exploit * (min_pf3 * F3_Exploit))

        Convergence[Iter - 1] = Best.Cost

    return Best.X.copy(), Best.Cost, Convergence


po = puma


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def po_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        pos, score, curve = puma(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
        return score, pos, curve

    run_experiment(
        algo_name="PO",
        algo_func=po_adapter,
        full_name="Puma Optimizer (PO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

