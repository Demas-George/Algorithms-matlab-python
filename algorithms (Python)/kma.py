"""
Komodo Mlipir Algorithm (KMA)
Source: Komodo Mlipir Algorithm
        Applied Soft Computing, 2021, 108043. DOI: 10.1016/j.asoc.2021.108043
Authors: S. Suyanto, A. A. Ariyanto, A. F. Ariyanto
"""

import math
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


def _levy(n, m, beta=1.5):
    num = math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = np.random.normal(0.0, sigma_u, (n, m))
    v = np.random.normal(0.0, 1.0, (n, m))
    denom = np.abs(v) ** (1.0 / beta)
    denom[denom == 0] = 1e-16
    return u / denom


def kma(PopSize, Max_iter, lb, ub, dim, fobj):
    """
    Komodo Mlipir Algorithm (KMA)

    Parameters:
        PopSize  : Population size (number of Komodo individuals)
        Max_iter : Maximum number of generations / iterations
        lb       : Lower bound(s)
        ub       : Upper bound(s)
        dim      : Dimension (Nvar)
        fobj     : Objective function handle f(x) -> float

    Returns:
        OptVal            : Best fitness value found
        BestIndiv         : Best decision vector found
        convergence_curve : Convergence history across iterations
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Pop = initialization(PopSize, dim, ub_arr, lb_arr)
    FX = np.zeros(PopSize)
    for i in range(PopSize):
        FX[i] = fobj(Pop[i, :])

    ind_sort = np.argsort(FX)
    FX = FX[ind_sort]
    Pop = Pop[ind_sort, :]

    OptVal = float(FX[0])
    BestIndiv = Pop[0, :].copy()
    convergence_curve = np.zeros(Max_iter)

    NumBM = max(1, PopSize // 2)
    MutRate = 0.5
    MutRadius = 0.5

    for gen in range(1, Max_iter + 1):
        # Mlipir rate transitions from exploration to exploitation
        if gen <= Max_iter // 2:
            MlipirRate = (dim - 1.0) / dim if dim > 1 else 0.5
        else:
            MlipirRate = 0.5

        # Split population: Big Males, Female, Small Males
        BigMales = Pop[0:NumBM, :].copy()
        BigMalesFX = FX[0:NumBM].copy()

        female_idx = NumBM
        Female = Pop[female_idx, :].copy()
        FemaleFX = FX[female_idx]

        SmallMales = Pop[NumBM + 1:, :].copy()
        SmallMalesFX = FX[NumBM + 1:].copy()

        # ── 1. Move Big Males ────────────────────────────────────
        TempBM = BigMales.copy()
        TempBMFX = BigMalesFX.copy()
        n_bm = TempBM.shape[0]

        for ss in range(n_bm):
            MaxFolHQ = np.random.randint(1, 3)
            VM = np.zeros(dim)
            RHQ = np.random.permutation(n_bm)
            fol_count = 0
            for fs in range(len(RHQ)):
                ind = RHQ[fs]
                if ind != ss:
                    if TempBMFX[ind] < TempBMFX[ss] or np.random.rand() < 0.5:
                        VM += np.random.rand() * (BigMales[ind, :] - TempBM[ss, :])
                    else:
                        VM += np.random.rand() * (TempBM[ss, :] - BigMales[ind, :])
                    fol_count += 1
                    if fol_count >= MaxFolHQ:
                        break

            NewBM = np.clip(TempBM[ss, :] + VM, lb_arr, ub_arr)
            new_fx = fobj(NewBM)
            if new_fx < TempBMFX[ss]:
                TempBM[ss, :] = NewBM
                TempBMFX[ss] = new_fx

        # Replacement for Big Males
        joint_BM = np.vstack([BigMales, TempBM])
        joint_BMFX = np.concatenate([BigMalesFX, TempBMFX])
        bm_sort = np.argsort(joint_BMFX)
        BigMales = joint_BM[bm_sort[:NumBM], :]
        BigMalesFX = joint_BMFX[bm_sort[:NumBM]]

        WinnerBM = BigMales[0, :]
        WinnerFX = BigMalesFX[0]

        # ── 2. Female Reproduction ───────────────────────────────
        if WinnerFX < FemaleFX or np.random.rand() < 0.5:
            # Sexual reproduction (whole arithmetic crossover)
            rval = np.random.rand(dim)
            off1 = np.clip(rval * WinnerBM + (1.0 - rval) * Female, lb_arr, ub_arr)
            off2 = np.clip(rval * Female + (1.0 - rval) * WinnerBM, lb_arr, ub_arr)
            fx1 = fobj(off1)
            fx2 = fobj(off2)
            if fx1 < fx2:
                if fx1 < FemaleFX:
                    Female = off1
                    FemaleFX = fx1
            else:
                if fx2 < FemaleFX:
                    Female = off2
                    FemaleFX = fx2
        else:
            # Asexual reproduction (mutation)
            NewFemale = Female.copy()
            MaxStep = MutRadius * (ub_arr - lb_arr)
            for ii in range(dim):
                if np.random.rand() < MutRate:
                    NewFemale[ii] += (2.0 * np.random.rand() - 1.0) * MaxStep[ii]
            NewFemale = np.clip(NewFemale, lb_arr, ub_arr)
            fx = fobj(NewFemale)
            if fx < FemaleFX:
                Female = NewFemale
                FemaleFX = fx

        # ── 3. Move (Mlipir) Small Males ─────────────────────────
        n_sm = SmallMales.shape[0]
        if n_sm > 0:
            for ww in range(n_sm):
                VMlipir = np.zeros(dim)
                RHQ = np.random.permutation(n_bm)
                ind = RHQ[0]
                A = np.random.permutation(dim)
                D = int(round(MlipirRate * dim))
                D = max(1, min(D, dim - 1 if dim > 1 else 1))
                M = A[:D]
                B = np.zeros(dim)
                B[M] = 1.0

                VMlipir = np.random.rand(dim) * (BigMales[ind, :] * B) - (SmallMales[ww, :] * B)
                NewSM = np.clip(SmallMales[ww, :] + VMlipir, lb_arr, ub_arr)
                fx_sm = fobj(NewSM)
                if fx_sm < SmallMalesFX[ww]:
                    SmallMales[ww, :] = NewSM
                    SmallMalesFX[ww] = fx_sm

        # Combine population
        if n_sm > 0:
            Pop = np.vstack([BigMales, Female.reshape(1, -1), SmallMales])
            FX = np.concatenate([BigMalesFX, [FemaleFX], SmallMalesFX])
        else:
            Pop = np.vstack([BigMales, Female.reshape(1, -1)])
            FX = np.concatenate([BigMalesFX, [FemaleFX]])

        ind_sort = np.argsort(FX)
        FX = FX[ind_sort]
        Pop = Pop[ind_sort, :]

        if FX[0] < OptVal:
            OptVal = float(FX[0])
            BestIndiv = Pop[0, :].copy()

        convergence_curve[gen - 1] = OptVal

    return OptVal, BestIndiv, convergence_curve


KMA = kma


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="KMA",
        algo_func=kma,
        full_name="Komodo Mlipir Algorithm (KMA)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
