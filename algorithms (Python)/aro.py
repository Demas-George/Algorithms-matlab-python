"""
Artificial Rabbits Optimization (ARO)

Derived from:
    Sources(MATLAB)/ARO/ARO/ARO_for_Functions/ARO.m
    Authors: Liying Wang, Qingke Zhang (2022)
    Paper: "Artificial rabbits optimization: A new bio-inspired meta-heuristic
           algorithm for solving engineering problems"
           Engineering Applications of Artificial Intelligence, 114, 105082.

Original MATLAB Function Signature:
    [BestX, BestF, HisBestF] = ARO(F_index, MaxIt, nPop)

Python Entry Point:
    aro(nPop, MaxIt, Low, Up, Dim, fobj)

Parameters:
    nPop    : int, population size (number of rabbits)
    MaxIt   : int, maximum number of iterations
    Low     : float or numpy.ndarray of shape (Dim,), lower search boundaries
    Up      : float or numpy.ndarray of shape (Dim,), upper search boundaries
    Dim     : int, dimensionality of the problem
    fobj    : callable, objective function to minimize; takes 1D array of shape (Dim,), returns float

Returns:
    BestX    : numpy.ndarray of shape (Dim,), best solution vector found
    BestF    : float, best fitness score corresponding to BestX
    HisBestF : numpy.ndarray of shape (MaxIt,), convergence history of best fitness
"""

import numpy as np


def _space_bound(X, Up, Low):
    """
    Boundary handling matching MATLAB SpaceBound.m:
    Randomly reinitializes dimensions that exceed upper or lower bounds.
    """
    Dim = len(X)
    S = (X > Up) | (X < Low)
    rand_pos = np.random.rand(Dim) * (Up - Low) + Low
    return np.where(S, rand_pos, X)


def aro(nPop, MaxIt, Low, Up, Dim, fobj):
    """
    Execute the Artificial Rabbits Optimization algorithm.
    """
    Low = np.full(Dim, Low, dtype=float) if np.isscalar(Low) else np.asarray(Low, dtype=float)
    Up = np.full(Dim, Up, dtype=float) if np.isscalar(Up) else np.asarray(Up, dtype=float)

    PopPos = np.zeros((nPop, Dim))
    PopFit = np.zeros(nPop)

    for i in range(nPop):
        PopPos[i, :] = np.random.rand(Dim) * (Up - Low) + Low
        PopFit[i] = fobj(PopPos[i, :])

    BestF = np.inf
    BestX = None

    for i in range(nPop):
        if PopFit[i] <= BestF:
            BestF = PopFit[i]
            BestX = PopPos[i, :].copy()

    HisBestF = np.zeros(MaxIt)

    for It in range(1, MaxIt + 1):
        Direct1 = np.zeros((nPop, Dim))
        Direct2 = np.zeros((nPop, Dim))
        theta = 2.0 * (1.0 - It / MaxIt)

        for i in range(nPop):
            L = (np.exp(1.0) - np.exp(((It - 1.0) / MaxIt) ** 2)) * np.sin(2.0 * np.pi * np.random.rand())  # Eq.(3)
            rd = int(np.ceil(np.random.rand() * Dim))
            rand_perm = np.random.choice(Dim, rd, replace=False)
            Direct1[i, rand_perm] = 1.0
            c = Direct1[i, :]  # Eq.(4)
            R = L * c  # Eq.(2)

            A = 2.0 * np.log(1.0 / np.random.rand()) * theta  # Eq.(15)

            if A > 1.0:
                K = [k for k in range(nPop) if k != i]
                RandInd = K[np.random.randint(0, len(K))]
                newPopPos = (PopPos[RandInd, :]
                             + R * (PopPos[i, :] - PopPos[RandInd, :])
                             + np.round(0.5 * (0.05 + np.random.rand())) * np.random.randn())  # Eq.(1)
            else:
                rand_dim_idx = int(np.ceil(np.random.rand() * Dim)) - 1
                Direct2[i, rand_dim_idx] = 1.0
                gr = Direct2[i, :]  # Eq.(12)
                H = ((MaxIt - It + 1.0) / MaxIt) * np.random.randn()  # Eq.(8)
                b = PopPos[i, :] + H * gr * PopPos[i, :]  # Eq.(13)
                newPopPos = PopPos[i, :] + R * (np.random.rand() * b - PopPos[i, :])  # Eq.(11)

            newPopPos = _space_bound(newPopPos, Up, Low)
            newPopFit = fobj(newPopPos)

            if newPopFit < PopFit[i]:
                PopFit[i] = newPopFit
                PopPos[i, :] = newPopPos

        for i in range(nPop):
            if PopFit[i] < BestF:
                BestF = PopFit[i]
                BestX = PopPos[i, :].copy()

        HisBestF[It - 1] = BestF

    return BestX, BestF, HisBestF


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def aro_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        pos, score, curve = aro(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
        return score, pos, curve

    run_experiment(
        algo_name="ARO",
        algo_func=aro_adapter,
        full_name="Artificial Rabbits Optimization (ARO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

