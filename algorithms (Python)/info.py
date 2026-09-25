"""
Weighted Mean of Vectors (INFO)
Source: INFO: An Efficient Optimization Algorithm based on Weighted Mean of Vectors
        Expert Systems With Applications, 2022. DOI: 10.1016/j.eswa.2022.116516
Authors: Iman Ahmadianfar, Ali Asghar Heidari, Saeed Noushadian, Huiling Chen, Amir H. Gandomi
"""

import numpy as np


def initialization(nP, dim, ub, lb):
    ub = np.array(ub, dtype=float)
    lb = np.array(lb, dtype=float)
    if ub.size == 1:
        Positions = np.random.rand(nP, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((nP, dim))
        for i in range(dim):
            Positions[:, i] = np.random.rand(nP) * (ub[i] - lb[i]) + lb[i]
    return Positions


def info(nP, MaxIt, lb, ub, dim, fobj):
    """
    Weighted Mean of Vectors (INFO) Optimizer

    Parameters:
        nP    : Number of population particles
        MaxIt : Maximum number of iterations
        lb    : Lower bound(s)
        ub    : Upper bound(s)
        dim   : Dimensionality
        fobj  : Objective function f(x) -> float

    Returns:
        Best_Cost         : Global minimum fitness found
        Best_X            : Position vector of global optimum
        Convergence_curve : History of best fitness per iteration
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Cost = np.zeros(nP)
    M = np.zeros(nP)
    X = initialization(nP, dim, ub_arr, lb_arr)

    for i in range(nP):
        Cost[i] = fobj(X[i, :])
        M[i] = Cost[i]

    ind = np.argsort(Cost)
    Best_X = X[ind[0], :].copy()
    Best_Cost = float(Cost[ind[0]])

    Worst_Cost = float(Cost[ind[-1]])
    Worst_X = X[ind[-1], :].copy()

    I = np.random.randint(1, min(5, nP))
    Better_X = X[ind[I], :].copy()
    Better_Cost = float(Cost[ind[I]])

    Convergence_curve = np.zeros(MaxIt)

    for it in range(1, MaxIt + 1):
        alpha = 2.0 * np.exp(-4.0 * (it / MaxIt))

        M_Best = Best_Cost
        M_Better = Better_Cost
        M_Worst = Worst_Cost

        for i in range(nP):
            del_val = 2.0 * np.random.rand() * alpha - alpha
            sigm = 2.0 * np.random.rand() * alpha - alpha

            A1 = [idx for idx in range(nP) if idx != i]
            a, b, c = np.random.choice(A1, 3, replace=False)

            e = 1e-25
            epsi = e * np.random.rand()

            omg = max(M[a], M[b], M[c], 1e-16)
            MM = np.array([M[a] - M[b], M[a] - M[c], M[b] - M[c]])

            W = np.zeros(3)
            W[0] = np.cos(MM[0] + np.pi) * np.exp(-np.clip(MM[0] / omg, -50, 50))
            W[1] = np.cos(MM[1] + np.pi) * np.exp(-np.clip(MM[1] / omg, -50, 50))
            W[2] = np.cos(MM[2] + np.pi) * np.exp(-np.clip(MM[2] / omg, -50, 50))
            Wt = np.sum(W)

            WM1 = del_val * (
                W[0] * (X[a, :] - X[b, :]) + W[1] * (X[a, :] - X[c, :]) + W[2] * (X[b, :] - X[c, :])
            ) / (Wt + 1.0) + epsi

            omg2 = max(M_Best, M_Better, M_Worst, 1e-16)
            MM2 = np.array([M_Best - M_Better, M_Best - M_Better, M_Better - M_Worst])
            W2 = np.zeros(3)
            W2[0] = np.cos(MM2[0] + np.pi) * np.exp(-np.clip(MM2[0] / omg2, -50, 50))
            W2[1] = np.cos(MM2[1] + np.pi) * np.exp(-np.clip(MM2[1] / omg2, -50, 50))
            W2[2] = np.cos(MM2[2] + np.pi) * np.exp(-np.clip(MM2[2] / omg2, -50, 50))
            Wt2 = np.sum(W2)

            WM2 = del_val * (
                W2[0] * (Best_X - Better_X) + W2[1] * (Best_X - Worst_X) + W2[2] * (Better_X - Worst_X)
            ) / (Wt2 + 1.0) + epsi

            r = np.random.uniform(0.1, 0.5)
            MeanRule = r * WM1 + (1.0 - r) * WM2

            if np.random.rand() < 0.5:
                denom1 = M_Best - M[a] + 1.0
                denom2 = M[a] - M[b] + 1.0
                z1 = X[i, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (Best_X - X[a, :]) / (denom1 if abs(denom1) > 1e-12 else 1.0)
                z2 = Best_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (denom2 if abs(denom2) > 1e-12 else 1.0)
            else:
                denom1 = M[b] - M[c] + 1.0
                denom2 = M[a] - M[b] + 1.0
                z1 = X[a, :] + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[b, :] - X[c, :]) / (denom1 if abs(denom1) > 1e-12 else 1.0)
                z2 = Better_X + sigm * (np.random.rand() * MeanRule) + np.random.randn() * (X[a, :] - X[b, :]) / (denom2 if abs(denom2) > 1e-12 else 1.0)

            # Vector combining stage
            u = np.zeros(dim)
            for j in range(dim):
                mu = 0.05 * np.random.randn()
                if np.random.rand() < 0.5:
                    if np.random.rand() < 0.5:
                        u[j] = z1[j] + mu * abs(z1[j] - z2[j])
                    else:
                        u[j] = z2[j] + mu * abs(z1[j] - z2[j])
                else:
                    u[j] = X[i, j]

            # Local search stage
            if np.random.rand() < 0.5:
                L = 1.0 if np.random.rand() < 0.5 else 0.0
                v1 = (1.0 - L) * 2.0 * np.random.rand() + L
                v2 = np.random.rand() * L + (1.0 - L)
                Xavg = (X[a, :] + X[b, :] + X[c, :]) / 3.0
                phi = np.random.rand()
                Xrnd = phi * Xavg + (1.0 - phi) * (phi * Better_X + (1.0 - phi) * Best_X)
                Randn = L * np.random.randn(dim) + (1.0 - L) * np.random.randn()
                if np.random.rand() < 0.5:
                    u = Best_X + Randn * (MeanRule + np.random.randn() * (Best_X - X[a, :]))
                else:
                    u = Xrnd + Randn * (MeanRule + np.random.randn() * (v1 * Best_X - v2 * Xrnd))

            New_X = np.clip(u, lb_arr, ub_arr)
            New_Cost = fobj(New_X)

            if New_Cost < Cost[i]:
                X[i, :] = New_X.copy()
                Cost[i] = New_Cost
                M[i] = Cost[i]
                if Cost[i] < Best_Cost:
                    Best_X = X[i, :].copy()
                    Best_Cost = float(Cost[i])

        ind = np.argsort(Cost)
        Worst_X = X[ind[-1], :].copy()
        Worst_Cost = float(Cost[ind[-1]])
        I = np.random.randint(1, min(5, nP))
        Better_X = X[ind[I], :].copy()
        Better_Cost = float(Cost[ind[I]])

        Convergence_curve[it - 1] = Best_Cost

    return Best_Cost, Best_X, Convergence_curve


INFO = info


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="INFO",
        algo_func=info,
        full_name="Weighted Mean of Vectors (INFO)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
