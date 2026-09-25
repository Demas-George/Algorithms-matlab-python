"""
Runge Kutta Optimization (RUN)

Derived from:
    Sources(MATLAB)/Runge Kutta Optimization (RUN)/Runge Kutta Optimization (RUN)-2021/RUN.m
    Authors: Iman Ahmadianfar, Ali Asghar Heidari, Amir H. Gandomi, Xuefeng Chu, Huiling Chen (2021)
    Paper: "RUN Beyond the Metaphor: An Efficient Optimization Algorithm Based on Runge Kutta Method"
           Expert Systems With Applications, 181, 115079.
           DOI: 10.1016/j.eswa.2021.115079

Original MATLAB Function Signature:
    [Best_Cost, Best_X, Convergence_curve] = RUN(nP, MaxIt, lb, ub, dim, fobj)

Python Entry Point:
    run_opt(nP, MaxIt, lb, ub, dim, fobj)

Parameters:
    nP                : int, population size (number of search agents)
    MaxIt             : int, maximum number of iterations
    lb                : float or numpy.ndarray of shape (dim,), lower boundaries
    ub                : float or numpy.ndarray of shape (dim,), upper boundaries
    dim               : int, dimensionality of the problem
    fobj              : callable, objective function f(x) returning a scalar float

Returns:
    Best_Cost         : float, best objective value found
    Best_X            : numpy.ndarray of shape (dim,), optimal solution vector
    Convergence_curve : numpy.ndarray of shape (MaxIt,), convergence history
"""

import numpy as np


def _runge_kutta(XB, XW, DelX, dim):
    """
    Search mechanism of RUN based on 4th-order Runge Kutta method.
    """
    C = np.random.randint(1, 3) * (1.0 - np.random.rand())
    r1 = np.random.rand(dim)
    r2 = np.random.rand(dim)

    K1 = 0.5 * (np.random.rand() * XW - C * XB)
    K2 = 0.5 * (np.random.rand() * (XW + r2 * K1 * DelX / 2.0) - (C * XB + r1 * K1 * DelX / 2.0))
    K3 = 0.5 * (np.random.rand() * (XW + r2 * K2 * DelX / 2.0) - (C * XB + r1 * K2 * DelX / 2.0))
    K4 = 0.5 * (np.random.rand() * (XW + r2 * K3 * DelX) - (C * XB + r1 * K3 * DelX))

    XRK = K1 + 2.0 * K2 + 2.0 * K3 + K4
    return (1.0 / 6.0) * XRK


def _rnd_x(nP, i):
    candidates = [k for k in range(nP) if k != i]
    if len(candidates) < 3:
        perm = np.random.choice(candidates, size=3, replace=True)
    else:
        perm = np.random.choice(candidates, size=3, replace=False)
    return perm[0], perm[1], perm[2]


def run_opt(nP, MaxIt, lb, ub, dim, fobj):
    """
    Execute the Runge Kutta Optimization algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Cost = np.zeros(nP)
    X = np.zeros((nP, dim))
    for i in range(dim):
        X[:, i] = np.random.rand(nP) * (ub[i] - lb[i]) + lb[i]

    for i in range(nP):
        Cost[i] = fobj(X[i, :])

    best_idx = np.argmin(Cost)
    Best_Cost = Cost[best_idx]
    Best_X = X[best_idx, :].copy()

    Convergence_curve = np.zeros(MaxIt)
    Convergence_curve[0] = Best_Cost

    for it in range(2, MaxIt + 1):
        f = 20.0 * np.exp(-12.0 * (it / MaxIt))
        Xavg = np.mean(X, axis=0)
        SF = 2.0 * (0.5 - np.random.rand(nP)) * f

        for i in range(nP):
            ind_l = np.argmin(Cost)
            lBest = X[ind_l, :].copy()

            A, B, C = _rnd_x(nP, i)
            sub_indices = [A, B, C]
            ind1 = sub_indices[np.argmin([Cost[A], Cost[B], Cost[C]])]

            gama = np.random.rand() * (X[i, :] - np.random.rand(dim) * (ub - lb)) * np.exp(-4.0 * it / MaxIt)
            Stp = np.random.rand(dim) * ((Best_X - np.random.rand() * Xavg) + gama)
            DelX = 2.0 * np.random.rand(dim) * np.abs(Stp)

            if Cost[i] < Cost[ind1]:
                Xb = X[i, :].copy()
                Xw = X[ind1, :].copy()
            else:
                Xb = X[ind1, :].copy()
                Xw = X[i, :].copy()

            SM = _runge_kutta(Xb, Xw, DelX, dim)

            L = np.random.rand(dim) < 0.5
            Xc = np.where(L, X[i, :], X[A, :])
            Xm = np.where(L, Best_X, lBest)

            flag = np.random.randint(0, 2, size=dim)
            r = np.where(flag == 0, 1.0, -1.0)
            g = 2.0 * np.random.rand()
            mu = 0.5 + 0.1 * np.random.randn(dim)

            if np.random.rand() < 0.5:
                Xnew = (Xc + r * SF[i] * g * Xc) + SF[i] * SM + mu * (Xm - Xc)
            else:
                Xnew = (Xm + r * SF[i] * g * Xm) + SF[i] * SM + mu * (X[A, :] - X[B, :])

            Xnew = np.clip(Xnew, lb, ub)
            CostNew = fobj(Xnew)

            if CostNew < Cost[i]:
                X[i, :] = Xnew
                Cost[i] = CostNew

            # Enhanced solution quality (ESQ)
            if np.random.rand() < 0.5:
                EXP = np.exp(-5.0 * np.random.rand() * it / MaxIt)
                r_int = int(np.floor(np.random.uniform(-1.0, 2.0)))

                u = 2.0 * np.random.rand(dim)
                w = np.random.uniform(0.0, 2.0, size=dim) * EXP

                A, B, C = _rnd_x(nP, i)
                Xavg_esq = (X[A, :] + X[B, :] + X[C, :]) / 3.0

                beta = np.random.rand(dim)
                Xnew1 = beta * Best_X + (1.0 - beta) * Xavg_esq

                Xnew2 = np.zeros(dim)
                for j in range(dim):
                    if w[j] < 1.0:
                        Xnew2[j] = Xnew1[j] + r_int * w[j] * abs((Xnew1[j] - Xavg_esq[j]) + np.random.randn())
                    else:
                        Xnew2[j] = (Xnew1[j] - Xavg_esq[j]) + r_int * w[j] * abs((u[j] * Xnew1[j] - Xavg_esq[j]) + np.random.randn())

                Xnew2 = np.clip(Xnew2, lb, ub)
                CostNew = fobj(Xnew2)

                if CostNew < Cost[i]:
                    X[i, :] = Xnew2
                    Cost[i] = CostNew
                else:
                    if np.random.rand() < w[np.random.randint(0, dim)]:
                        SM2 = _runge_kutta(X[i, :], Xnew2, DelX, dim)
                        Xnew = (Xnew2 - np.random.rand() * Xnew2) + SF[i] * (SM2 + (2.0 * np.random.rand(dim) * Best_X - Xnew2))
                        Xnew = np.clip(Xnew, lb, ub)
                        CostNew = fobj(Xnew)
                        if CostNew < Cost[i]:
                            X[i, :] = Xnew
                            Cost[i] = CostNew

            if Cost[i] < Best_Cost:
                Best_X = X[i, :].copy()
                Best_Cost = Cost[i]

        Convergence_curve[it - 1] = Best_Cost

    return Best_Cost, Best_X, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="RUN",
        algo_func=run_opt,
        full_name="Runge Kutta Optimization (RUN)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

