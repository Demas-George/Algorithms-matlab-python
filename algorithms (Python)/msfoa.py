"""
Modified Starfish Optimization Algorithm (MSFOA)

Derived from:
    Sources(MATLAB)/MSFOA/MSFOA-MATLAB-main/MSFOA.m
    Author: H. Akbulut (2026)
    Paper: "A modified starfish optimization algorithm (M-SFOA) for global optimization problems
           and its application to heart disease risk prediction"
           Expert Systems with Applications, vol. 307, Article 131088.
           DOI: 10.1016/j.eswa.2026.131088

Original MATLAB Function Signature:
    [xposbest, fvalbest, Curve] = MSFOA(Npop, Max_it, lb, ub, nD, fobj)

Python Entry Point:
    msfoa(Npop, Max_it, lb, ub, nD, fobj)

Parameters:
    Npop     : int, population size
    Max_it   : int, maximum number of iterations
    lb       : float or numpy.ndarray of shape (nD,), lower boundaries
    ub       : float or numpy.ndarray of shape (nD,), upper boundaries
    nD       : int, dimensionality of the problem
    fobj     : callable, objective function f(x) returning a scalar float

Returns:
    xposbest : numpy.ndarray of shape (nD,), optimal solution vector found
    fvalbest : float, best objective function value found
    Curve    : numpy.ndarray of shape (Max_it,), convergence history
"""

import numpy as np


def msfoa(Npop, Max_it, lb, ub, nD, fobj):
    """
    Execute the Modified Starfish Optimization Algorithm.
    """
    lb = np.full(nD, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(nD, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    GP = 0.5
    step_size_initial = 1.0
    step_size_final = 0.01
    momentum_factor = 0.7

    Curve = np.zeros(Max_it)
    Xpos = np.zeros((Npop, nD))
    Fitness = np.zeros(Npop)

    for i in range(nD):
        Xpos[:, i] = np.random.rand(Npop) * (ub[i] - lb[i]) + lb[i]

    for i in range(Npop):
        Fitness[i] = fobj(Xpos[i, :])

    best_idx = np.argmin(Fitness)
    fvalbest = Fitness[best_idx]
    xposbest = Xpos[best_idx, :].copy()
    prev_best = xposbest.copy()

    newX = np.zeros((Npop, nD))

    for T in range(1, Max_it + 1):
        current_step_size = step_size_initial * ((step_size_final / step_size_initial) ** (T / Max_it))
        momentum = xposbest - prev_best
        prev_best = xposbest.copy()

        theta = (np.pi / 2.0) * (T / Max_it)
        tEO = ((Max_it - T) / Max_it) * np.cos(theta)

        if np.random.rand() < GP:
            # Exploration phase
            for i in range(Npop):
                if nD > 5:
                    jp1 = np.random.choice(nD, size=5, replace=False)
                    for j in range(5):
                        dim_idx = jp1[j]
                        pm = (2.0 * np.random.rand() - 1.0) * np.pi
                        if np.random.rand() < GP:
                            newX[i, dim_idx] = (Xpos[i, dim_idx]
                                                + current_step_size * (pm * (xposbest[dim_idx] - Xpos[i, dim_idx]) * np.cos(theta)
                                                                      + momentum_factor * momentum[dim_idx]))
                        else:
                            newX[i, dim_idx] = (Xpos[i, dim_idx]
                                                - pm * (xposbest[dim_idx] - Xpos[i, dim_idx]) * np.sin(theta))

                        if newX[i, dim_idx] > ub[dim_idx] or newX[i, dim_idx] < lb[dim_idx]:
                            newX[i, dim_idx] = Xpos[i, dim_idx]
                else:
                    jp2 = np.random.randint(0, nD)
                    im = np.random.choice(Npop, size=2, replace=False)
                    rand1 = 2.0 * np.random.rand() - 1.0
                    rand2 = 2.0 * np.random.rand() - 1.0

                    candidate1 = (tEO * Xpos[i, jp2]
                                  + rand1 * (Xpos[im[0], jp2] - Xpos[i, jp2])
                                  + rand2 * (Xpos[im[1], jp2] - Xpos[i, jp2]))
                    candidate2 = (Xpos[i, jp2]
                                  + current_step_size * (rand1 * (xposbest[jp2] - Xpos[i, jp2])
                                                        + momentum_factor * momentum[jp2]))
                    candidate3 = Xpos[i, jp2] + current_step_size * np.random.randn()

                    candidates = np.array([candidate1, candidate2, candidate3])
                    best_cand_idx = np.argmin(np.abs(candidates - xposbest[jp2]))
                    newX[i, jp2] = candidates[best_cand_idx]

                    if newX[i, jp2] > ub[jp2] or newX[i, jp2] < lb[jp2]:
                        newX[i, jp2] = Xpos[i, jp2]

                newX[i, :] = np.clip(newX[i, :], lb, ub)
        else:
            # Exploitation phase
            df = np.random.choice(Npop, size=5, replace=False)
            dm = np.zeros((5, nD))
            for k in range(5):
                dm[k, :] = xposbest - Xpos[df[k], :]

            for i in range(Npop):
                r1 = np.random.rand()
                r2 = np.random.rand()
                kp = np.random.choice(5, size=2, replace=False)

                move1 = Xpos[i, :] + r1 * dm[kp[0], :] + r2 * dm[kp[1], :]
                move2 = Xpos[i, :] + current_step_size * (r1 * dm[kp[0], :] + momentum_factor * momentum)

                if i == Npop - 1:
                    move3 = np.exp(-T * Npop / Max_it) * Xpos[i, :]
                else:
                    move3 = Xpos[i, :] + current_step_size * np.random.randn(nD)

                candidates = [move1, move2, move3]
                fit_cands = [fobj(c) for c in candidates]
                best_cand_idx = np.argmin(fit_cands)
                newX[i, :] = np.clip(candidates[best_cand_idx], lb, ub)

        for i in range(Npop):
            newFit = fobj(newX[i, :])
            if newFit < Fitness[i]:
                Fitness[i] = newFit
                Xpos[i, :] = newX[i, :]
                if newFit < fvalbest:
                    fvalbest = newFit
                    xposbest = Xpos[i, :].copy()

        Curve[T - 1] = fvalbest

    return xposbest, fvalbest, Curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def msfoa_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        pos, score, curve = msfoa(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
        return score, pos, curve

    run_experiment(
        algo_name="MSFOA",
        algo_func=msfoa_adapter,
        full_name="Modified Starfish Optimization Algorithm (MSFOA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

