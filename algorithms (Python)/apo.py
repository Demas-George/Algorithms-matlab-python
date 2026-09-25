"""
Artificial Protozoa Optimizer (APO)

Derived from:
    Sources(MATLAB)/Artificial Protozoa Optimizer (APO)/Artificial Protozoa Optimizer/APO-CEC2022-Matlab/APO_func.m
    Authors: X. Wang et al. (2024)
    Paper: "Artificial protozoa optimizer: A novel bio-inspired metaheuristic algorithm
           for engineering optimization problems"
           Knowledge-Based Systems.

Original MATLAB Function Signature:
    [bestProtozoa,bestFit,recordtime] = APO_func(fhd,dim,pop_size,iter_max,Xmin,Xmax,varargin)

Python Entry Point:
    apo(pop_size, iter_max, Xmin, Xmax, dim, fobj)

Parameters:
    pop_size : int, number of protozoa in the population (ps)
    iter_max : int, maximum number of iterations
    Xmin     : float or numpy.ndarray of shape (dim,), lower boundaries
    Xmax     : float or numpy.ndarray of shape (dim,), upper boundaries
    dim      : int, dimensionality of the problem
    fobj     : callable, objective function f(x) returning a scalar float

Returns:
    bestProtozoa      : numpy.ndarray of shape (dim,), global optimal position vector
    bestFit           : float, global optimal fitness score
    convergence_curve : numpy.ndarray of shape (iter_max,), best fitness at each iteration
"""

import numpy as np


def apo(pop_size, iter_max, Xmin, Xmax, dim, fobj):
    """
    Execute the Artificial Protozoa Optimizer algorithm.
    """
    Xmin = np.full(dim, Xmin, dtype=float) if np.isscalar(Xmin) else np.asarray(Xmin, dtype=float)
    Xmax = np.full(dim, Xmax, dtype=float) if np.isscalar(Xmax) else np.asarray(Xmax, dtype=float)

    ps = pop_size
    n_pairs = 1  # np in source: neighbor pairs
    pf_max = 0.1  # proportion fraction maximum
    eps = 1e-15

    protozoa = np.zeros((ps, dim))
    for i in range(ps):
        protozoa[i, :] = Xmin + np.random.rand(dim) * (Xmax - Xmin)

    protozoa_Fit = np.zeros(ps)
    for i in range(ps):
        protozoa_Fit[i] = fobj(protozoa[i, :])

    bestid = np.argmin(protozoa_Fit)
    bestProtozoa = protozoa[bestid, :].copy()
    bestFit = protozoa_Fit[bestid]

    convergence_curve = np.zeros(iter_max)
    convergence_curve[0] = bestFit

    newprotozoa = np.zeros((ps, dim))
    epn = np.zeros((n_pairs, dim))

    for iter_idx in range(2, iter_max + 1):
        # Sort protozoa by fitness
        index = np.argsort(protozoa_Fit)
        protozoa_Fit = protozoa_Fit[index]
        protozoa = protozoa[index, :]

        pf = pf_max * np.random.rand()
        k_count = int(np.ceil(ps * pf))
        # 0-based indices for protozoa in dormancy or reproduction forms
        ri = set(np.random.choice(ps, size=k_count, replace=False))

        for i in range(ps):
            # 1-based index equivalent for formula: (1 - (i+1)/ps)*pi
            rank_ratio = 1.0 - (i + 1) / ps

            if i in ri:
                # Dormancy or reproduction form
                pdr = 0.5 * (1.0 + np.cos(rank_ratio * np.pi))
                if np.random.rand() < pdr:
                    # Dormancy form
                    newprotozoa[i, :] = Xmin + np.random.rand(dim) * (Xmax - Xmin)
                else:
                    # Reproduction form
                    flag = 1.0 if np.random.rand() < 0.5 else -1.0
                    Mr = np.zeros(dim)
                    num_mut = int(np.ceil(np.random.rand() * dim))
                    Mr[np.random.choice(dim, size=num_mut, replace=False)] = 1.0
                    newprotozoa[i, :] = (protozoa[i, :]
                                         + flag * np.random.rand() * (Xmin + np.random.rand(dim) * (Xmax - Xmin)) * Mr)
            else:
                # Foraging form
                f = np.random.rand() * (1.0 + np.cos((iter_idx / iter_max) * np.pi))
                num_map = int(np.ceil(dim * (i + 1) / ps))
                Mf = np.zeros(dim)
                Mf[np.random.choice(dim, size=num_map, replace=False)] = 1.0
                pah = 0.5 * (1.0 + np.cos((iter_idx / iter_max) * np.pi))

                if np.random.rand() < pah:
                    # Autotroph form
                    j = np.random.randint(0, ps)
                    for k in range(n_pairs):
                        if i == 0:
                            km = 0
                            kp = np.random.randint(1, ps)
                        elif i == ps - 1:
                            km = np.random.randint(0, ps - 1)
                            kp = ps - 1
                        else:
                            km = np.random.randint(0, i)
                            kp = np.random.randint(i + 1, ps)

                        denom = protozoa_Fit[kp] + eps
                        wa = np.exp(-abs(protozoa_Fit[km] / denom))
                        epn[k, :] = wa * (protozoa[km, :] - protozoa[kp, :])

                    newprotozoa[i, :] = (protozoa[i, :]
                                         + f * (protozoa[j, :] - protozoa[i, :] + (1.0 / n_pairs) * np.sum(epn, axis=0)) * Mf)
                else:
                    # Heterotroph form
                    for k in range(1, n_pairs + 1):
                        imk = max(0, i - k)
                        ipk = min(ps - 1, i + k)
                        denom = protozoa_Fit[ipk] + eps
                        wh = np.exp(-abs(protozoa_Fit[imk] / denom))
                        epn[k - 1, :] = wh * (protozoa[imk, :] - protozoa[ipk, :])

                    flag = 1.0 if np.random.rand() < 0.5 else -1.0
                    Xnear = (1.0 + flag * np.random.rand(dim) * (1.0 - iter_idx / iter_max)) * protozoa[i, :]
                    newprotozoa[i, :] = (protozoa[i, :]
                                         + f * (Xnear - protozoa[i, :] + (1.0 / n_pairs) * np.sum(epn, axis=0)) * Mf)

        # Boundary clamping
        newprotozoa = np.clip(newprotozoa, Xmin, Xmax)

        newprotozoa_Fit = np.zeros(ps)
        for i in range(ps):
            newprotozoa_Fit[i] = fobj(newprotozoa[i, :])

        improved = newprotozoa_Fit < protozoa_Fit
        protozoa[improved, :] = newprotozoa[improved, :]
        protozoa_Fit[improved] = newprotozoa_Fit[improved]

        cur_min_idx = np.argmin(protozoa_Fit)
        if protozoa_Fit[cur_min_idx] < bestFit:
            bestFit = protozoa_Fit[cur_min_idx]
            bestProtozoa = protozoa[cur_min_idx, :].copy()

        convergence_curve[iter_idx - 1] = bestFit

    return bestProtozoa, bestFit, convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    def apo_adapter(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
        pos, score, curve = apo(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
        return score, pos, curve

    run_experiment(
        algo_name="APO",
        algo_func=apo_adapter,
        full_name="Artificial Protozoa Optimizer (APO)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

