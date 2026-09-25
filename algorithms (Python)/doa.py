"""
Dream Optimization Algorithm (DOA)

Derived from:
    Sources(MATLAB)/Dream-Optimization-Algorithm (DOA)/DOA.m
    Authors: Original DOA research team (MATLAB implementation)
    Paper: "Dream Optimization Algorithm: A novel metaheuristic for numerical optimization"

Original MATLAB Function Signature:
    [fbest, sbest, fbest_history] = DOA(pop, T, lb, ub, D, fobj)

Python Entry Point:
    doa(pop, T, lb, ub, D, fobj)

Parameters:
    pop           : int, population size
    T             : int, maximum number of iterations
    lb            : float or numpy.ndarray of shape (D,), lower search boundaries
    ub            : float or numpy.ndarray of shape (D,), upper search boundaries
    D             : int, problem dimension
    fobj          : callable, objective function f(x) returning a scalar float

Returns:
    fbest         : float, optimal fitness value found
    sbest         : numpy.ndarray of shape (D,), optimal solution vector
    fbest_history : numpy.ndarray of shape (T,), history of best fitness per iteration
"""

import numpy as np


def doa(pop, T, lb, ub, D, fobj):
    """
    Execute the Dream Optimization Algorithm.
    """
    lb = np.full(D, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(D, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    x = np.zeros((pop, D))
    for i in range(D):
        x[:, i] = np.random.rand(pop) * (ub[i] - lb[i]) + lb[i]

    sbest = np.ones(D)
    sbestd = np.ones((5, D))
    fbest = np.inf
    fbestd = np.full(5, np.inf)
    fbest_history = np.zeros(T)

    t_split = int(np.floor(9.0 * T / 10.0))

    # Exploration phase
    for i in range(1, t_split + 1):
        for m in range(1, 6):
            low_k = max(1, int(np.ceil(D / 8.0 / m)))
            high_k = max(low_k, int(np.ceil(D / 3.0 / m)))
            k = np.random.randint(low_k, high_k + 1)

            group_start = int((m - 1) * pop / 5)
            group_end = int(m * pop / 5)

            for j in range(group_start, group_end):
                val = fobj(x[j, :])
                if val < fbestd[m - 1]:
                    sbestd[m - 1, :] = x[j, :].copy()
                    fbestd[m - 1] = val

            for j in range(group_start, group_end):
                x[j, :] = sbestd[m - 1, :].copy()
                in_dims = np.random.choice(D, size=k, replace=False)

                if np.random.rand() < 0.9:
                    factor = (np.cos((i + T / 10.0) * np.pi / T) + 1.0) / 2.0
                    for h in range(k):
                        dim_idx = in_dims[h]
                        perturbation = np.random.rand() * (ub[dim_idx] - lb[dim_idx]) + lb[dim_idx]
                        x[j, dim_idx] += perturbation * factor

                        if x[j, dim_idx] > ub[dim_idx] or x[j, dim_idx] < lb[dim_idx]:
                            if D > 15:
                                candidates = [c for c in range(pop) if c != j]
                                sel = candidates[np.random.randint(0, len(candidates))]
                                x[j, dim_idx] = x[sel, dim_idx]
                            else:
                                x[j, dim_idx] = np.random.rand() * (ub[dim_idx] - lb[dim_idx]) + lb[dim_idx]
                else:
                    for h in range(k):
                        dim_idx = in_dims[h]
                        x[j, dim_idx] = x[np.random.randint(0, pop), dim_idx]

            if fbestd[m - 1] < fbest:
                fbest = fbestd[m - 1]
                sbest = sbestd[m - 1, :].copy()

        fbest_history[i - 1] = fbest

    # Exploitation phase
    for i in range(t_split + 1, T + 1):
        for p in range(pop):
            val = fobj(x[p, :])
            if val < fbest:
                sbest = x[p, :].copy()
                fbest = val

        for j in range(pop):
            km = max(2, int(np.ceil(D / 3.0)))
            k = np.random.randint(2, km + 1)
            x[j, :] = sbest.copy()
            in_dims = np.random.choice(D, size=k, replace=False)

            factor = (np.cos(i * np.pi / T) + 1.0) / 2.0
            for h in range(k):
                dim_idx = in_dims[h]
                perturbation = np.random.rand() * (ub[dim_idx] - lb[dim_idx]) + lb[dim_idx]
                x[j, dim_idx] += perturbation * factor

                if x[j, dim_idx] > ub[dim_idx] or x[j, dim_idx] < lb[dim_idx]:
                    if D > 15:
                        candidates = [c for c in range(pop) if c != j]
                        sel = candidates[np.random.randint(0, len(candidates))]
                        x[j, dim_idx] = x[sel, dim_idx]
                    else:
                        x[j, dim_idx] = np.random.rand() * (ub[dim_idx] - lb[dim_idx]) + lb[dim_idx]

        fbest_history[i - 1] = fbest

    return fbest, sbest, fbest_history


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="DOA",
        algo_func=doa,
        full_name="Dream Optimization Algorithm (DOA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

