"""
Marine Predators Algorithm (MPA)

Derived from:
    Sources(MATLAB)/Marine-Predators-Algorithm (MPA)/Marine-Predators-Algorithm-master/MPA.m
    Authors: Afshin Faramarzi, Mohammad Heidarinejad, Seyedali Mirjalili, Amir H. Gandomi (2020)
    Paper: "Marine Predators Algorithm: A Nature-inspired Metaheuristic"
           Expert Systems with Applications, 152, 113377.
           DOI: 10.1016/j.eswa.2020.113377

Original MATLAB Function Signature:
    [Top_predator_fit, Top_predator_pos, Convergence_curve] = MPA(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

Python Entry Point:
    mpa(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

Parameters:
    SearchAgents_no   : int, population size (number of preys/predators)
    Max_iter          : int, maximum number of iterations
    lb                : float or numpy.ndarray of shape (dim,), lower boundaries
    ub                : float or numpy.ndarray of shape (dim,), upper boundaries
    dim               : int, problem dimension
    fobj              : callable, objective function f(x) returning a scalar float

Returns:
    Top_predator_fit  : float, best objective value found
    Top_predator_pos  : numpy.ndarray of shape (dim,), optimal position vector
    Convergence_curve : numpy.ndarray of shape (Max_iter,), convergence history
"""

import math
import numpy as np


def _levy(n, m, beta=1.5):
    num = math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma_u = (num / den) ** (1.0 / beta)
    u = np.random.normal(0.0, sigma_u, size=(n, m))
    v = np.random.normal(0.0, 1.0, size=(n, m))
    return u / (np.abs(v) ** (1.0 / beta))


def mpa(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    Execute the Marine Predators Algorithm.
    """
    lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)

    Top_predator_pos = np.zeros(dim)
    Top_predator_fit = np.inf

    Convergence_curve = np.zeros(Max_iter)
    fitness = np.full(SearchAgents_no, np.inf)

    Prey = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Prey[:, i] = np.random.rand(SearchAgents_no) * (ub[i] - lb[i]) + lb[i]

    Xmin = np.tile(lb, (SearchAgents_no, 1))
    Xmax = np.tile(ub, (SearchAgents_no, 1))

    FADs = 0.2
    P = 0.5
    fit_old = None
    Prey_old = None

    for Iter in range(Max_iter):
        # 1. Detecting top predator
        Prey = np.clip(Prey, lb, ub)
        for i in range(SearchAgents_no):
            fitness[i] = fobj(Prey[i, :])
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :].copy()

        # Marine Memory saving
        if Iter == 0:
            fit_old = fitness.copy()
            Prey_old = Prey.copy()

        Inx = fit_old < fitness
        Prey = np.where(Inx[:, None], Prey_old, Prey)
        fitness = np.where(Inx, fit_old, fitness)
        fit_old = fitness.copy()
        Prey_old = Prey.copy()

        Elite = np.tile(Top_predator_pos, (SearchAgents_no, 1))
        CF = (1.0 - Iter / Max_iter) ** (2.0 * Iter / Max_iter)

        RL = 0.05 * _levy(SearchAgents_no, dim, 1.5)
        RB = np.random.randn(SearchAgents_no, dim)

        stepsize = np.zeros((SearchAgents_no, dim))

        for i in range(SearchAgents_no):
            for j in range(dim):
                R = np.random.rand()
                # Phase 1
                if Iter < Max_iter / 3.0:
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
                # Phase 2
                elif Iter < 2.0 * Max_iter / 3.0:
                    if i >= SearchAgents_no // 2:
                        stepsize[i, j] = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                    else:
                        stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                        Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
                # Phase 3
                else:
                    stepsize[i, j] = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                    Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]

        # 2. Detecting top predator
        Prey = np.clip(Prey, lb, ub)
        for i in range(SearchAgents_no):
            fitness[i] = fobj(Prey[i, :])
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :].copy()

        # Marine Memory saving
        Inx = fit_old < fitness
        Prey = np.where(Inx[:, None], Prey_old, Prey)
        fitness = np.where(Inx, fit_old, fitness)
        fit_old = fitness.copy()
        Prey_old = Prey.copy()

        # Eddy formation and FADs effect
        if np.random.rand() < FADs:
            U = (np.random.rand(SearchAgents_no, dim) < FADs).astype(float)
            Prey = Prey + CF * ((Xmin + np.random.rand(SearchAgents_no, dim) * (Xmax - Xmin)) * U)
        else:
            r = np.random.rand()
            perm1 = np.random.permutation(SearchAgents_no)
            perm2 = np.random.permutation(SearchAgents_no)
            fads_step = (FADs * (1.0 - r) + r) * (Prey[perm1, :] - Prey[perm2, :])
            Prey = Prey + fads_step

        Convergence_curve[Iter] = Top_predator_fit

    return Top_predator_fit, Top_predator_pos, Convergence_curve


# =========================================================
# Run Configuration & Execution
# =========================================================
if __name__ == "__main__":
    from problem import run_experiment

    n_agents = 30
    max_iter = 1000
    n_runs   = 30

    run_experiment(
        algo_name="MPA",
        algo_func=mpa,
        full_name="Marine Predators Algorithm (MPA)",
        n_agents=n_agents,
        max_iter=max_iter,
        n_runs=n_runs,
    )

