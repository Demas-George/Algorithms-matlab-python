"""
Escape Optimization Algorithm (ESC)
Source: Escape: An optimization method based on crowd evacuation behaviors
        Journal of Artificial Intelligence Review, 2024.
Authors: Kaichen OuYang, Shengwei Fu, Yi Chen, Qifeng Cai, Ali Asghar Heidari, Huiling Chen
"""

import math
import numpy as np


def initialization(SearchAgents_no, dim, ub, lb):
    ub = np.array(ub, dtype=float)
    lb = np.array(lb, dtype=float)
    if ub.size == 1:
        Positions = np.random.rand(SearchAgents_no, dim) * (ub - lb) + lb
    else:
        Positions = np.zeros((SearchAgents_no, dim))
        for i in range(dim):
            Positions[:, i] = np.random.rand(SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
    return Positions


def _adaptive_levy_weight(beta_base, dim, t, maxIter):
    beta = beta_base + 0.5 * np.sin(np.pi / 2.0 * t / maxIter)
    beta = max(min(beta, 2.0), 0.1)
    num = math.gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0)
    den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
    sigma = (num / den) ** (1.0 / beta)
    u = np.random.normal(0.0, sigma, dim)
    v = np.random.normal(0.0, 1.0, dim)
    denom_v = np.abs(v) ** (1.0 / beta)
    denom_v[denom_v == 0] = 1e-16
    w = np.abs(u / denom_v)
    max_w = np.max(w)
    w = w / (max_w + 1e-16)
    return w


def esc(N, maxIter, lb, ub, dim, fobj):
    """
    Escape Optimization Algorithm (ESC)

    Parameters:
        N       : Population size (number of agents)
        maxIter : Maximum number of iterations
        lb      : Lower bounds (scalar or vector)
        ub      : Upper bounds (scalar or vector)
        dim     : Dimensionality
        fobj    : Objective function handle f(x) -> float

    Returns:
        best_fitness      : Global minimum objective found
        best_solution     : Global best solution vector
        Convergence_curve : History of best fitness per iteration
    """
    lb_arr = np.ones(dim) * lb if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.ones(dim) * ub if np.isscalar(ub) else np.asarray(ub, dtype=float)

    population = initialization(N, dim, ub_arr, lb_arr)
    fitness = np.zeros(N)
    for i in range(N):
        fitness[i] = fobj(population[i, :])

    idx = np.argsort(fitness)
    fitness = fitness[idx]
    population = population[idx, :]

    eliteSize = min(5, N)
    best_solutions = population[0:eliteSize, :].copy()
    beta_base = 1.5
    mask_probability = 0.5
    fitness_history = np.zeros(maxIter)

    t = 0
    while t < maxIter:
        panicIndex = np.cos(np.pi / 2.0 * (t / (3.0 * maxIter)))
        idx = np.argsort(fitness)
        fitness = fitness[idx]
        population = population[idx, :]

        a = 0.15
        b = 0.35

        populationNew = population.copy()
        if t / maxIter <= 0.5:
            calmCount = int(math.ceil(round(a * N)))
            calmCount = max(1, min(calmCount, N - 2))
            conformCount = int(math.ceil(round(b * N)))
            conformCount = max(1, min(conformCount, N - calmCount - 1))

            calm = population[0:calmCount, :]
            conform = population[calmCount:calmCount + conformCount, :]
            panic = population[calmCount + conformCount:, :]
            calmCenter = np.mean(calm, axis=0)

            for i in range(N):
                panicIndividual = panic[np.random.randint(0, len(panic)), :] if len(panic) > 0 else calmCenter
                mask1 = (np.random.rand(dim) > mask_probability).astype(float)

                if i < calmCount:
                    minCalm = np.min(calm, axis=0)
                    maxCalm = np.max(calm, axis=0)
                    randomPositionVector = minCalm + np.random.rand(dim) * (maxCalm - minCalm)
                    weightVector1 = _adaptive_levy_weight(beta_base, dim, t, maxIter)
                    populationNew[i, :] = population[i, :] + (
                        mask1 * (weightVector1 * (calmCenter - population[i, :]) +
                        (randomPositionVector - population[i, :] + np.random.randn(dim) / 50.0)) * panicIndex
                    )
                elif i < calmCount + conformCount:
                    minConform = np.min(conform, axis=0)
                    maxConform = np.max(conform, axis=0)
                    randomPositionVector = minConform + np.random.rand(dim) * (maxConform - minConform)
                    weightVector1 = _adaptive_levy_weight(beta_base, dim, t, maxIter)
                    weightVector2 = _adaptive_levy_weight(beta_base, dim, t, maxIter)
                    mask2 = (np.random.rand(dim) > mask_probability).astype(float)
                    populationNew[i, :] = population[i, :] + (
                        mask1 * (weightVector1 * (calmCenter - population[i, :])) +
                        mask2 * (weightVector2 * (panicIndividual - population[i, :])) +
                        (randomPositionVector - population[i, :] + np.random.randn(dim) / 50.0) * panicIndex
                    )
                else:
                    elite_idx = np.random.randint(0, eliteSize)
                    elite = best_solutions[elite_idx, :]
                    randomIndividual = population[np.random.randint(0, N), :]
                    weightVector1 = _adaptive_levy_weight(beta_base, dim, t, maxIter)
                    weightVector2 = _adaptive_levy_weight(beta_base, dim, t, maxIter)
                    mask2 = (np.random.rand(dim) > mask_probability).astype(float)
                    randomPositionVector = elite + weightVector1 * (randomIndividual - elite)
                    populationNew[i, :] = population[i, :] + (
                        mask1 * (weightVector1 * (elite - population[i, :])) +
                        mask2 * (weightVector2 * (randomIndividual - population[i, :])) +
                        (randomPositionVector - population[i, :] + np.random.randn(dim) / 50.0) * panicIndex
                    )
        else:
            for i in range(N):
                elite_idx = np.random.randint(0, eliteSize)
                elite = best_solutions[elite_idx, :]
                random_individual = population[np.random.randint(0, N), :]
                weightVector1 = _adaptive_levy_weight(beta_base, dim, t, maxIter)
                weightVector2 = _adaptive_levy_weight(beta_base, dim, t, maxIter)
                mask1 = (np.random.rand(dim) > mask_probability).astype(float)
                mask2 = (np.random.rand(dim) > mask_probability).astype(float)
                populationNew[i, :] = population[i, :] + (
                    mask1 * (weightVector1 * (elite - population[i, :])) +
                    mask2 * (weightVector2 * (random_individual - population[i, :]))
                )

        # Boundary control
        populationNew = np.clip(populationNew, lb_arr, ub_arr)

        # Greedy selection
        for i in range(N):
            fitnessNew = fobj(populationNew[i, :])
            if fitnessNew < fitness[i]:
                population[i, :] = populationNew[i, :].copy()
                fitness[i] = fitnessNew

        idx = np.argsort(fitness)
        fitness = fitness[idx]
        population = population[idx, :]
        best_solutions = population[0:eliteSize, :].copy()

        best_fitness = float(fitness[0])
        best_solution = population[0, :].copy()
        fitness_history[t] = best_fitness
        t += 1

    return best_fitness, best_solution, fitness_history


ESC = esc


if __name__ == "__main__":
    from problem import run_experiment
    run_experiment(
        algo_name="ESC",
        algo_func=esc,
        full_name="Escape Optimization Algorithm (ESC)",
        n_agents=30,
        max_iter=1000,
        n_runs=5,
    )
