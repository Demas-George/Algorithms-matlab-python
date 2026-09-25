function [Best_fit, Best_sol, Convergence_curve] = BTO(N, MaxIter, lb, ub, dim, fobj)
%BTO  Barrel Theory-Based Optimizer (BTO)
%
%   Authors:
%     Van Tai Tran, Quynh T.T. Nhu, Chitsutha Soomlek,
%     Punyaphol Horata, Khamron Sunat
%
%   Affiliations:
%     1) College of Computing, Khon Kaen University, Khon Kaen, Thailand
%     2) School of Computing and Information Technology, Eastern International University,
%        Ho Chi Minh, Vietnam
%
%   Paper:
%     "BTO: A Barrel Theory-Based Optimizer for Engineering Design Problems"
%     DOI: 10.46793/aeletters.2025.10.4.2
%


    Planks = initialization(N, dim, ub, lb);
    Convergence_curve = zeros(1, MaxIter);
    Fitness = zeros(1, N);
    epsilon = 1e-8;

    for i = 1:N
        Fitness(i) = fobj(Planks(i, :));
    end

    [Best_fit, best_idx] = min(Fitness);
    Best_sol = Planks(best_idx, :);

    for it = 1:MaxIter
        E = sqrt(it / MaxIter);
        normalized_fitness = barrel_AdjustmentProbability(Fitness, it, MaxIter, dim);

        K = max(1, ceil(10 * (1 - E)));
        [~, topK_idx] = mink(Fitness, K);

        for i = 1:N
            base  = Planks(i, :);
            jrand = randi(dim);

            F = 2 * (rand(1, dim) - 0.5) ...
                .* sin((2 * rand(1, dim) * pi * it / (MaxIter / 10))) ...
                .* (1 - it / MaxIter);

            elite_idx = topK_idx(randi(K));

            for j = 1:dim
                r1 = rand; r2 = rand;

                if r1 < normalized_fitness(i) || j == jrand
                    if r2 < E
                        sel = randi(N);
                        while sel == elite_idx, sel = randi(N); end

                        base(j) = Planks(sel, j) ...
                            + F(j) * (Planks(elite_idx, j) - Planks(sel, j)) ...
                            + F(j) * 0.25 * (1 - E) * randn * (ub(j) - lb(j));
                    else
                        d = Planks(elite_idx, j) - Planks(i, j);
                        direction = -sign(d + (d == 0) * (2 * (rand < E) - 1));
                        step = abs(F(j) * (ub(j) - lb(j)) * 0.3 ...
                            + (1 - abs(F(j))) * E * Planks(i, j) * randn ...
                            + epsilon);
                        base(j) = Planks(i, j) + direction * step;
                    end
                end
            end

            for j = 1:dim
                base(j) = handle_boundary(base(j), Planks(i, j), lb(j), ub(j));
            end

            fitTrial = fobj(base);
            if fitTrial < Fitness(i)
                Fitness(i) = fitTrial;
                Planks(i, :) = base;
                if fitTrial < Best_fit
                    Best_fit = fitTrial;
                    Best_sol = base;
                end
            end
        end

        Convergence_curve(it) = Best_fit;
    end
end

% ======================= Local functions =======================
function val = handle_boundary(val, original, lb, ub)
    if val < lb
        val = (original + lb) / 2;
    elseif val > ub
        val = (original + ub) / 2;
    end
end

function fitness = barrel_AdjustmentProbability(obj_vals, it, MaxIter, D)
    min_val = min(obj_vals);
    max_val = max(obj_vals);
    if max_val == min_val
        norm_fitness = zeros(size(obj_vals));
    else
        norm_fitness = (obj_vals - min_val) / (max_val - min_val);
    end
    progress = it / MaxIter;
    lower = 0.3*(1 - sqrt(progress));
    upper_end = 0.3 * (1 - D / 100);
    upper = 1.0 - (1.0 - upper_end) * sqrt(progress);
    fitness = lower + (upper - lower) * norm_fitness;
end

function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    for i = 1:dim
        Positions(:, i) = rand(SearchAgents_no, 1) .* (ub(i) - lb(i)) + lb(i);
    end
end
