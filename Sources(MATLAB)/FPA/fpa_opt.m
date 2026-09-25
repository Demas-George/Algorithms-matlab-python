function [best, fmin, Convergence_curve] = fpa_opt(n, N_iter, lb, ub, dim, fobj, p)
if nargin < 7, p = 0.8; end
if isscalar(lb), lb = lb * ones(1, dim); end
if isscalar(ub), ub = ub * ones(1, dim); end

Sol = zeros(n, dim);
Fitness = zeros(n, 1);
for i = 1:n
    Sol(i, :) = lb + (ub - lb) .* rand(1, dim);
    Fitness(i) = fobj(Sol(i, :));
end

[fmin, I] = min(Fitness);
best = Sol(I, :);
S = Sol;
Convergence_curve = zeros(1, N_iter);

for t = 1:N_iter
    for i = 1:n
        if rand > p
            beta = 1.5;
            sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
            u = randn(1, dim) * sigma;
            v = randn(1, dim);
            step = u ./ (abs(v).^(1/beta));
            L = 0.01 * step;
            dS = L .* (Sol(i, :) - best);
            S(i, :) = Sol(i, :) + dS;
        else
            epsilon = rand;
            JK = randperm(n);
            S(i, :) = S(i, :) + epsilon * (Sol(JK(1), :) - Sol(JK(2), :));
        end
        S(i, :) = max(S(i, :), lb);
        S(i, :) = min(S(i, :), ub);
        Fnew = fobj(S(i, :));
        if Fnew <= Fitness(i)
            Sol(i, :) = S(i, :);
            Fitness(i) = Fnew;
        end
        if Fnew <= fmin
            best = S(i, :);
            fmin = Fnew;
        end
    end
    Convergence_curve(t) = fmin;
end
end
