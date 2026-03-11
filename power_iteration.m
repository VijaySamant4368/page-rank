function [v, lambda, iters, history] = power_iteration(A, maxIter, tol, v0)

    if nargin < 2 || isempty(maxIter), maxIter = 1000; end
    if nargin < 3 || isempty(tol), tol = 1e-10; end
    
    n = size(A,1);
    if size(A,2) ~= n
        error('A must be square.');
    end
    
    if nargin < 4 || isempty(v0)
        v = ones(n,1) / n;
    else
        v = v0(:);
        if length(v) ~= n
            error('v0 must have length n.');
        end
        %Normalizetion
        if norm(v,1) == 0
            v = ones(n,1) / n;
        else
            v = v / norm(v,1);
        end
    end

    history.diff1 = zeros(maxIter,1);
    history.lambda = zeros(maxIter,1);

    lambda = NaN;
    for k = 1:maxIter
        w = A * v;

        if norm(w,1) == 0
            warning('A*v became the zero vector; returning current estimate.');
            iters = k;
            history.diff1 = history.diff1(1:k);
            history.lambda = history.lambda(1:k);
            return;
        end
        
        v_new = w / norm(w,1);

        lambda = (v_new' * (A * v_new)) / (v_new' * v_new);

        d = norm(v_new - v, 1);
        history.diff1(k) = d;
        history.lambda(k) = lambda;

        v = v_new;

        if d < tol
            iters = k;
            history.diff1 = history.diff1(1:k);
            history.lambda = history.lambda(1:k);
            %As a probability vector
            v = v / sum(v);
            return;
        end
    end

    iters = maxIter;
    history.diff1 = history.diff1(1:maxIter);
    history.lambda = history.lambda(1:maxIter);
    v = v / sum(v);
end
