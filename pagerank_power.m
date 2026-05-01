function [rank_vec, info, rank_vec_hist] = pagerank_power(A, maxIter, tol, alpha)
% A(i,j) = 1 means i -> j (is later transposed)

    if nargin < 4 || isempty(alpha)
        alpha = 0.85;
    end
    
    n = size(A,1);

    if size(A,2) ~= n
        error('Adjacency matrix must be square (n x n).');
    end

    if ~(isscalar(alpha) && alpha >= 0 && alpha <= 1)
        error('alpha must be a scalar in [0,1].');
    end

    if ~(isscalar(tol) && tol > 0)
        error('tol must be a positive scalar.');
    end

    outDeg = full(sum(A,2));

    M = zeros(n,n);

    for i = 1:n
        if outDeg(i) == 0
            M(i,:) = 1/n;
        else
            M(i,:) = A(i,:) / outDeg(i);
        end
    end

    M = M';

    rank_vec = ones(n,1) / n;
    teleport = ones(n,1) / n;

    fixedIters = (maxIter >= 0);
    if fixedIters
        itLimit = maxIter;
    else
        itLimit = 1e7;
    end

    rank_vec_hist = rank_vec;

    converged = false;
    lastErr = inf;

    for k = 1:itLimit

        r_old = rank_vec;

        rank_vec = alpha * (M * r_old) + (1 - alpha) * teleport;

        rank_vec = rank_vec / sum(rank_vec);
        sum(rank_vec)

        rank_vec_hist(:,end+1) = rank_vec;

        lastErr = norm(rank_vec - r_old,1);

        if ~fixedIters && lastErr < tol
            converged = true;
            break
        end

    end

    info = struct();
    info.iters = k;
    info.converged = converged || fixedIters;
    info.lastError = lastErr;

end