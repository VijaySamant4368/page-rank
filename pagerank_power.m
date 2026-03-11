function [r, info] = pagerank_power(G, maxIter, tol, alpha)

    if nargin < 4 || isempty(alpha), alpha = 0.85; end
    n = [];

    % ---- build adjacency matrix if edge list ----
    if ~ismatrix(G)
        error('G must be an adjacency matrix or an edge list (m x 2).');
    end

    if size(G,2) == 2 && size(G,1) >= 1 && ~isequal(size(G,1), size(G,2))
        % Treat as edge list: [from to]
        edges = G;
        if isempty(n)
            n = max(edges(:));
        end
        A = sparse(edges(:,1), edges(:,2), 1, n, n);
    else
        % Treat as adjacency matrix
        A = sparse(G);
        n = size(A,1);
        if size(A,2) ~= n
            error('Adjacency matrix must be square (n x n).');
        end
    end

    % ---- validate alpha/tol ----
    if ~(isscalar(alpha) && alpha >= 0 && alpha <= 1)
        disp(alpha)
        error('alpha must be a scalar in (0,1).');
    end
    if ~(isscalar(tol) && tol > 0)
        error('tol must be a positive scalar.');
    end

    % ---- out-degree (row sum if A(i,j)=i->j) ----
    outDeg = full(sum(A, 2)); % n x 1
    dangling = (outDeg == 0);

    % ---- initialize rank ----
    r = ones(n,1) / n;

    % teleportation vector (uniform)
    v = ones(n,1) / n;

    % choose iteration mode
    fixedIters = (maxIter >= 0);
    if fixedIters
        itLimit = maxIter;
    else
        itLimit = 10^7; % very high cap for safety
    end

    converged = false;
    lastErr = inf;

    for k = 1:itLimit
        r_old = r;

        % ---- power iteration step ----
        % Distribute rank through links:
        % For non-dangling nodes i, distribute r(i)/outDeg(i) to its out-neighbors.
        contrib = r_old ./ max(outDeg, 1);         % avoid divide-by-zero
        contrib(dangling) = 0;                      % dangling handled separately

        % r_link = A' * contrib because A(i,j)=i->j contributes to j
        r_link = A' * contrib;

        % dangling mass redistributed uniformly
        dmass = sum(r_old(dangling));

        % PageRank update
        r = alpha * (r_link + dmass * v) + (1 - alpha) * v;

        % normalize for numerical stability
        r = r / sum(r);

        % ---- error check (L1 norm) ----
        lastErr = norm(r - r_old, 1);

        if ~fixedIters && lastErr < tol
            converged = true;
            break;
        end
    end

    info = struct();
    info.iters = k;
    info.converged = converged || fixedIters;
    info.lastError = lastErr;
end


