function [final_score, iters, meta_data] = pagerank(adj, alpha, maxIter, tol, initial_vector)
    %initial_vector can be either row vector or col vector

    if nargin < 2 || isempty(alpha), alpha = 0.85; end
    if nargin < 3 || isempty(maxIter), maxIter = 1000; end
    if nargin < 4 || isempty(tol), tol = 1e-10; end

    n = size(adj,1);
    if size(adj,2) ~= n
        error('adj is not square.');
    end

    if nargin < 5 || isempty(initial_vector)
        v = ones(n,1);
    else
        %Initial vector is forced to be column vector
        v = initial_vector(:);
        if length(v) ~= n
            error('initial vector must have length n.');
        end
        if any(v < 0) || sum(v) == 0
            error('initial vector must be nonnegative and not all zeros.');
        end
    end
    processed_initial_vector = v / sum(v);

    %adj(i,j)=1 means edge i -> j.
    A = double(adj);
    A(A < 0) = 0;

    out_weight = sum(A, 2); % outdeg(i) = number of edges i -> *; i.e. sum of rows
    P = zeros(n,n);

    for i = 1:n
        if out_weight(i) > 0
            P(:, i) = (A(i, :)' / out_weight(i));
        else
            P(:, i) = ones(n,1) / n;
        end
    end

    Google_matrix = alpha * P + (1 - alpha) * (processed_initial_vector * ones(1,n));
    [p, lambda, iters, hist] = power_iteration(Google_matrix, maxIter, tol, processed_initial_vector);

    raw_score  = max(p, 0);
    final_score = raw_score  / sum(raw_score );

    meta_data.raw_score = raw_score ;
    meta_data.P = P;
    meta_data.G = Google_matrix;
    meta_data.lambda = lambda;
    meta_data.history = hist;
end
