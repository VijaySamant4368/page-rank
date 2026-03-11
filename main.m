A = importdata('karate.txt');
A = A;
maxIter = -1;      % negative => iterate until converged
tol = 1e-10;
alpha = 0.95;

%adj(i,j)=1 means edge i -> j.
[r, info] = pagerank_power(A, maxIter, tol, alpha);

disp('PageRank:');

[sorted_r, org_indx] = sort(r, "descend");

disp(info);
disp(sorted_r(1:10));
disp(org_indx(1:10));
