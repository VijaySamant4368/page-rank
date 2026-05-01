A = importdata('karate.txt');
maxIter = -1;      % negative => iterate until converged
tol = 1e-10;
alpha = 0.85;

% adj(i,j)=1 means edge i -> j.
[r, info, r_hist] = pagerank_power(A, maxIter, tol, alpha);

disp('PageRank:');

[sorted_r, org_indx] = sort(r, "descend");

disp(info);
disp(sorted_r(1:10));
disp(org_indx(1:10));

disp('First 5 iteration vectors:');
disp(r_hist(:,1:min(5,size(r_hist,2))));

disp('Last vector in history:');
disp(r_hist(:,end));

for p = 1:5
    figure;
    bar(r_hist(:, p));
    xlabel('Node index');
    ylabel('PageRank value');
    ylim([0 0.15]);
    title(sprintf('PageRank Vector before iteration %d', p));
    grid on;
    exportgraphics(gcf, [sprintf('graph/iter_%d', p) '.png'], 'Resolution', 300);
end
figure;
bar(r);
xlabel('Node index');
ylabel('PageRank value');
ylim([0 0.15]);
title(sprintf('Final PageRank vector'));
grid on;
exportgraphics(gcf, [sprintf('graph/final_vector') '.png'], 'Resolution', 300);