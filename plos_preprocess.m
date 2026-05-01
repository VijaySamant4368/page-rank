A = importdata('dataset/Plos/USAir.txt');
d = configureDictionary("double", "double");
N = 4941;
MATRIX = zeros(N, N);
mins = min(A, [], 'all');
maxs = max(A, [], 'all');
N = maxs-mins+1;
disp([mins, maxs, N])
for i=1:length(A)
    node_from = A(i, 1);
    node_to = A(i, 2);
    MATRIX(node_from, node_to) = MATRIX(node_from, node_to) + 1;
end

writematrix(MATRIX, "adj_M/plos_matrix.txt");
disp("Wrote the matrix");