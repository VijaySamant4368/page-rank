A = importdata('dataset/cora/cora.cites');
d = configureDictionary("double", "double");
N = 2708;
MATRIX = zeros(N, N);
index_count = 1;                            %Index starts with 1 in MATLAB
for i=1:length(A)
    cited = A(i, 1);
    citer = A(i, 2);
    cited_index_exist = isKey(d, cited);
    if cited_index_exist
       cited_index = d(cited);
    else
        d(cited) = index_count;
        cited_index = index_count;
        index_count=index_count+1;
    end
    citer_index_exist = isKey(d, citer);
    if citer_index_exist
       citer_index = d(citer);
    else
        d(citer) = index_count;
        citer_index = index_count;
        index_count=index_count+1;
    end
    MATRIX(cited_index, citer_index) = 1;
end

writematrix(MATRIX, "MATRIX");
writedictionary(d, "Mapping.json");