function [X]= seidel(A,b)
 
n = length(A);
 
X0 = zeros(n,1);  
 
L = zeros(n,n);
U = zeros(n,n);
D = zeros(n,n);
 
for i = 1:n
	for j = 1:i-1
        L(i,j) = A(i,j);
	end
	for j = i+1:n
        U(i,j) = A(i,j);
	end
	D(i,i) = A(i,i);
end
 
disp('L ='); disp(L)
disp('U ='); disp(U)
disp('D ='); disp(D)
 
max_iteration = 50;
tolerance = 5e-6;
history = zeros(n, max_iteration);
 
X = X0;
 
for iter = 1:max_iteration
 
	X_old = X;  
	for k = 1:n
 
    	sumL = L(k,:) * X;   	
    	sumU = U(k,:) * X_old;  
 
    	X(k) = (b(k) - sumL - sumU) / D(k,k);
 
	end
    history(:,iter) = X;
	if norm(X - X_old, inf) < tolerance
        fprintf('Converged in %d iterations\n', iter);
    	break;
	end
 
end
 
disp('Solution X =');
disp(X)
figure
plot(1:iter, history(:,1:iter)', 'LineWidth', 2)
xlabel('Iteration')
ylabel('Solution Value')
title('Gauss-Seidel Convergence')
grid on
