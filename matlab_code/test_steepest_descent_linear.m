% test steepest descent algorithm for quadratic problem: 
N  = 100; 
Q = eye(N); Q(1,1) = 10; 
b = zeros(N,1);
x0 = 10*ones(N,1); 
Tol = 1e-8; 
maxiter = 1000; 

[x,fval,grad] = steepest_descent_linear(Q, b, x0, Tol, maxiter);
disp(x)
figure; plot(0:length(fval)-1,fval); title('objective function'); xlabel('iteration k'); ylabel('f(x_k)');
figure; plot(0:length(fval)-1,grad); title('norm of objective function gradient'); xlabel('iteration k');  