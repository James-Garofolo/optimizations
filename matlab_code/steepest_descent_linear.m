function [sol,fval,grad] = steepest_descent_linear(Q, b, x0, Tol, maxiter)
% The steepest gradient descent algorithm for quadratic objective function:
% f(x) = 1/2 * x' Q x - b' * x 
% where Q is a symmetric positive definite matrix. 
% input: 
%     Q: a symmetric positive definite matrix
%     b: a vector
%     x0: vector of initial guess
%     Tol: scalar number of tolerance
%     maxiter: maximal number of iterations allowed (optional, not
%     provided, it is assigned to 10,000. 
% Output: 
%     sol: solution (the last iterate)
%     fval: a vector of values of the objective function at all iterations. 
%     grad: vector of norm of gradient at each iteration
N = length(x0); 

if nargin < 5
    maxiter = 10000; % default maximal number of iterations
end
fval = zeros(maxiter+1,1); 
grad = zeros(maxiter+1,1); 


iter = 0; STOP = 0; xNEW = x0; gradf = Q*x0 - b;  
fval(1) = 1/2*x0'*Q*x0 - b'*x0; 
grad(1) = norm(gradf);

if N == 2
    figure; plot(xNEW(1), xNEW(2),'*r'); 
end
while ~STOP
    xOLD = xNEW; 
    stepsize = (gradf'*gradf)/(gradf'*Q*gradf); 
    xNEW = xOLD - stepsize*gradf; % next iterate
    
    iter = iter + 1;

    if norm(xNEW - xOLD) < Tol
        STOP = 1;
    end
    gradf = Q*xNEW - b; 
    if norm(gradf) < Tol
        STOP = 1; 
    end
    fval(iter) = 1/2*xNEW'*Q*xNEW - b'*xNEW;
    grad(iter) = norm(gradf); 
    if iter >= maxiter 
        STOP = 1;
    end
    if N == 2
        hold on; plot(xNEW(1), xNEW(2),'*k'); hold off;
        pause(0.5); 

    end
end
sol = xNEW;
fval = fval(1:iter); 
grad = grad(1:iter); 

