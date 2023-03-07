function [solution,fval,xnormval] = Newton_method(objfun, InitGuess,Tol,MaxIter)
% solve an optimization problem using Newton's method
% what you have to provide as input: 
%     1. objfun: this is a function handle whose output include the value of the objective function,
%     the gradient, and the Hessian at a given point x. 
%     2. InitGuess: this is a COLUMN vector of an initial guess of the solution. 
%     3. Tol: tolerance for stopping
%     4. MaxIter: maximum number of iterations, provide this number to
%     avoid the case when the algorithm does not converge and run "forever"
% Output: 
%     solution: Approximate minimizer found by Newton's method
%     fval: values of the objective function at all iterations. 
%     xnormval: vector of ||x_{k+1} - x_{k}|| for checking the convergence
%     rate

iter = 0; 
STOP = 0; 
fval = zeros(MaxIter,1); % to store the values of the objective function
xnormval = zeros(MaxIter,1); 

% evaluate the function at the initial guess: 
xOLD = InitGuess; 
[f,gradf,Hess] = objfun(xOLD);
fprintf('%s%d%s%10.5f\n','Iteration = ',iter,', objective function = ',f); 
hold on; plot3(xOLD(1), xOLD(2), f,'g*'); hold off; % plot iterates (only for 2 variables) 

if norm(gradf) < Tol 
    STOP = 1; 
end

% iteration: 
fval(1) = f; 

while ~STOP
    iter = iter + 1;
    xNEW = xOLD - Hess\gradf; % Hess\gradf = inv(Hess)*gradf but is more efficient

    [f,gradf,Hess] = objfun(xNEW);
    hold on; plot3(xNEW(1), xNEW(2), f,'g*'); hold off; % plot iterates (only for 2 variables) 
 %   pause; 
    soldiff = norm(xNEW - xOLD); 
    if (soldiff < Tol) || (norm(gradf) < Tol) || (iter >=MaxIter)
        STOP = 1;
    end
    fprintf('%s%d%s%10.8f%s%10.8f\n','Iter=',iter,', obj fun=',f, ', x_{k+1}-x_k=',soldiff); 

    fval(iter+1) = f;
    xnormval(iter) = soldiff; 
    xOLD = xNEW; % for next iteration
end
fval = fval(1:iter+1); 
xnormval = xnormval(1:iter); 

solution = xOLD; 
