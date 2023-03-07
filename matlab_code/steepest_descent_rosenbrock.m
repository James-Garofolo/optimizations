% Steepest descent method for Rosenbrock function: 
% Note: the global minimizer is at (1,1). 

x0 = [2; 2];
Tol = 1e-6;
stepsize = 1e-3; 

format long 
% for plotting: 

x = linspace(-3,3,61); y = linspace(-3,3,61); 
[X,Y]  = meshgrid(x,y);
f = rosenbrock(X,Y);
figure(1); surfc(X,Y,f); 
hold on; plot3(1, 1, rosenbr([1,1]),'g*'); hold off; % the solution

% 
% disp('Newton method:'); 
% [xN,fvalN] = Newton_method(@rosenbr, x0,Tol); 

% The steepest gradient descent algorithm: 
iter = 0; STOP = 0; xNEW = x0;  
while ~STOP
    xOLD = xNEW; 
    [f,gradf] = objfun(xOLD,scaling);
    xNEW = xOLD - stepsize*gradf;
    if norm(xNEW - xOLD) < Tol
        STOP = 1;
    end
    fprintf('%s%20.15f\n','objective function = ',f); 
    hold on; plot3(xNEW(1), xNEW(2), f,'r*'); hold off;
    iter = iter + 1 
    pause(0.3);
end





% objective function: 
function [f,gradf,Hess] = rosenbr(x)
a = 100; 
f = (1-x(1)).^2 + a*(x(2)-x(1)^2)^2;

gradf = [2*(x(1)-1) + a*4*(x(1)^2 - x(2)); 
         a*2*(x(2) - x(1)^2)];
     
Hess = [2 + 4*a*(3*x(1)^3-x(2)), -4*a*x(1); 
        -4*a*x(1),               2*a];

end