function [f,gradf,Hess] = general_rosenbrock(x,a)
% general Rosenbrock function 
% f(x) = \sum\limts_{i= 1}^{d-1} [a*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
% Input: x is a column vector
%        a: a scalar number. If not provided, the default value is a = 100;
% output: f is the function value (scalar)
% gradf: gradient of f (column vector of the same size as x
% Hess: Hessian matrix

if nargin < 2
    a = 100; % default constant a is 100. 
end

f = 0;
d = length(x); % number of components of x
gradf = zeros(d,1); 
Hess = zeros(d,d); 

for i = 1:d-1
    r1 = x(i+1) - x(i)^2; 
    r2 = x(i) - 1; 
    
    f = f + a*r1^2 + r2^2;
    gradf(i) = gradf(i) + 2*a*r1*(-2*x(i)) + 2*r2;
    gradf(i+1) = gradf(i+1) + 2*r1*a;
    Hess(i,i) = Hess(i,i) - 4*a*(r1 -2*x(i)^2) + 2; 
    Hess(i,i+1) = Hess(i,i+1) - 4*x(i)*a; 
    Hess(i+1,i) = Hess(i,i+1); 
    Hess(i+1,i+1) = Hess(i+1,i+1) + 2*a; 
end



    