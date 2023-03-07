% Steepest descent method for Rosenbrock function: 
% Note: the global minimizer is at (1,1). 

x0 = [8; 20];
Tol = 1e-6;
stepsize = 1e-2; 
scaling = 100; 

format long 
% for plotting: 

x = linspace(-3,3,61); y = linspace(-3,3,61); 
[X,Y]  = meshgrid(x,y);
f = X.^2 + scaling*Y.^2; 
figure(1); surfc(X,Y,f); 
hold on; plot3(1, 1, objfun([0,0],scaling),'g*'); hold off;

%  Newton's method: 
iter = 0; xNEW = x0; STOP = 0; 
disp('Newton method'); 

while ~STOP
    xOLD = xNEW; 
    [f,gradf,Hess] = objfun(xOLD,scaling);
    xNEW = xOLD - inv(Hess)*gradf;
    if norm(xNEW - xOLD) < Tol
        STOP = 1;
    end
    fprintf('%s%20.15f\n','objective function = ',f); 
    hold on; plot3(xNEW(1), xNEW(2), f,'g*'); hold off;
    iter= iter + 1 
    pause(0.3);
end
% 
% pause; 
% 
% % The steepest gradient descent algorithm: 
% iter = 0; STOP = 0; xNEW = x0;  
% while ~STOP
%     xOLD = xNEW; 
%     [f,gradf] = objfun(xOLD,scaling);
%     xNEW = xOLD - stepsize*gradf;
%     if norm(xNEW - xOLD) < Tol
%         STOP = 1;
%     end
%     fprintf('%s%20.15f\n','objective function = ',f); 
%     hold on; plot3(xNEW(1), xNEW(2), f,'r*'); hold off;
%     iter = iter + 1 
%     pause(0.3);
% end

% 
% % Nesterov algorithm: 
% [sol, fval]= Nesterov_algorithm(@(x)(objfun(x,scaling)), x0, 2.5, 0.9, Tol, 100);
% figure; plot(fval); 
% sol
% 
% [sol, fval, betaArray, muArray]= modified_Nesterov_algorithm(@(x)objfun(x,scaling), x0, 1.1, Tol, 100);
% figure; plot(fval); 
% sol



% objective function: 
function [f,gradf,Hess] = objfun(x,a)
f = x(1)^2 + a*x(2)^2;

gradf = [2*x(1); 2*a*x(2)];
Hess = [2, 0; 0, 2*a];

end



function [f,gradf] = objfun2(x)
f = x(1)^2 + a*x(2)^2;

gradf = [2*x(1); 2*a*x(2)];
Hess = [2, 0; 0, 2*a];

end



