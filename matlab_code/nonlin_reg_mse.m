function [f,gradf] = nonlin_reg_mse(p)

gt_x = [0:0.1:1]'; % sample points
% observation values
gt_y = [2.7179, 2.6409, 2.4260,  2.1291, 1.8763, 1.4671, 0.9689, 0.1223, -0.7330, -1.4696, -1.7403]';
m = length(gt_x); % number of components of x
n = length(p); % number of components of p 

model_y = zeros(m, 1);
n_vector = (0:n-1)';

for a = 1:n
    model_y = model_y + p(a).*cos((a-1).*pi.*gt_x);
end

f = (norm(model_y-gt_y)^2)/2;

gradf = zeros(n, 1);
for a = 1:m
    gradf = gradf + cos(pi*gt_x(a)*n_vector) .* (sum(p.*cos(pi*gt_x(a)*n_vector))-gt_y(a));
end




    