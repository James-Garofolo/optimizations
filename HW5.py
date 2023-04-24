import numpy as np
import matplotlib.pyplot as plt
import math
from opt_algorithms import steepest_descent_backtrack


gt_x = np.arange(0, 1.1, 0.1)
gt_y = np.array([2.7179, 2.6409, 2.4260,  2.1291, 1.8763, 1.4671, 0.9689, 0.1223, -0.7330, -1.4696, -1.7403])

plot_x = np.arange(-1, 2, 0.001)
actual_y = 1 + 2*np.cos(math.pi*plot_x) - 0.5*np.cos(2*math.pi*plot_x) + 0.2*np.cos(3*math.pi*plot_x)

def model_f(xs, ps):
    model_y = np.zeros_like(xs)
    for a, p in enumerate(ps):
        model_y += p*np.cos(a*math.pi*xs)

    return model_y

def model_j(ps):
    model_y = model_f(gt_x, ps)
    return (np.linalg.norm(model_y-gt_y)**2)/2

def model_grad(ps):
    grad = np.zeros_like(ps)
    n = np.arange(0, ps.size)
    for a, x in enumerate(gt_x):
        grad += np.cos(n*math.pi*x)*(np.sum(ps*np.cos(n*math.pi*x))-gt_y[a])

    return grad

if __name__ == "__main__":
    Ns = [5, 7, 9, 11]

    for N in Ns:
        p0 = np.zeros(N)
        sd = steepest_descent_backtrack(model_j, model_grad, p0, norm_grad_tol=1e-6 ,func_change_tol=None)
        ps, fs, b = sd.solve()
        model_y = model_f(plot_x, ps[-1])
        eval_mse = (np.linalg.norm(model_y-actual_y)**2)/2
        print(f"----RESULTS----\nN = {N}, \nparameters = {ps[-1]}, \nSampled MSE = {fs[-1]},\
               \nactual MSE = {eval_mse}, \nnum iterations = {b}")

        plt.scatter(gt_x, gt_y, label="sampled data")
        plt.plot(plot_x, actual_y, label="ground truth")
        plt.plot(plot_x, model_y, label="model")
        plt.legend(loc="upper right")
        plt.show()