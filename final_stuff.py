import numpy as np
from opt_algorithms import *

def exam_f(x):
    return np.linalg.norm(x[1:] - x[:-1])**2 + ((x[0]**2)-1)**2

def nerdy_f(x):
    f = ((x[0]**2)-1)**2
    for i in range(x.size-1):
        f += (x[i+1]-x[i])**2

    return f

def exam_grad(x):
    grad = np.zeros_like(x)
    grad[0] = 4*x[0]*((x[0]**2)-1) - 2*(x[1]-x[0])
    grad[1:-1] = 2*(x[1:-1]-x[:-2]) - 2*(x[2:]-x[1:-1]) 
    grad[-1] = 2*(x[-1]-x[-2])
    return grad

def exam_hess(x):
    hess = np.zeros((x.size, x.size))
    hess[0,0] = 12.0*(x[0]**2) - 2.0
    for a in range(x.size-1):
        hess[a+1,a] = -2.0
        hess[a+1,a+1] = 4.0
        hess[a, a+1] = -2.0

    hess[-1,-1] = 2.0

    return hess

x0s = [np.ones(10)*10.0, np.ones(10)*-20.0]

if __name__ == "__main__":
    """for x0 in x0s:
        sd = steepest_descent_backtrack(exam_f, exam_grad, x0, rho=0.2, norm_grad_tol=1e-6, norm_step_tol=1e-6, 
                                        func_change_tol=None, print_every=10000)
        xs, fs, b = sd.solve()

        print(f"---sd results---\nnum iter: {b}\nx: {xs[-1]}\nf: {fs[-1]}\ngrad: {exam_grad(xs[-1])}")

        n = newton(exam_f, exam_grad, exam_hess, x0, norm_grad_tol=1e-6, norm_step_tol=1e-6)
        xs, fs, b = n.solve()

        print(f"---newton results---\nnum iter: {b}\nx: {xs[-1]}\nf: {fs[-1]}\ngrad: {exam_grad(xs[-1])}")

        qn = BFGS(exam_f, exam_grad, x0, r=0.2, norm_grad_tol=1e-6, norm_step_tol=1e-6)
        xs, fs, b = qn.solve()

        print(f"---q newton results---\nnum iter: {b}\nx: {xs[-1]}\nf: {fs[-1]}\ngrad: {exam_grad(xs[-1])}")"""
    
    x0 = np.ones(100000)*10    

    lmqn = L_BFGS(exam_f, exam_grad, x0, 10, r=0.2, norm_grad_tol=1e-6, norm_step_tol=1e-6, print_every=5000)
    xs, fs, b = lmqn.solve()

    print(f"---l bfgs results---\nnum iter: {b}\nx: {xs[-1]}\nf: {fs[-1]}\ngrad: {exam_grad(xs[-1])}")

