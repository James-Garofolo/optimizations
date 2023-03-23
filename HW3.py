import numpy as np
import matplotlib.pyplot as plt
from HW2_P8 import steepest_descent

ros_a = 100

def ros_f(x, a=ros_a):
    return a*(x[1]-x[0]**2)**2 + (1-x[0])**2

def ros_grad(x, a=ros_a):
    g = np.zeros_like(x)
    g[0] = -2*(1-x[0]) -4*a*(x[1]-x[0]**2)*x[0]
    g[1] = 2*a*(x[1]-x[0]**2)

    return g

def ros_hessian(x,a=ros_a):
    h = np.zeros((x.size, x.size))
    h[0,0] = -4*a*x[1] + 4*3*a*(x[0]**2) + 2
    h[0,1] = -4*a*x[0]
    h[1,0] = -4*a*x[0]
    h[1,1] = 2*a

    return h

class newton:
    def __init__(self, function, gradient, hessian, initial, norm_grad_tol=1e-5, iter_lim=None) -> None:
        self.f = function
        self.g = gradient
        self.h = hessian
        self.x = initial
        self.ngt = norm_grad_tol
        self.iter_lim = iter_lim

    def step(self, gradient=None, hessian=None):
        if gradient is None:
            grad = self.g(self.x)
        else:
            grad = gradient

        if hessian is None:
            hess = self.h(self.x)
        else:
            hess = hessian

        self.x = self.x - np.matmul(np.linalg.inv(hess),grad)

        return self.x, self.f(self.x), self.g(self.x)

    def solve(self):
        grad = self.g(self.x)

        xs = []
        fs = []
        a = 0
        while np.linalg.norm(grad,1) > self.ngt:
            x, func, grad = self.step(grad)
            xs.append(x)
            fs.append(func)
            a+=1
            if not (self.iter_lim is None):
                if a > self.iter_lim:
                    print(np.linalg.norm(grad,1))
                    break

        return np.array(xs), np.array(fs), a
    

class steepest_descent_backtrack:
    def __init__(self, function, gradient, initial, c=1e-4, rho=0.9, norm_grad_tol=1e-5, iter_lim=None) -> None:
        self.f = function
        self.g = gradient
        self.c = c
        self.rho = rho
        self.x = initial
        self.ngt = norm_grad_tol
        self.iter_lim = iter_lim

    def step(self, f_val=None, gradient=None):
        if gradient is None:
            grad = self.g(self.x)
        else:
            grad = gradient

        if f_val is None:
            func = self.f(self.x)
        else:
            func = f_val

        a = 1
        while True:
            xnew = self.x - a*grad
            if self.f(xnew) <= (func-self.c*a*np.matmul(grad,grad)):
                break
            else:
                a *= self.rho

        self.x = xnew

        return self.x, self.f(self.x), self.g(self.x)

    def solve(self):
        func = self.f(self.x)
        grad = self.g(self.x)

        xs = []
        fs = []
        b = 0
        while np.linalg.norm(grad,1) > self.ngt:
            x, func, grad = self.step(func, grad)
            xs.append(x)
            fs.append(func)
            b+=1
            if not (self.iter_lim is None):
                if b > self.iter_lim:
                    print(np.linalg.norm(grad,1))
                    break

        return np.array(xs), np.array(fs), b
    

if __name__ == "__main__":
    x0 = np.array([-1.2,1])
    
    print("initial: ", x0)
    sd = steepest_descent_backtrack(ros_f, ros_grad, x0,rho=0.1, iter_lim=1e7)
    sd_no_backtrack = steepest_descent(ros_f, ros_grad, 0.000001, x0, 1e7)
    n = newton(ros_f, ros_grad, ros_hessian, x0, iter_lim=1e6)
    xs, fs, a = n.solve()
    delta_x = np.linalg.norm(xs[1:]-xs[:-1], axis=1)
    plt.plot(delta_x, label='newtons')
    print(f"newton's: \n solution: {xs[-1]}, \n num iterations {a}")
    xs, fs, a = sd.solve()
    delta_x = np.linalg.norm(xs[1:]-xs[:-1], axis=1)
    #plt.plot(delta_x, label='steepest descent')
    print(f"steepest descent: \n solution: {xs[-1]}, \n num iterations {a}")
    
    xs, fs, a = sd_no_backtrack.solve()
    #delta_x = np.linalg.norm(xs[1:]-xs[:-1], axis=1)
    #plt.plot(delta_x, label='steepest descent')
    print(f"steepest descent: \n solution: {xs[-1]}, \n num iterations {a}")

    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('l2norm of change in x')
    plt.show()