import numpy as np
import matplotlib.pyplot as plt

class steepest_descent:
    def __init__(self, function, gradient, step_size, initial, norm_grad_tol=1e-5, iter_lim=None) -> None:
        self.f = function
        self.g = gradient
        self.s = step_size
        self.x = initial
        self.ngt = norm_grad_tol
        self.iter_lim = iter_lim

    def step(self, gradient=None):
        if gradient is None:
            grad = self.g(self.x)
        else:
            grad = gradient
        self.x = self.x - (self.s*grad/np.linalg.norm(grad))

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
    

def f(x, a=100):
    return x[0]**2 + (a*(x[1]**2))

def grad(x, a=100):
    g = np.zeros_like(x)
    g[0] = 2*x[0]
    g[1] = 2*a*x[1]
    return g

def hess(x, a=100):
    h = np.zeros((x.size,x.size))
    h[0,0] = 2
    h[0,1] = 0
    h[1,0] = 0
    h[1,1] = 2*a
    return h


if __name__ == "__main__":
    x0s = np.array([[2,2],[-5,5],[7,8]])

    for x0 in x0s:
        print("initial: ", x0)
        sd = steepest_descent(f, grad, 1e-5, x0, iter_lim=1e7)
        n = newton(f, grad, hess, x0, iter_lim=1e6)
        xs, fs, a = n.solve()
        print(f"newton's: \n solution: {xs[-1]}, \n num iterations {a}")
        xs, fs, a = sd.solve()
        print(f"steepest descent: \n solution: {xs[-1]}, \n num iterations {a}")

    