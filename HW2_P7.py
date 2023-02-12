import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

ros_a = 1

def f(x, a=ros_a):
    return a*(x[1]-x[0]**2)**2 + (1-x[0])**2

def grad(x, a=ros_a):
    g = np.zeros_like(x)
    g[0] = -2*(1-x[0]) -4*a*(x[1]-x[0]**2)*x[0]
    g[1] = 2*a*(x[1]-x[0]**2)

    return g

def hessian(x,a=ros_a):
    h = np.zeros((x.size, x.size))
    h[0,0] = -4*a*x[1] + 4*3*a*x[0] - 2
    h[0,1] = -4*a*x[0]
    h[1,0] = -4*a*x[0]
    h[1,1] = 2*a

    return h


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


if __name__ == "__main__":
    x0 = np.array([2,2])
    sd_rosenbrock = steepest_descent(f, grad, 1e-4, x0, iter_lim=1e6)

    xs, fs, a = sd_rosenbrock.solve()
    #print(xs[-1], fs[-1], a)
    
    
    x0s = np.linspace(0, 2.5, 2000)
    x1s = np.linspace(0, 2.5, 2000)
    fvals = np.zeros((x0s.size,x1s.size))

    for a, x_0 in enumerate(x0s):
        for b, x_1 in enumerate(x1s):
            fvals[a,b] = f(np.array([x_0,x_1]))

    x0s,x1s = np.meshgrid(x0s,x1s)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1s, x0s, fvals,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha=0.6)
    
    ax.scatter(xs[:,0],xs[:,1],fs,color='orange',s=3)
    endpoint_x0s = [x0[0],xs[-1,0]]
    endpoint_x1s = [x0[1],xs[-1,1]]
    endpoint_fs = [f(x0),fs[-1]]
    ax.scatter(endpoint_x0s,endpoint_x1s,endpoint_fs,color='green',s=10)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    dx = xs[1:]-xs[:-1]
    dx = np.linalg.norm(dx, 1, 1)
    #print(xs.shape, dx.shape)

    plt.plot(dx, label='exp1')

    x0 = np.array([2,2])
    sd_rosenbrock = steepest_descent(f, grad, 2e-4, x0, iter_lim=1e6)

    xs, fs, a = sd_rosenbrock.solve()
    #print(xs[-1], fs[-1], a)
    dx = xs[1:]-xs[:-1]
    dx = np.linalg.norm(dx, 1, 1)
    print(xs.shape, dx.shape)

    plt.plot(dx, label='exp2')

    x0 = np.array([2.5,0.5])
    sd_rosenbrock = steepest_descent(f, grad, 5e-5, x0, iter_lim=1e6)

    xs, fs, a = sd_rosenbrock.solve()
    #print(xs[-1], fs[-1], a)
    dx = xs[1:]-xs[:-1]
    dx = np.linalg.norm(dx, 1, 1)
    print(xs.shape, dx.shape)

    plt.plot(dx, label='exp3')

    x0 = np.array([2.5,0.5])
    sd_rosenbrock = steepest_descent(f, grad, 7e-5, x0, iter_lim=1e6)

    xs, fs, a = sd_rosenbrock.solve()
    #print(xs[-1], fs[-1], a)
    dx = xs[1:]-xs[:-1]
    dx = np.linalg.norm(dx, 1, 1)
    print(xs.shape, dx.shape)

    plt.plot(dx, label='exp4')
    plt.ylabel('||x_{k+1}-x_k||')
    plt.xlabel('iteration number')
    plt.legend()
    plt.show()