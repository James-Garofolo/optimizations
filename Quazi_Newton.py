import numpy as np

class q_newton:
    def __init__(self, function, gradient, initial, c=1e-4, r=0.9, beta=1, norm_grad_tol=1e-5, iter_lim=None) -> None:
        self.f = function
        self.g = gradient
        self.c = c
        self.r = r
        self.x = initial
        self.h_mat = beta * np.identity(self.x.size)
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

        p = np.matmul(self.h_mat,-grad)

        a = 1
        while True:
            xnew = self.x + a*p
            fnew = self.f(xnew)
            if fnew <= (func-self.c*a*np.matmul(grad,grad)):
                break
            else:
                a *= self.r

        s = xnew - self.x
        y = self.g(xnew) - grad
        self.x = xnew
        self.h_mat = self.h(s, y, self.h_mat)

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
    
    def h(self, sk, yk, hk):
        rho = 1/np.matmul(sk,yk)
        return np.matmul((np.identity(sk.size)-rho*np.outer(sk,yk)), \
                        np.matmul(hk, (np.identity(sk.size)-rho*np.outer(yk,sk))))\
                        +(np.identity(sk.size)-rho*np.outer(sk,sk))