import numpy as np



def generate_finite_diff_grad(f, delta_x=1e-7, second_order=False):
    """
    generates an approximated gradient function for an arbitrary objective function using finite differences

    inputs:
        f: function to compute the gradient of, must take in a numpy vector and return a scalar
        delta_x: step size used to compute the finite differences
        second_order: boolean, decides whether to generate a first or second order divided difference approximator

    returns: 
        approximated gradient function that returns a vector of the same size that it takes in
    """
    if second_order:
        def ddg_2nd(x):
            grad = np.empty_like(x)

            for a,_ in enumerate(x):
                xnew = np.copy(x)
                xnew[a] += delta_x
                yp = f(xnew)

                xnew[a] -= 2*delta_x
                ym = f(xnew)

                grad[a] = (yp - ym)/(2*delta_x)

            return grad
        
        return ddg_2nd
        
    else:
        def ddg_1st(x):
            y = f(x)
            grad = np.empty_like(x)

            for a,_ in enumerate(x):
                xnew = np.copy(x)
                xnew[a] += delta_x
                yp = f(xnew)

                grad[a] = (yp - y)/delta_x

            return grad
        
        return ddg_1st

class newton:
    def __init__(self, function, gradient, hessian, initial, norm_grad_tol=1e-5, iter_lim=None, print_every=100) -> None:
        """
        class that implements Newton's algorithm

        inputs:
            function: objective function to be optimized, must take in a numpy vector and return a scalar 
            gradient: gradient of the objective function w.r.t x, must take in and return numpy vectors of the same size
            hessian: hessian of the objective function w.r.t x, must take in a numpy vector and return a square matrix
            initial: the initial guess for the algorithm, must be a numpy vector of the size that f(x) expects
            norm_grad_tol: value that the 2-norm of the gradient must be less than in order to consider a problem solved
            iter_lim: number of iterations that the algorithm will attempt before considering a problem unsolvable
            print_every: the number of iterations the algorithm prints an update after
        """
        self.f = function
        self.g = gradient
        self.h = hessian
        self.x = initial
        self.ngt = norm_grad_tol
        self.iter_lim = iter_lim
        self.print_every = print_every

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
        while np.linalg.norm(grad) > self.ngt:
            x, func, grad = self.step(grad)
            if a%self.print_every == 0:
                print(f"iteration:{a} \ninput:{x},\n function value:{func})")
            xs.append(x)
            fs.append(func)
            a+=1
            if not (self.iter_lim is None):
                if a > self.iter_lim:
                    print(np.linalg.norm(grad,1))
                    break

        return np.array(xs), np.array(fs), a
    

class steepest_descent_backtrack:
    def __init__(self, function, gradient, initial, c=1e-4, rho=0.9, norm_grad_tol=1e-5, iter_lim=None, print_every=100) -> None:
        """
        class that implements the steepest descent algorithm with step sizes that are backtracked to satisfy the Armijo condition

        inputs:
            function: objective function to be optimized, must take in a numpy vector and return a scalar 
            gradient: gradient of the objective function w.r.t x, must take in and return numpy vectors of the same size
            initial: the initial guess for the algorithm, must be a numpy vector of the size that f(x) expects
            c: tolerance parameter for the Armijo condition
            rho: value that the step length gets multiplied by when the Armijo condition is not satisfied in backtracking, must be (0,1)
            norm_grad_tol: value that the 2-norm of the gradient must be less than in order to consider a problem solved
            iter_lim: number of iterations that the algorithm will attempt before considering a problem unsolvable
            print_every: the number of iterations the algorithm prints an update after
        """
        self.f = function
        self.g = gradient
        self.c = c
        self.rho = rho
        self.x = initial
        self.ngt = norm_grad_tol
        self.iter_lim = iter_lim
        self.print_every = print_every

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
        while np.linalg.norm(grad) > self.ngt:
            x, func, grad = self.step(func, grad)
            if b%self.print_every == 0:
                print(f"iteration:{b} \ninput:{x},\n function value:{func})")
            xs.append(x)
            fs.append(func)
            b+=1
            if not (self.iter_lim is None):
                if b > self.iter_lim:
                    print(np.linalg.norm(grad,1))
                    break

        return np.array(xs), np.array(fs), b

class BFGS:
    def __init__(self, function, gradient, initial, c=1e-4, r=0.9, beta=1, norm_grad_tol=1e-5, iter_lim=None, print_every=100) -> None:
        """
        class that implements the Broyden Fletcher Goldfarb Shanno algorithm

        inputs:
            function: objective function to be optimized, must take in a numpy vector and return a scalar 
            gradient: gradient of the objective function w.r.t x, must take in and return numpy vectors of the same size
            initial: the initial guess for the algorithm, must be a numpy vector of the size that f(x) expects
            c: tolerance parameter for the Armijo condition
            r: value that the step length gets multiplied by when the Armijo condition is not satisfied in backtracking, must be (0,1)
            beta: scalar by which the identity matrix gets scaled to create H0
            norm_grad_tol: value that the 2-norm of the gradient must be less than in order to consider a problem solved
            iter_lim: number of iterations that the algorithm will attempt before considering a problem unsolvable
            print_every: the number of iterations the algorithm prints an update after
        """
        self.f = function
        self.g = gradient
        self.c = c
        self.r = r
        self.x = initial
        self.h_mat = beta * np.identity(self.x.size)
        self.ngt = norm_grad_tol
        self.iter_lim = iter_lim
        self.print_every = print_every

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
        while np.linalg.norm(grad) > self.ngt:
            x, func, grad = self.step(func, grad)
            if b%self.print_every == 0:
                print(f"iteration:{b} \ninput:{x},\n function value:{func})")
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