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
    def __init__(self, function, gradient, hessian, initial, norm_grad_tol=1e-5, norm_step_tol=1e-6, iter_lim=None, print_every=100) -> None:
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
        self.nst = norm_step_tol
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
            if (a%self.print_every == 0) and (a > 0):
                print(f"iteration:{a} \ninput:{x},\n function value:{func})")
            xs.append(x)
            fs.append(func)
            a+=1
            if not (self.iter_lim is None):
                if a > self.iter_lim:
                    print(np.linalg.norm(grad))
                    break

            if len(xs) > 1:
                if not (self.nst is None):
                    if np.linalg.norm(xs[-1]-xs[-2]) < self.nst:
                        print("STEP SIZE:", np.linalg.norm(xs[-1]-xs[-2]))
                        break

        return np.array(xs), np.array(fs), a
    

class steepest_descent_backtrack:
    def __init__(self, function, gradient, initial, c=1e-4, rho=0.9, norm_grad_tol=1e-5, norm_step_tol=1e-6, func_change_tol=1e-5,
                iter_lim=None, print_every=100) -> None:
        """
        class that implements the steepest descent algorithm with step sizes that are backtracked to satisfy the Armijo condition

        inputs:
            function: objective function to be optimized, must take in a numpy vector and return a scalar 
            gradient: gradient of the objective function w.r.t x, must take in and return numpy vectors of the same size
            initial: the initial guess for the algorithm, must be a numpy vector of the size that f(x) expects
            c: tolerance parameter for the Armijo condition
            rho: value that the step length gets multiplied by when the Armijo condition is not satisfied in backtracking, must be (0,1)
            norm_grad_tol: value that the 2-norm of the gradient must be less than in order to consider a problem solved
            func_change_tol: value that the change in function value for a given step needs to exceed to keep iterating
            iter_lim: number of iterations that the algorithm will attempt before considering a problem unsolvable
            print_every: the number of iterations the algorithm prints an update after
        """
        self.f = function
        self.g = gradient
        self.c = c
        self.rho = rho
        self.x = initial
        self.ngt = norm_grad_tol
        self.nst = norm_step_tol
        self.fct = func_change_tol
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
                #print(self.f(xnew), (func-self.c*a*np.matmul(grad,grad)))
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
            if (b%self.print_every == 0) and (b > 0):
                print(f"iteration:{b} \ninput:{x},\n function value:{func})")
            xs.append(x)
            fs.append(func)
            b+=1
            if not (self.iter_lim is None):
                if b > self.iter_lim:
                    print(np.linalg.norm(grad))
                    break
            
            if not(self.fct is None):
                if len(fs) > 2:
                    if (fs[-1]-fs[-2]) < self.fct:
                        break

            if len(xs) > 1:
                if not (self.nst is None):
                    if np.linalg.norm(xs[-1]-xs[-2]) < self.nst:
                        print("STEP SIZE:", np.linalg.norm(xs[-1]-xs[-2]))
                        break

        return np.array(xs), np.array(fs), b

class BFGS:
    def __init__(self, function, gradient, initial, c=1e-4, r=0.9, beta=1, norm_grad_tol=1e-5, norm_step_tol=1e-6, iter_lim=None, 
                print_every=100) -> None:
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
        self.nst = norm_step_tol
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
            if (b%self.print_every == 0) and (b > 0):
                print(f"iteration:{b} \ninput:{x},\n function value:{func})")
            xs.append(x)
            fs.append(func)
            b+=1
            if not (self.iter_lim is None):
                if b > self.iter_lim:
                    print(np.linalg.norm(grad))
                    break

            if len(xs) > 1:
                if not (self.nst is None):
                    if np.linalg.norm(xs[-1]-xs[-2]) < self.nst:
                        print("STEP SIZE:", np.linalg.norm(xs[-1]-xs[-2]))
                        break

        return np.array(xs), np.array(fs), b
    
    def h(self, sk, yk, hk):
        rho = 1/np.matmul(sk,yk)
        return np.matmul((np.identity(sk.size)-rho*np.outer(sk,yk)), \
                        np.matmul(hk, (np.identity(sk.size)-rho*np.outer(yk,sk))))\
                        +(np.identity(sk.size)-rho*np.outer(sk,sk))
    

class conjugate_gradient:
    def __init__(self, A, b, initial, norm_grad_tol=1e-6) -> None:
        """
        class that solves the linear equation Ax=b by finding argmin(x.T A x - x.T B, x)

        inputs:
            A: linear system coefficients
            b: equality vector
            initial: the initial guess for the algorithm, must be a numpy vector of the size that f(x) expects
            norm_grad_tol: value that the 2-norm of the gradient must be less than in order to consider a problem solved
            iter_lim: number of iterations that the algorithm will attempt before considering a problem unsolvable
            print_every: the number of iterations the algorithm prints an update after
        """
        self.A = A
        self.b = b
        self.x = initial
        self.ngt = norm_grad_tol

        self.r = np.matmul(self.A, self.x,) - b
        self.p = -self.r

    def step(self):
        a = np.matmul(self.r, self.r)/np.matmul(self.p.T, np.matmul(self.A, self.p))
        self.x += a*self.p
        rnew = self.r + a*np.matmul(self.A, self.p)
        self.p *= np.matmul(rnew.T, rnew)/np.matmul(self.r.T, self.r) # this is the beta calculation, but without reserving a new var
        self.p -= rnew
        self.r = rnew
    
    def solve(self):
        xs = [self.x]
        b = 0
        while np.linalg.norm(self.r) >= self.ngt:
            self.step()
            b += 1
            xs.append(self.x)

        return xs, b
    

class L_BFGS:
    def __init__(self, function, gradient, initial, m=20, c=1e-4, r=0.9, beta=1, norm_grad_tol=1e-5, norm_step_tol=1e-6, 
                iter_lim=None, print_every=100, save_steps=False) -> None:
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
        self.ss = []
        self.ys = []
        self.rhos = []
        self.m = m
        self.h_mat = beta 
        self.ngt = norm_grad_tol
        self.nst = norm_step_tol
        self.iter_lim = iter_lim
        self.print_every = print_every
        self.save_steps = save_steps

    def step(self, f_val=None, gradient=None):
        if gradient is None:
            grad = self.g(self.x)
        else:
            grad = gradient

        if f_val is None:
            func = self.f(self.x)
        else:
            func = f_val

        if type(self.h_mat) == np.ndarray:
            p = -self.h_mat
        else:
            p = self.h_mat*-grad

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
        rho = 1/np.matmul(y.T,s)
        
        self.ss.append(s)
        self.ys.append(y)
        self.rhos.append(rho)
        if len(self.rhos) > self.m:
            self.ss.pop(0)
            self.ys.pop(0)
            self.rhos.pop(0)

        self.x = xnew
        grad = self.g(self.x)
        self.h_mat = self.h(grad)

        return self.x, self.f(self.x), grad
    
    def solve(self):
        func = self.f(self.x)
        grad = self.g(self.x)

        xs = []
        fs = []
        b = 0
        while np.linalg.norm(grad) > self.ngt:
            x, func, grad = self.step(func, grad)
            if (b%self.print_every == 0) and (b > 0):
                print(f"iteration:{b} \ninput:{x},\n function value:{func})")
            xs.append(x)
            fs.append(func)
            
            if (not self.save_steps) and (len(xs) > 2):
                xs.pop(0)
                fs.pop(0)
                
            b+=1
            if not (self.iter_lim is None):
                if b > self.iter_lim:
                    print(np.linalg.norm(grad))
                    break

            if len(xs) > 1:
                if not (self.nst is None):
                    if np.linalg.norm(xs[-1]-xs[-2]) < self.nst:
                        print("STEP SIZE:", np.linalg.norm(xs[-1]-xs[-2]))
                        break

        return np.array(xs), np.array(fs), b
    
    def h(self, grad):
        q = np.copy(grad)
        alphas = np.empty(len(self.rhos))
        for a in reversed(range(len(self.rhos))):
            alpha = self.rhos[a]*np.matmul(self.ss[a].T, q)
            alphas[a] = alpha
            q -= alpha*self.ys[a]

        r = (np.matmul(self.ss[-1].T, self.ys[-1])/np.matmul(self.ys[-1].T, self.ys[-1]))*q

        for a in range(len(self.rhos)):
            beta = self.rhos[a]*np.matmul(self.ys[a].T,r)
            r += (alphas[a] - beta)*self.ss[a]
        
        return r