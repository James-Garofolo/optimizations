import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
from scipy.optimize import rosen, rosen_der, rosen_hess
import math
from opt_algorithms import *


def ackley(x, a=20, b=0.2, c=2*math.pi):
    return -a*math.exp(-b*math.sqrt(np.matmul(x.T,x)/x.size)) - math.exp(np.sum(np.cos(c*x))/x.size) + a + math.exp(1) 

def ackley_grad(x, a=20, b=0.2, c=2*math.pi):
    return a*math.exp(-b*math.sqrt(np.matmul(x.T,x)/x.size))*b*math.sqrt(np.matmul(x.T,x)/x.size)*x/x.size\
        - math.exp(np.sum(np.cos(c*x))/x.size)*c*(-np.sin(c*x))/x.size


if __name__ == "__main__":
    #n = 20#12#8#5
    #cg = conjugate_gradient(hilbert(n), np.ones(n), np.zeros(n))
    #xs, b = cg.solve()

    #print(xs[-1], b)

    d = 5
    x0 = np.ones(d)*1.2#np.array([-1.2, 1])
    
    sd = steepest_descent_backtrack(ackley, ackley_grad, x0, rho = 0.1, print_every=10000)
    xs, fs, b = sd.solve()

    print("steepest", xs[-1], b)

    nm = newton(rosen, rosen_der, rosen_hess, x0, print_every=10000)
    xs, fs, b = nm.solve()

    print("newton", xs[-1], b)
    
    qnm =BFGS(rosen, rosen_der, x0, r = 0.1, print_every=10000)
    xs, fs, b = qnm.solve()

    print("bfgs", xs[-1], b)
    

    
    d = 1000
    x0 = np.ones(d)*1.2#np.array([-1.2, 1])
    lqnm = L_BFGS(rosen, rosen_der, x0, 20, r=0.1, print_every=1000)
    xs, fs, b = lqnm.solve()

    print("l bfgs", fs[-1], b)

         

    