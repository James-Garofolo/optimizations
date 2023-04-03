import numpy as np
import matplotlib.pyplot as plt
import os
from math import pi, floor
from opt_algorithms import steepest_descent_backtrack, generate_finite_diff_grad
#from scipy.signal import find_peaks


baseline_data = np.loadtxt(os.getcwd()+r"/data/baseline_spectrum.csv", delimiter=',')
eval_wavelengths = baseline_data[0]*1e-9
dl = eval_wavelengths[1]-eval_wavelengths[0]

ground_truth = 10**(baseline_data[1]/10)

def through_drop(l, d = (1550 * (10**-9)),neff=1.5, a = 1, r = 0.99):
    """
    Tp(phi) in the paper
    """
    phi = 4*(pi**2)*d*neff/l
    pass_top = (a*r)**2 - 2*(r**2)*a*np.cos(phi) + r**2
    drop_top = ((1 - r**2)**2) * a
    bottom = 1 - 2*(r**2)*a*np.cos(phi) + (a*(r**2))**2
    return pass_top/bottom, drop_top/bottom        


def mrr_bank_weights(l,ds,neffs=[1.5]*4, a=1, r=0.95):
    throughs = np.ones_like(l)
    drops = np.zeros_like(l)
    for i, d in enumerate(ds):
        through, drop = through_drop(l, d, neffs[i], a, r)
        throughs *= through
        drops += drop

    return throughs, drops


def pnn_f(x):
    ds = x[:4]
    neffs = x[4:8]
    a = x[8]
    r = x[9]
    tia_g = x[10]

    through, _ = mrr_bank_weights(eval_wavelengths, ds, neffs, a, r)
    through *= tia_g
    
    return np.linalg.norm(through - ground_truth)

ds = np.array([1543e-9, 1544e-9, 1545e-9, 1546e-9])*18.7
neffs = np.array([1.5, 1.5, 1.5, 1.5])#x[4:8]
a = np.array([1])
r = np.array([0.95])
tia_g = np.array([0.07169736])#np.array([0.1])


if __name__ == "__main__":
    x0 = np.concatenate([ds,neffs,a,r, tia_g])
    sd = steepest_descent_backtrack(pnn_f, generate_finite_diff_grad(pnn_f, 1e-11, second_order=True), x0,c=1e-5,rho=0.2, 
                                    iter_lim=1000, print_every=100)

    xs, fs, num_iterations = sd.solve()
    print("\nfinal gradient: \n", sd.g(sd.x))
    print("\nresults:\n", xs[-1], fs[-1], num_iterations)

    through, _ = mrr_bank_weights(eval_wavelengths, xs[-1,:4], xs[-1,4:8], xs[-1,8], xs[-1,9])
    through *= xs[-1,10]
    
    through2, _ = mrr_bank_weights(eval_wavelengths, ds, neffs, a, r)
    through2 *= tia_g[0]

    fig, axs = plt.subplots(2)
    axs[0].plot(eval_wavelengths, ground_truth, label="ground truth")
    axs[0].plot(eval_wavelengths, through2, label="pre-optimization")
    axs[0].legend()
    axs[1].plot(eval_wavelengths, ground_truth, label="ground truth")
    axs[1].plot(eval_wavelengths, through, label="model")
    axs[1].legend()
    plt.show()
