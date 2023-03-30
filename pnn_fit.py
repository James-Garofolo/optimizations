import numpy as np
import matplotlib.pyplot as plt
from HW3 import steepest_descent_backtrack
import os
from math import pi


baseline_data = np.loadtxt(os.getcwd()+r"/data/baseline_spectrum.csv", delimiter=',')
eval_wavelengths = baseline_data[0]
ground_truth = baseline_data[1]

def through_drop(l, d = (1550 * (10**-9)), neff = 1.5, a = 1, r = 0.99):
    """
    Tp(phi) in the paper
    """
    phi = 4*(pi**2)*d*neff/l
    pass_top = (a*r)**2 - 2*(r**2)*a*np.cos(phi) + r**2
    drop_top = ((1 - r**2)**2) * a
    bottom = 1 - 2*(r**2)*a*np.cos(phi) + (a*(r**2))**2
    return pass_top/bottom, drop_top/bottom        


def mrr_bank_weights(l,ds,neffs, a=1, r=0.95):
    throughs = np.ones_like(l)
    drops = np.zeros_like(l)
    for i, d in enumerate(ds):
        through, drop = through_drop(l, d, neffs[i], a, r)
        throughs *= through
        drops += drop

    return drops-throughs


def pnn_f(x):
    ds = x[:4]
    neffs = x[4:8]
    a = x[8]
    r = x[9]

    through, _ = mrr_bank_weights(eval_wavelengths, ds, neffs, a, r)

    return np.linalg.norm(through-ground_truth)




if __name__ == "__main__":

    pass