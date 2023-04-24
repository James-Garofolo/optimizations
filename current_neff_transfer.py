import numpy as np
from sklearn import linear_model
from torch.nn.functional import relu
import torch

#powers = np.arange(0, 1.4, 0.2)*1e-3
#neffs_pwr = np.array([1.5, 1.5000374, 1.50007501, 1.50011288, 1.50015087, 1.50020777, 1.5002345])
#neff_wrt_pwr = interpolate.CubicSpline(powers, neffs_pwr)#, bc_type='clamped')

#currents = np.array([0.0, 1.0e-3, 2.0e-3, 2.5e-3])
#neffs_current = np.array([1.5, 1.50009474, 1.5003873, 1.50065793])
#neff_wrt_current = interpolate.CubicSpline(currents, neffs_current, bc_type=('clamped', 'natural'))

powers = np.array([[0.0, 0.0002, 0.0004, 0.00048989, 0.0006, 0.0008, 0.001, 0.0012, 0.00195956, 0.00306181]]).T
neffs = np.array([1.5, 1.5000374, 1.50007501, 1.50009474, 1.50011288, 1.50015087, 1.50020777, 1.5002345, 1.5003873, 1.50065793])
#neff_wrt_pwr = interpolate.CubicSpline(powers, neffs)#, bc_type='clamped')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    """
    from math import sqrt
    Rs = []
    for a, pwr in enumerate(powers):
        R = 1
        neff = neff_wrt_current(sqrt(pwr/R))
        while abs(neff-neffs_pwr[a]) > 1e-9:
            if (neff > neffs_pwr[a]) or (sqrt(pwr/R) > currents[-1]):
                R *= 1.1
            else:
                R *= 0.9

            neff = neff_wrt_current(sqrt(pwr/R))
        
        Rs.append(R)
        print(a, sqrt(pwr/R), neff)

    print(Rs, np.mean(Rs[1:]))
    """
    #print(powers, "\n", 489.8896407309116*currents**2)

    neff_wrt_pwr = linear_model.LinearRegression()
    neff_wrt_pwr.fit(powers, neffs)
    print(neff_wrt_pwr.coef_, neff_wrt_pwr.intercept_)

    eval_currents = torch.arange(-2e-3, 4e-3, 0.01e-3)
    #eval_currents = np.array([eval_currents]).T

    plt.scatter(powers, neffs, label="ground truth")
    plt.plot(eval_currents, relu(neff_wrt_pwr.coef_[0]*eval_currents)+neff_wrt_pwr.intercept_, label="interpolation function")
    current_values = plt.gca().get_yticks()
    # using format string '{:.0f}' here but you can choose others
    plt.gca().set_yticklabels(['{:}'.format(x) for x in current_values])
    plt.xlabel("tuning power")
    plt.ylabel("effective index of refraction")
    plt.legend()
    plt.show()
