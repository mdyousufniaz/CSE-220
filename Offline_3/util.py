import numpy as np

from math import pi

def integration(signal, func, var_domain, const_domain):
    return np.array([
        np.trapezoid(
            signal * func(2 * pi * point * const_domain),
            var_domain
        ) for point in var_domain
    ])