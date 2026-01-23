import numpy as np


def allen_bounds(func):
    def inner(mw):
        if mw < 7.1 or mw > 9.5:
            raise ValueError("Magnitude out of range for model")
        return func(mw)

    return inner


@allen_bounds
def mw_to_a_allen_2017_slab(mw):
    return np.power(10, 0.96 * mw - 3.89)


@allen_bounds
def mw_to_w_allen_2017_slab(mw):
    return np.power(10, 0.35 * mw - 1.01)


@allen_bounds
def mw_to_l_allen_2017_slab(mw):
    return np.power(10, 0.63 * mw - 3.03)


@allen_bounds
def mw_to_a_allen_2017_linear_interface(mw):
    return np.power(10, 0.96 * mw - 3.63)


@allen_bounds
def mw_to_w_allen_2017_linear_interface(mw):
    return np.power(10, 0.35 * mw - 0.86)


@allen_bounds
def mw_to_l_allen_2017_linear_interface(mw):
    return np.power(10, 0.63 * mw - 2.90)


@allen_bounds
def mw_to_a_allen_2017_bilinear_interface(mw):
    if mw <= 8.63:
        A_a = np.power(10, 1.22 * mw - 5.62)
    else:
        A_a = np.power(10, 0.31 * mw + 2.23)

    return A_a


@allen_bounds
def mw_to_w_allen_2017_bilinear_interface(mw):
    if mw <= 8.67:
        w = np.power(10, 0.48 * mw - 1.91)
    else:
        w = np.power(10, 0.00 * mw + 2.29)
    return w


@allen_bounds
def mw_to_l_allen_2017_bilinear_interface(mw):
    return np.power(10, 0.63 * mw - 2.90)
