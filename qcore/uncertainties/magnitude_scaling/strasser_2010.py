import numpy as np


def strasser_interface_bounds(func):
    def inner(mw):
        if mw < 6.3 or mw > 9.4:
            raise ValueError("Magnitude out of range for model [6.3, 9.4]")
        return func(mw)

    return inner


def strasser_slab_bounds(func):
    def inner(mw):
        if mw < 5.9 or mw > 7.8:
            raise ValueError("Magnitude out of range for model [5.9, 7.8]")
        return func(mw)

    return inner


@strasser_interface_bounds
def mw_to_a_strasser_2010_interface(mw):
    """
    strasser2010scaling
    interface [6.3-9.4]
    """
    return np.power(10, 0.952 * mw - 3.476)


def a_to_mw_strasser_2010_interface(a):
    """
    Inversion of the above equation
    strasser2010scaling
    interface [6.3-9.4]
    """
    if a < 332 or a > 297000:
        raise ValueError("Magnitude out of range for model")
    return (np.log10(a) + 3.476) / 0.952


@strasser_interface_bounds
def mw_to_w_strasser_2010_interface(mw):
    return np.power(10, 0.351 * mw - 0.882)


@strasser_interface_bounds
def mw_to_l_strasser_2010_interface(mw):
    return np.power(10, 0.585 * mw - 2.477)


@strasser_slab_bounds
def mw_to_a_strasser_2010_slab(mw):
    """
    strasser2010scaling
    slab [5.9-7.8]
    """
    return np.power(10, 0.890 * mw - 3.225)


@strasser_slab_bounds
def mw_to_w_strasser_2010_slab(mw):
    return np.power(10, 0.356 * mw - 1.058)


@strasser_slab_bounds
def mw_to_l_strasser_2010_slab(mw):
    return np.power(10, 0.562 * mw - 2.350)
