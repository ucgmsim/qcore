import numpy as np

def contreras_slab_bounds(func):
    def inner(mw):
#        if mw < 5.9 or mw > 7.8:
#            raise ValueError("Magnitude out of range for model [5.9, 7.8]")
        return func(mw)

    return inner


@contreras_slab_bounds
def mw_to_a_contreras_2022_slab(mw):
    """
    contreras2022nga
    slab [5.9-7.8]
    """
    return np.power(10, 0.890 * mw - 3.225)

@contreras_slab_bounds
def a_to_mw_contreras_2022_slab(a):
    """
    contreras2022nga
    slab [5.9-7.8]
    """
    return (np.log10(a)+3.225)/0.890


@contreras_slab_bounds
def mw_to_w_contreras_2022_slab(mw):
    if mw >= 6.5:
        aspect_ratio = 10 ** (0.216*(mw-6.5)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = np.power(10, 0.890 * mw - 3.225)

    return (A/aspect_ratio)**0.5


@contreras_slab_bounds
def mw_to_l_contreras_2022_slab(mw):
    if mw >= 6.5:
        aspect_ratio = 10 ** (0.216*(mw-6.5)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = np.power(10, 0.890 * mw - 3.225)

    return (A*aspect_ratio)**0.5


def contreras_interface_bounds(func):
    def inner(mw):
#        if mw < 6.75 or mw > 9.1:
#            raise ValueError("Magnitude out of range for model [6.75, 9.1]")
        return func(mw)

    return inner


@contreras_interface_bounds
def mw_to_a_contreras_2022_interface(mw):
    """
    contreras2022nga
    interface [6.75-9.1]
    """
    return 10 ** ((-8.890/np.log(10)) + mw)

@contreras_interface_bounds
def a_to_mw_contreras_2022_interface(a):
    """
    contreras2022nga
    interface [6.75-9.1]
    """
    return np.log10(a) + 8.890/np.log(10)


@contreras_interface_bounds
def mw_to_w_contreras_2022_interface(mw):
    if mw >= 7.25:
        aspect_ratio = 10 ** (0.6248*(mw-7.25)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = 10 ** ((-8.890/np.log(10)) + mw)

    return (A/aspect_ratio)**0.5


@contreras_interface_bounds
def mw_to_l_contreras_2022_interface(mw):
    if mw >= 7.25:
        aspect_ratio = 10 ** (0.6248*(mw-7.25)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = 10 ** ((-8.890/np.log(10)) + mw)

    return (A*aspect_ratio)**0.5
