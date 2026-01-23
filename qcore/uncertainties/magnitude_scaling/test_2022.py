import numpy as np

def test_slab_bounds(func):
    def inner(mw):
#        if mw < 5.9 or mw > 7.8:
#            raise ValueError("Magnitude out of range for model [5.9, 7.8]")
        return func(mw)

    return inner


@test_slab_bounds
def mw_to_a_test_2022_slab(mw):
    """
    contreras2022nga
    slab [5.9-7.8]
    """
    return np.power(10, 0.890 * mw - 3.225)*5 # TEST

@test_slab_bounds
def a_to_mw_test_2022_slab(a):
    """
    contreras2022nga
    slab [5.9-7.8]
    """
    return (np.log10(a/5)+3.225)/0.890 # TEST


@test_slab_bounds
def mw_to_w_test_2022_slab(mw):
    if mw >= 6.5:
        aspect_ratio = 10 ** (0.216*(mw-6.5)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = np.power(10, 0.890 * mw - 3.225)*5 # TEST

    return (A/aspect_ratio)**0.5


@test_slab_bounds
def mw_to_l_test_2022_slab(mw):
    if mw >= 6.5:
        aspect_ratio = 10 ** (0.216*(mw-6.5)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = np.power(10, 0.890 * mw - 3.225)*5 # TEST

    return (A*aspect_ratio)**0.5


def test_interface_bounds(func):
    def inner(mw):
#        if mw < 6.75 or mw > 9.1:
#            raise ValueError("Magnitude out of range for model [6.75, 9.1]")
        return func(mw)

    return inner


@test_interface_bounds
def mw_to_a_test_2022_interface(mw):
    """
    contreras2022nga
    interface [6.75-9.1]
    """
    return 10 ** ((-8.890/np.log(10)) + mw)*5 # TEST

@test_interface_bounds
def a_to_mw_test_2022_interface(a):
    """
    contreras2022nga
    interface [6.75-9.1]
    """
    return np.log10(a/5) + 8.890/np.log(10) # TEST


@test_interface_bounds
def mw_to_w_test_2022_interface(mw):
    if mw >= 7.25:
        aspect_ratio = 10 ** (0.6248*(mw-7.25)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = 10 ** ((-8.890/np.log(10)) + mw)*5 # TEST

    return (A/aspect_ratio)**0.5


@test_interface_bounds
def mw_to_l_test_2022_interface(mw):
    if mw >= 7.25:
        aspect_ratio = 10 ** (0.6248*(mw-7.25)/np.log(10)) # L/W
    else:
        aspect_ratio = 10 ** (0) # L/W

    A = 10 ** ((-8.890/np.log(10)) + mw)*5 # TEST

    return (A*aspect_ratio)**0.5
