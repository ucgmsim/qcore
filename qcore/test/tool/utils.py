"""
Common functions for the test script
"""

import numpy as np
import os

ERROR_LIMIT = 1e-06


def compare_np_array(array1, array2, error_limit=ERROR_LIMIT):
    """first test if the two numpy arrays are of the same shape
       if pass, then test if the relative error between the two arrays are <= the preset error limit
       if any of the test fails, assertion error will be raised;
       array1: a numpy array from sample output, will be used as the denominator,
       array2: a numpy array from test output, makes part of the numerator.
       error_limit: preset error_limit to be compared with the relative error (array1-array2)/array1
    """

    assert array1.shape == array2.shape
    assert np.all(np.isclose(array1, array2, rtol=error_limit))


def remove_file(abs_path):
    """ remove test output file if successfully passed the test
        abs_path: abs path of the file to be removed
    """
    try:
        os.remove(abs_path)
    except (IOError, OSError):
        raise
