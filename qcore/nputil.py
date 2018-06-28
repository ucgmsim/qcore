"""
Functions that work on numpy objects.
"""

import numpy as np

def argsearch(needles, haystack):
    """
    Allows mapping 2 arrays like DB tables to get index of one in another.
    source https://stackoverflow.com/a/8251757
    needles: array of items to find the index for
    haystack: array of items to look into
    """

    index = np.argsort(haystack)
    needle_index = np.take(index, np.searchsorted(haystack[index], needles), \
                           mode="clip")

    # unfound values are np.ma.masked
    return np.ma.array(needle_index, mask=haystack[needle_index] != needles)
