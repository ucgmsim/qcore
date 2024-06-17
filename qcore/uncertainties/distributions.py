"""

This module provides functions to generate random values from a few quake-specific distributions.

Default scaling parameters are applied to the weibull and rand_shyp
distributions to match rupture ddhyp and shyp generation values.

Functions:
- truncated_normal: Generates random values from a truncated normal distribution.
- truncated_weibull: Generates random values from a truncated Weibull distribution.
- truncated_weibull_expected_value: Calculates the expected value of a truncated Weibull distribution.
- truncated_log_normal: Generates random values from a truncated log-normal distribution.
- rand_shyp: Generates random hypocentre values along the length of a fault.
"""

from typing import Optional

import numpy as np
import scipy as sp


def truncated_normal(mean: float, std_dev: float, std_dev_limit: float = 2) -> float:
    """
    Generate a random value from a truncated normal distribution.

    Parameters
    ----------
    mean : float
        Mean of the normal distribution.
    std_dev : float
        Standard deviation of the normal distribution.
    std_dev_limit : float, optional
        Number of standard deviations to limit the truncation (default is 2).

    Returns
    -------
    float
        Random value from the truncated normal distribution.
    """
    return sp.stats.truncnorm(
        -std_dev_limit, std_dev_limit, loc=mean, scale=std_dev
    ).rvs()


def truncated_weibull(
    upper_value: float,
    c: float = 3.353,
    scale_factor: float = 0.612,
    seed: Optional[int] = None,
) -> float:
    """
    Generate a random value from a truncated Weibull distribution.

    Parameters
    ----------
    upper_value : float
        Upper value for truncation of the Weibull distribution.
    c : float, optional
        Shape parameter of the Weibull distribution (default is 3.353).
    scale_factor : float, optional
        Scale factor of the Weibull distribution (default is 0.612).
    seed : int or None, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    float
        Random value from the truncated Weibull distribution.
    """
    return upper_value * sp.stats.truncweibull_min(
        c, 0, 1 / scale_factor, scale=scale_factor
    ).rvs(random_state=seed)


def truncated_weibull_expected_value(
    upper_value: float, c: float = 3.353, scale_factor: float = 0.612
) -> float:
    """
    Calculate the expected value for a truncated Weibull distribution.

    Parameters
    ----------
    upper_value : float
        Upper value for truncation of the Weibull distribution.
    c : float, optional
        Shape parameter of the Weibull distribution (default is 3.353).
    scale_factor : float, optional
        Scale factor of the Weibull distribution (default is 0.612).

    Returns
    -------
    float
        Expected value for the truncated Weibull distribution.
    """
    return (
        upper_value
        * sp.stats.truncweibull_min(c, 0, 1 / scale_factor, scale=scale_factor).expect()
    )


def truncated_log_normal(
    mean: float, std_dev: float, std_dev_limit: float = 2, seed: Optional[int] = None
) -> float:
    """
    Generate a random value from a truncated log-normal distribution.

    Parameters
    ----------
    mean : float
        Mean of the log-normal distribution.
    std_dev : float
        Standard deviation of the log-normal distribution.
    std_dev_limit : float, optional
        Number of standard deviations to limit the truncation (default is 2).
    seed : int or None, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    float
        Random value from the truncated log-normal distribution.
    """
    return np.exp(
        sp.stats.truncnorm(
            -std_dev_limit,
            std_dev_limit,
            loc=np.log(np.asarray(mean).astype(np.float64)),
            scale=std_dev,
        ).rvs(random_state=seed)
    )


def rand_shyp() -> float:
    """
    Generate a random hypocentre value along the length of a fault.

    Returns
    -------
    float
        Random value from a truncated normal distribution (mean=0, std_dev=0.25).
    """
    return truncated_normal(0, 0.25)
