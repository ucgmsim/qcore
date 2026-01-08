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

from typing import Literal, overload

import numpy as np
import numpy.typing as npt
import scipy as sp


@overload
def truncated_normal(
    mean: float,
    std_dev: float,
    std_dev_limit: float = ...,
    size: Literal[1] = 1,
    seed: int | None = None,
) -> float: ...  # numpydoc ignore=GL08


@overload
def truncated_normal(
    mean: float,
    std_dev: float,
    std_dev_limit: float = ...,
    size: int = 1,
    seed: int | None = None,
) -> np.ndarray: ...  # numpydoc ignore=GL08


def truncated_normal(
    mean: float,
    std_dev: float,
    std_dev_limit: float = 2,
    size: int = 1,
    seed: int | None = None,
) -> float | np.ndarray:
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
    size : int, optional
        The number of samples to take (default is 1).
    seed : int or None, optional
        Random seed for reproducibility (default is None).


    Returns
    -------
    float or array of floats
        Random value from the truncated normal distribution.
    """
    x = sp.stats.truncnorm(-std_dev_limit, std_dev_limit, loc=mean, scale=std_dev).rvs(
        size=size, random_state=seed
    )
    if x.size == 1:
        return float(x.item())
    else:
        return x


@overload
def truncated_weibull(
    upper_value: float,
    c: float = ...,
    scale_factor: float = ...,
    size: Literal[1] = 1,
    seed: int | None = ...,
) -> float: ...  # numpydoc ignore=GL08


@overload
def truncated_weibull(
    upper_value: float,
    c: float = ...,
    scale_factor: float = ...,
    size: int = 1,
    seed: int | None = ...,
) -> np.ndarray: ...  # numpydoc ignore=GL08


def truncated_weibull(
    upper_value: float,
    c: float = 3.353,
    scale_factor: float = 0.612,
    size: int = 1,
    seed: int | None = None,
) -> float | np.ndarray:
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
    size : int, optional
        The number of samples to take (default is 1).
    seed : int or None, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    float or array of floats
        Random value from the truncated Weibull distribution.
    """
    x = upper_value * sp.stats.truncweibull_min(
        c, 0, 1 / scale_factor, scale=scale_factor
    ).rvs(random_state=seed, size=size)
    if x.size == 1:
        return float(x.item())
    else:
        return x


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
    return float(
        upper_value
        * sp.stats.truncweibull_min(c, 0, 1 / scale_factor, scale=scale_factor).expect()
    )


@overload
def truncated_log_normal(
    mean: npt.ArrayLike,
    std_dev: float,
    std_dev_limit: float = ...,
    size: Literal[1] = 1,
    seed: int | None = ...,
) -> float: ...  # numpydoc ignore=GL08


@overload
def truncated_log_normal(
    mean: npt.ArrayLike,
    std_dev: float,
    std_dev_limit: float = ...,
    size: int = ...,
    seed: int | None = ...,
) -> np.ndarray: ...  # numpydoc ignore=GL08


def truncated_log_normal(
    mean: npt.ArrayLike,
    std_dev: float,
    std_dev_limit: float = 2,
    size: int = 1,
    seed: int | None = None,
) -> float | np.ndarray:
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
    size : int, optional
        The number of samples to take (default is 1).
    seed : int or None, optional
        Random seed for reproducibility (default is None).

    Returns
    -------
    float
        Random value from the truncated log-normal distribution.
    """
    x = np.exp(
        sp.stats.truncnorm(
            -std_dev_limit,
            std_dev_limit,
            loc=np.log(np.asarray(mean).astype(np.float64)),
            scale=std_dev,
        ).rvs(size=size, random_state=seed)
    )
    if x.size == 1:
        return float(x.item())
    else:
        return x


@overload
def rand_shyp(
    size: Literal[1] = 1, seed: int | None = ...
) -> float: ...  # numpydoc ignore=GL08


@overload
def rand_shyp(
    size: int = 1, seed: int | None = ...
) -> np.ndarray: ...  # numpydoc ignore=GL08


def rand_shyp(size: int = 1, seed: int | None = None) -> float | np.ndarray:
    """
    Generate a random hypocentre value along the length of a fault.

    Returns
    -------
    float
        Random value from a truncated normal distribution (mean=0, std_dev=0.25).
    """
    x = truncated_normal(0, 0.25, size=size, seed=seed)
    if x.size == 1:
        return float(x.item())
    else:
        return x
