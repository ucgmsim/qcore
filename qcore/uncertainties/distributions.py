import random

import numpy as np
from scipy.stats import truncnorm, weibull_min


def relative_uniform(mean, scale_factor):
    """Returns a value from a uniform distribution where the min and max
    are scaled relative to the middle value given as the mean"""
    return random.uniform(mean * (1 - scale_factor), mean * (1 + scale_factor))


def uniform(mean, half_range):
    return random.uniform(mean - half_range, mean + half_range)


def truncated_normal(mean, std_dev, std_dev_limit=2):
    return float(
        truncnorm(-std_dev_limit, std_dev_limit, loc=mean, scale=std_dev).rvs()
    )


def bounded_truncated_normal(mean, upper_limit, lower_limit):
    dist_range = upper_limit - lower_limit
    return float(
        truncnorm(
            (mean - lower_limit) / dist_range,
            (upper_limit - mean) / dist_range,
            loc=mean,
            scale=dist_range,
        ).rvs()
    )


def weibull(k=3.353, scale_factor=0.612):
    """Weibull distribution. Defaults are for nhm2srf dhypo generation"""
    return scale_factor * np.random.weibull(k)


def truncated_weibull(truncation_threshold, k=3.353, scale_factor=0.612):
    """Forces the weibull distribution to have a value less than some threshold.
    Used for generating hypocentre down dip with default values and a threshold of 1.
    With these parameters there is a 0.56% chance of the value being greater than 1, so this is a good enough solution."""
    return_value = 2 * truncation_threshold
    while return_value > truncation_threshold:
        return_value = weibull(k=k, scale_factor=scale_factor)
    return return_value


def proper_weibull(k=3.353, scale_factor=0.612):
    return weibull_min(c=k, scale=scale_factor).rvs()


def proper_truncated_weibull(
    upper_truncation_threshold, lower_truncation_threshold, k=3.353, scale_factor=0.612
):
    dist = weibull_min(c=k, scale=scale_factor)
    upper_value, lower_value = dist.cdf(
        (upper_truncation_threshold, lower_truncation_threshold)
    )
    dist_range = upper_value - lower_value
    val = (
        dist.cdf(
            np.random.uniform(lower_truncation_threshold, upper_truncation_threshold)
        )
        - lower_value
    ) / dist_range
    # if val < lower_truncation_threshold or val > upper_truncation_threshold:
    #     print("Broken")
    # print(val, dist_range, upper_value, lower_value, upper_truncation_threshold, lower_truncation_threshold)
    return val


def truncated_log_normal(mean, std_dev, std_dev_limit=2) -> float:
    return np.exp(
        truncnorm(
            -std_dev_limit,
            std_dev_limit,
            loc=np.log(np.asarray(mean).astype(np.float)),
            scale=std_dev,
        ).rvs()
    )


def bounded_truncated_log_normal(mean, upper_limit, lower_limit) -> float:
    dist_range = upper_limit - lower_limit
    return np.exp(
        truncnorm(
            (mean - lower_limit) / dist_range,
            (upper_limit - mean) / dist_range,
            loc=np.log(np.asarray(mean).astype(np.float)),
            scale=dist_range,
        ).rvs()
    )


def rand_shyp():
    """Generates a value for the hypocentre along the length of the fault. Uses defaults from nhm2srf"""
    # normal distribution
    shyp_mu = 0.0
    shyp_sigma = 0.25

    shyp = truncated_normal(shyp_mu, shyp_sigma)
    return shyp
