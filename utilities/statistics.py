import numpy as np


def calc_normal_distribution_limits(avg_len, std_len, alpha=2):
    """
    Given an average len and the corresponding std, it is calculated the
    space of values that represent a particular percent of the measured quantity.
    E.g. For alpha=1 it corresponds to the 68.2%, alpha=2 to 95.4% alpha=3 to 99.7%.
    Args:
        avg_len (float):  A distribution average value
        std_len (float): A distribution standard deviation
        alpha (int): Coefficient to select the percentage of the sample.

    Returns:
        upper (int) : The higher value of the selected part of distribution.
        lower (int) : The lower value of the selected part of distribution.

    """
    upper = np.floor(avg_len + alpha * std_len)
    lower = np.ceil(avg_len - alpha * std_len)
    return upper, lower
