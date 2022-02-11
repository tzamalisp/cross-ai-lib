"""
Data augmentation methods for motion. Based on methods included in `tsaug` library and methods described in
Um, T. T., et. Al., (2017). Data augmentation of wearable sensor data for Parkinson’s disease monitoring
 using convolutional neural networks. ICMI 2017 - Proceedings of the 19th ACM International Conference on
 Multimodal Interaction, 2017-Janua, 216–220. https://doi.org/10.1145/3136755.3136817
 
 
It is expected that to any of the below methods, accelerometer data are provided and optionally, gyroscope.
It is also expected that the data are in a pandas.DataFrame format.
"""
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline  # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from utilities.lists_common_utils import match_list_in_list
from src.processing.timeseries.motion.project_processing_motion_variables\
    import axes_acc, axes_gyro
from tsaug import AddNoise
from crossai.ts.processing.motion.std_processing_motion_variables import axes_acc, axes_gyro


def aug_addnoise(data, loc_acc=0.0, loc_gyr=0.0, scale_acc=0.2, scale_gyr=15.0,
                 distr="gaussian", kind="additive", per_channel=False,
                 normalize=False):
    """
    For the parameters see https://tsaug.readthedocs.io/en/stable/references.html#tsaug.AddNoise
    Two different augmenters are used for accelerometer and gyroscope as the
    two sensors have different range values.
    Args:
        data (pandas.DataFrame): An instance of the data, containing axes_acc
        and/or axes_gyr columns
        loc_acc:
        loc_gyr:
        scale_acc:
        scale_gyr:
        distr:
        kind:
        per_channel:
        normalize:


    Returns:
        aug_data_df (pandas.DataFrame): The instance of the Notebook with augmented data
    """
    data_columns = data.columns
    aug_data_gyr = None
    data_acc = data[axes_acc].values
    augmenter_acc = AddNoise(loc=loc_acc, scale=scale_acc, distr=distr, kind=kind, per_channel=per_channel,
                             normalize=normalize)
    aug_data_acc = augmenter_acc.augment(data_acc)
    if match_list_in_list(axes_gyro, data_columns):
        data_gyr = data[axes_gyro].values
        augmenter_gyr = AddNoise(loc=loc_gyr, scale=scale_gyr, distr=distr, kind=kind, per_channel=per_channel,
                                 normalize=normalize)
        aug_data_gyr = augmenter_gyr.augment(data_gyr)
    if aug_data_gyr is not None:
        aug_data = np.hstack([aug_data_acc, aug_data_gyr])
    else:
        aug_data = aug_data_acc
    aug_data_df = pd.DataFrame(aug_data, columns=data_columns)
    parameters_str = "(loc acc: {}, scale acc : {}, " \
                     "loc gyr: {}, scale gyr : {}, distr : {} )".format(loc_acc, scale_acc, loc_gyr, scale_gyr, distr)
    logging.debug(parameters_str)
    return aug_data_df


def generate_random_curves(dims, sigma=0.2, knots=4, same_curve=True):
    """

    Args:
        dims:
        sigma:
        knots:
        same_curve:

    Returns:

    """
    xx = (np.ones((dims[1], 1)) * (np.arange(0, dims[0], (dims[0] - 1) / (knots + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knots + 2, dims[1]))
    x_range = np.arange(dims[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    if not same_curve:
        cs_y = CubicSpline(xx[:, 1], yy[:, 1])
        cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    else:
        cs_y, cs_z = cs_x, cs_x
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def distort_timesteps(dims, sigma=0.2, knots=4, same_curve=False):
    """

    Args:
        dims:
        sigma:
        knots:
        same_curve:

    Returns:

    """
    tt = generate_random_curves(dims, sigma=sigma, knots=knots, same_curve=same_curve)  # Regard these samples around 1
    # as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have data.shape[0]
    t_scale = [(dims[0] - 1) / tt_cum[-1, 0], (dims[0] - 1) / tt_cum[-1, 1], (dims[0] - 1) / tt_cum[-1, 2]]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def aug_magn_warp(data, sigma=0.2, knots=4, same_curve=False):
    """

    Args:
        data:
        sigma:
        knots:
        same_curve:

    Returns:

    """
    data_columns = data.columns
    data_acc = data[axes_acc].values
    random_curves = generate_random_curves(data_acc.shape, sigma, knots, same_curve)
    aug_data_acc = data_acc * random_curves
    aug_data_gyr = None
    if match_list_in_list(axes_gyro, data_columns):
        data_gyr = data[axes_gyro].values
        aug_data_gyr = data_gyr * generate_random_curves(data_gyr.shape, sigma, knots, same_curve)
    if aug_data_gyr is not None:
        aug_data = np.hstack([aug_data_acc, aug_data_gyr])
    else:
        aug_data = aug_data_acc
    aug_data_df = pd.DataFrame(aug_data, columns=data_columns)
    return aug_data_df


def aug_timewarp(data, sigma=0.2, knots=4, same_curve=True):
    """

    Args:
        data:
        sigma:
        knots:
        same_curve:

    Returns:

    """
    data_columns = data.columns
    data_acc = data[axes_acc].values
    aug_data_gyr = None
    tt_new = distort_timesteps(data_acc.shape, sigma=sigma, knots=knots,
                               same_curve=same_curve)

    def np_interp(data, tt_new):
        data_dims = data.shape
        data_new = np.zeros(data_dims)
        x_range = np.arange(data_dims[0])
        for i in range(0, data_dims[1]):
            data_new[:, i] = np.interp(x_range, tt_new[:, i], data[:, i])
        return data_new

    aug_data_acc = np_interp(data_acc, tt_new)
    if match_list_in_list(axes_gyro, data_columns):
        data_gyr = data[axes_gyro].values
        aug_data_gyr = np_interp(data_gyr, tt_new)
    if aug_data_gyr is not None:
        aug_data = np.hstack([aug_data_acc, aug_data_gyr])
    else:
        aug_data = aug_data_acc
    aug_data_df = pd.DataFrame(aug_data, columns=data_columns)
    return aug_data_df


def aug_rotation(data, **kwargs):
    """
    Rotation is applied only on accelerometer data.
    Args:
        data:
        **kwargs (unused): Only for compatibility with the calling of the rest
            of the functions.

    Returns:

    """
    data_columns = data.columns
    data_acc = data[axes_acc].values
    axis = np.random.uniform(low=-1, high=1, size=data_acc.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    aug_data_acc = np.matmul(data_acc, axangle2mat(axis, angle))
    if match_list_in_list(axes_gyro, data_columns):
        data_gyr = data[axes_gyro].values
        aug_data = np.hstack([aug_data_acc, data_gyr])
    else:
        aug_data = aug_data_acc
    aug_data_df = pd.DataFrame(aug_data, columns=data_columns)
    return aug_data_df
