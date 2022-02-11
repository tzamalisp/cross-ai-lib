from scipy.signal import butter, filtfilt, stft, sosfilt, sosfiltfilt
from scipy.ndimage import gaussian_filter1d, median_filter
from processing.signal import filters
import numpy as np
import logging
import pandas as pd

from crossai.ts.processing.signal.filters import butterworth_filter, apply_gaussian_filter



def calculate_magnitude(array, axis=1):
    """
    Calculates the magnitude of a given ndarray.
    Args:
        array (numpy.ndarray): numpy array holding the data.
        axis (int 0,1): axis of np array to calculate magnitude.

    Returns:
        (numpy.ndarray) the magnitude of the values of the input array.
    """
    return np.apply_along_axis(lambda x: np.sqrt(np.power(x, 2).sum()), axis,
                               array)


def add_signals(config, df):
    """
    Signals preprocessing occurs here. For each of the given signals names
    (according to configuration)
    the corresponding procedure takes place.
    Currently are supported the following signals names:
    acc : Raw accelerometer values (3 signals)
    gyr:  Raw gyroscope values (3 signals)
    filter_acc : Filtered accelerometer according to configuration (3-signals)
    filter_gyr: Filtered gyroscope according to configuration (3-signals)
    acc_magnitude : magnitude of raw accelerometer values (1 axe: sum of the
                    square of the elements)
    gyr_magnitude : magnitude of raw gyroscope values
    magn_filter_acc: magnitude of filtered accelerometer values
                     (1 axe: sum of the square of the elements)
    magn_filter_gyr: magnitude of filtered gyroscope values (1 axe: sum of the
                     square of the elements)
    wgc_acc: Raw Accelerometer without gravity component (Highpass butterworth
             at 0.3 Hz - 3 signals)
    magn_wgc: Magnitude of wgc_acc (1 axe)

    Args:
        config: ( dict ) configuration file
        df: pandas dataframe with the default signals names

    Returns:
        new_df (pandas dataFrame) with columns the signals that occured after
        processing, according to configuration.
    """
    new_axes = []
    new_values = []
    axes = config["axes"]

    # Set the kernel used in median filtering. If it does not exist in
    # configuration, use default size 3.
    kernel = config["filter"].get("kernel")
    if kernel is None:
        kernel = 3
        msg = "kernel was not found in configuration parameters. " \
              "The default value (kernel=3) will be used in median filtering"
        logging.warning(msg)

    for axe in axes:
        # First case where raw data are maintained:
        if axe in ["acc", "gyr"]:
            # print("{0} : simple case".format(axe))
            if axe in "acc":
                selected_axes = axes_acc
            else:
                selected_axes = axes_gyro
            values = df[selected_axes].values
            new_axes += [axe_name for axe_name in selected_axes]
            new_values.append(values)

        if axe in ["filter_acc", "filter_gyr", "magn_filter_acc",
                   "magn_filter_gyr"]:
            # print("{0} : filter case".format(axe))
            if axe in ["filter_acc", "magn_filter_acc"]:
                selected_axes = axes_acc
            else:
                selected_axes = axes_gyro
            if config["filter"]["filter_type"] == "butterworth":
                # First use a median filter, according to HAR dataset creation
                med_filt_values = median_filter(df[selected_axes].values,
                                                size=(kernel, 1))
                values = butterworth_filter(med_filt_values,
                                            cutoff=config["filter"]
                                            ["filter_lowpass_cutoff"],
                                            fs=config["sampling_frequency"],
                                            order=config["filter"]
                                            ["filter_order"],
                                            filter_type="lowpass")
            elif config["filter"]["filter_type"] == "gaussian":
                # print("{0} : gaussian case".format(axe))
                sigma = config["filter"]["sigma"]
                # TODO remove dataframe handling
                df_gaussianized = apply_gaussian_filter(df[selected_axes],
                                                        sigma)
                values = df_gaussianized.values()
            elif config["filter"]["filter_type"] == "median":
                # print("{0} : gaussian case".format(axe))
                kernel = config["filter"]["kernel"]
                values = median_filter(df[selected_axes].values,
                                       size=(kernel, 1))
            else:
                raise Exception("Not known filter function ({0})".format(
                    config["filter"]["filter_type"]))
            if axe in ["filter_acc", "filter_gyr"]:
                new_axes += ["flt_" + axe_name for axe_name in selected_axes]
                new_values.append(values)
            if axe in ["magn_filter_acc", "magn_filter_gyr"]:
                # print("{0} : magn filter case".format(axe))
                magn = calculate_magnitude(values)
                new_axes += [axe]
                new_values.append(magn)
        # Magnitude axes
        if axe in ["acc_magnitude", "gyr_magnitude"]:
            if axe in "acc_magnitude":
                np_axes = df[axes_acc].values
            else:
                np_axes = df[axes_gyro].values
            # print("{0} : simple magn case".format(axe))
            magn = calculate_magnitude(np_axes)
            new_axes += [axe]
            new_values.append(magn)
        # Without gravity component
        if axe in ["wgc_acc", "magn_wgc"]:
            # print("{0} : wgc case".format(axe))
            med_filt_values = median_filter(df[axes_acc].values, size=(kernel,
                                                                       1))
            gravity = butterworth_filter(med_filt_values,
                                         config["filter"]
                                         ["filter_highpass_cutoff"],
                                         config["sampling_frequency"],
                                         order=config["filter"]
                                         ["filter_order"],
                                         filter_type="lowpass")
            values = df[axes_acc].values - gravity
            if axe in "wgc_acc":
                new_axes += ["wgc_" + axe_name for axe_name in axes_acc]
                new_values.append(values)
            # Magnitude of axes where gravity component has been removed
            if axe in "magn_wgc":
                # print("{0} : magn wgc case".format(axe))
                magn = calculate_magnitude(values)
                new_axes += [axe]
                new_values.append(magn)
    new_df_values = new_values[0]
    for value_array in new_values[1:]:
        if len(value_array.shape) == 1:
            value_array = value_array[:, np.newaxis]
        # print("value_array shape : ", value_array.shape)
        new_df_values = np.hstack((new_df_values, value_array))
        # print(new_df_values.shape)
    new_df = pd.DataFrame(data=new_df_values, columns=new_axes)

    return new_df
