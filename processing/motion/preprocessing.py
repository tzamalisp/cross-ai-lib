import logging

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from processing.timeseries.preprocessing import butterworth_filter, apply_gaussian_filter
from processing.motion.project_processing_motion_variables import axes_acc, axes_gyro, accepted_keys_to_generic


def calc_acc_magnitude(df):
    """
    Accepts a dataframe that is expected to contain accelerometer sensors data. Returns the same dataFrame with a new
    column acc_magnitude.
    Args:
        df (pandas DataFrame):

    Returns:

    """
    df["acc_magnitude"] = calculate_magnitude(df[axes_acc].values)


def calc_gyr_magnitude(df):
    """
    Accepts a dataFrame that is expected to contain accelerometer sensors data. Returns the same dataFrame with a new
    column gyr_magnitude.
    Args:
        df (pandas DataFrame):

    Returns:

    """
    df["gyr_magnitude"] = calculate_magnitude(df[axes_gyro].values)


def get_motion_signals_data_from_document(dataset_document):

    """
    Accepts a mongodb document and returns a dataframe. The length of data is provided
    by `datalen` field which is expected to be calculated in mongo project querie. It converts the keys of the
    dictionary to generally accepted motion sensor key names.
    Args:
        dataset_document: dictionary document from mongodb

    Returns:
        Pandas dataframe with the available axes in the dataset_document.

    """
    dataset_dict = (dataset_document.get("data"))
    dataset_df = pd.DataFrame.from_dict(dataset_dict)
    datalen = dataset_document.get("datalen")
    if datalen is None:
        datalengths = list()
        for key in dataset_dict.keys():
            keylen = np.count_nonzero(~np.isnan(dataset_dict[key]))
            datalengths.append(keylen)
        datalen = np.min(datalengths)
    df_dict = dict()
    for key in dataset_dict.keys():
        if key in list(accepted_keys_to_generic.keys()) or key in axes_acc+axes_gyro:
            # this way all document keys would be converted to generic keys
            if len(dataset_dict[key]) > datalen:
                # logging.warning("Disagreement on DataFrame axes length. Rejecting values to obtain the accepted len.")
                dataset_dict[key] = dataset_dict[key][:datalen]
            if key not in axes_acc+axes_gyro:
                df_dict[accepted_keys_to_generic[key]] = dataset_dict[key]
            else:
                df_dict[key] = dataset_dict[key]
    df = pd.DataFrame.from_dict(df_dict)
    return df


def recreate_signal_column_names(axes):
    """
    Given a list of signal names regarding motion (accelerometer and/or gyroscope) the signal names
    are recreated by replacing the acc name in axes name with each of the accelerometer signals (x, y, z).
    E.g. if axes contains the axes category `filter_acc`, then the new list will contain `filter_acc_x`,
    `filter_acc_y`, `filter_acc_z`.
    Args:
        axes (list): Contains strings that should contain either `acc` or `gyr` substrings.

    Returns:
        A list with all the signals that occur from the categories.
    """
    # Recreate the axes column names
    axes_signals = list()
    for axes_category in axes:
        if "acc" in axes_category:
            for signal in axes_acc:
                axes_signals.append(axes_category.replace("acc", signal))
        if "gyr" in axes_category:
            for signal in axes_gyro:
                axes_signals.append(axes_category.replace("gyr", signal))
    return axes_signals


def recreate_dataframe_and_append_signals(instance, axes, axes_signals):
    """
    Function to recreate a dataframe from an instance of the dataset and further add the magnitude signal and the
    sum signal.
    """
    df = pd.DataFrame(instance, columns=axes_signals)
    accumulated_signals = list()
    for axes_category in axes:
        signals_cat = list()
        if "acc" in axes_category:
            for signal in axes_acc:
                signals_cat.append(axes_category.replace("acc", signal))
            accumulated_signals.append(signals_cat)
        if "gyr" in axes_category:
            for signal in axes_gyro:
                signals_cat.append(axes_category.replace("gyr", signal))
            accumulated_signals.append(signals_cat)
    for signals, signals_category in zip(accumulated_signals, axes):
        col_name = signals_category+"_magnitude"
        df[col_name] = np.apply_along_axis(lambda x: np.sqrt(np.power(x, 2).sum()), 1, df[signals].values)
        col_name = signals_category+"_sum"
        df[col_name] = np.apply_along_axis(lambda x: np.sum(x), 1, df[signals].values)
    return df


def calculate_magnitude(array, axis=1):
    """
    Calculates the magnitude of a given ndarray.
    Args:
        array (numpy.ndarray): numpy array holding the data
        axis (int 0,1): axis of np array to calculate magnitude

    Returns:
        (numpy.ndarray) the magnitude of the values of the input array
    """
    return np.apply_along_axis(lambda x: np.sqrt(np.power(x, 2).sum()), axis, array)


def calculate_sma(array, axis=1):
    """
    Calculates the sma (signal magnitude area) of a given ndarray.
    Args:
        array (numpy.ndarray): numpy array holding the data
        axis (int 0,1): axis of np array to calculate magnitude

    Returns:
        (numpy.ndarray) the magnitude of the values of the input array
    """
    return np.apply_along_axis(lambda x: np.abs(x).sum(), axis, array)


def calculate_signal_duration(samples, sampling_frequency):
    """
    Calculates the duration of a signal. Main hypothesis is that sampling is uniform in time.
    Args:
        samples (int): Number of values of signal.
        sampling_frequency (float): The frequency of the sampled signal.

    Returns:
        duration (float): duration of signal (by default in seconds if frequency is expressed
            in cycles per second).
    """
    return samples / sampling_frequency


def add_signals(config, df):
    """
    Signals preprocessing occurs here. For each of the given signals names (according to configuration)
    the corresponding procedure takes place. Currently are supported the following signals names:
    acc : Raw accelerometer values (3 signals)
    gyr:  Raw gyroscope values (3 signals)
    filter_acc : Filtered accelerometer according to configuration (3-signals)
    filter_gyr: Filtered gyroscope according to configuration (3-signals)
    acc_magnitude : magnitude of raw accelerometer values (1 axe: sum of the square of the elements)
    gyr_magnitude : magnitude of raw gyroscope values
    magn_filter_acc: magnitude of filtered accelerometer values (1 axe: sum of the square of the elements)
    magn_filter_gyr: magnitude of filtered gyroscope values (1 axe: sum of the square of the elements)
    wgc_acc: Raw Accelerometer without gravity component (Highpass butterworth at 0.3 Hz - 3 signals)
    magn_wgc: Magnitude of wgc_acc (1 axe)

    Args:
        config: ( dict ) configuration file
        df: pandas dataframe with the default signals names

    Returns:
        new_df (pandas dataFrame) with columns the signals that occured after processing, according to configuration.
    """
    new_axes = []
    new_values = []
    axes = config["axes"]

    # Set the kernel used in median filtering. If it does not exist in configuration, use default size 3.
    kernel = config["filter"].get("kernel")
    if kernel is None:
        kernel = 3
        msg = "kernel was not found in configuration parameters. The default value (kernel=3) will be used" \
              "in median filtering"
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

        if axe in ["filter_acc", "filter_gyr", "magn_filter_acc", "magn_filter_gyr"]:
            # print("{0} : filter case".format(axe))
            if axe in ["filter_acc", "magn_filter_acc"]:
                selected_axes = axes_acc
            else:
                selected_axes = axes_gyro
            if config["filter"]["filter_type"] == "butterworth":
                # First use a median filter, according to HAR dataset creation
                med_filt_values = median_filter(df[selected_axes].values, size=(kernel, 1))
                values = butterworth_filter(med_filt_values,
                                            cutoff=config["filter"]["filter_lowpass_cutoff"],
                                            fs=config["sampling_frequency"],
                                            order=config["filter"]["filter_order"],
                                            filter_type="lowpass")
            elif config["filter"]["filter_type"] == "gaussian":
                # print("{0} : gaussian case".format(axe))
                sigma = config["filter"]["sigma"]
                # TODO remove dataframe handling
                df_gaussianized = apply_gaussian_filter(df[selected_axes], sigma)
                values = df_gaussianized.values()
            elif config["filter"]["filter_type"] == "median":
                # print("{0} : gaussian case".format(axe))
                kernel = config["filter"]["kernel"]
                values = median_filter(df[selected_axes].values, size=(kernel, 1))
            else:
                raise Exception("Not known filter function ({0})".format(config["filter"]["filter_type"]))
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
            med_filt_values = median_filter(df[axes_acc].values, size=(kernel, 1))
            gravity = butterworth_filter(med_filt_values,
                                         config["filter"]["filter_highpass_cutoff"],
                                         config["sampling_frequency"],
                                         order=config["filter"]["filter_order"],
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


def add_preprocessing_axes(df):
    # TODO add documentation

    logging.debug("Adding preprocessing columns")
    calc_acc_magnitude(df)
    calc_gyr_magnitude(df)


def append_instances(dfs_list):
    """
    Creates a new dataframe with the acc and gyroscope axes of all the instances
    in the list.
    Args:
        dfs_list (list): List of DataFrames.

    Returns:
        type: Description of returned object.

    """
    new_df = dict()
    for signal_name in axes_acc + axes_gyro:
        new_df[signal_name] = list()
    for instance in dfs_list:
        for signal_name in axes_acc + axes_gyro:
            if signal_name in instance.columns:
                new_df[signal_name].append(instance[signal_name].values)
    for signal_name in axes_acc + axes_gyro:
        new_df[signal_name] = np.hstack(new_df[signal_name])
    new_df = pd.DataFrame.from_dict(new_df)
    return new_df
