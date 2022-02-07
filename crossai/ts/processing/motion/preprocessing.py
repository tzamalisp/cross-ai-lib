import logging

import numpy as np
import pandas as pd

from src.processing.timeseries.motion.project_processing_motion_variables\
    import axes_acc, axes_gyro, accepted_keys_to_generic


def get_motion_signals_data_from_document(dataset_document):

    """
    Accepts a mongodb document and returns a dataframe. The length of data is
    provided by `datalen` field which is expected to be calculated in mongo
    project querie. It converts the keys of the dictionary to generally
    accepted motion sensor key names.
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
        if key in list(accepted_keys_to_generic.keys()) or key in\
                axes_acc+axes_gyro:
            # this way all document keys would be converted to generic keys
            if len(dataset_dict[key]) > datalen:
                # logging.warning("Disagreement on DataFrame axes length.
                # Rejecting values to obtain the accepted len.")
                dataset_dict[key] = dataset_dict[key][:datalen]
            if key not in axes_acc+axes_gyro:
                df_dict[accepted_keys_to_generic[key]] = dataset_dict[key]
            else:
                df_dict[key] = dataset_dict[key]
    df = pd.DataFrame.from_dict(df_dict)
    return df


def recreate_signal_column_names(axes):
    """
    Given a list of signal names regarding motion (accelerometer and/or
    gyroscope) the signal names are recreated by replacing the acc name in axes
    name with each of the accelerometer signals (x, y, z).E.g. if axes contains
    the axes category `filter_acc`, then the new list will contain
    `filter_acc_x`, `filter_acc_y`, `filter_acc_z`.
    Args:
        axes (list): Contains strings that should contain either `acc` or
        `gyr` substrings.

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
    Function to recreate a dataframe from an instance of the dataset and
    further add the magnitude signal and the sum signal.
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
        df[col_name] = np.apply_along_axis(lambda x:
                                           np.sqrt(np.power(x, 2).sum()), 1,
                                           df[signals].values)
        col_name = signals_category+"_sum"
        df[col_name] = np.apply_along_axis(lambda x: np.sum(x),
                                           1, df[signals].values)
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
    return np.apply_along_axis(lambda x: np.sqrt(np.power(x, 2).sum()),
                               axis, array)


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
    Calculates the duration of a signal. Main hypothesis is that sampling is
    uniform in time.
    Args:
        samples (int): Number of values of signal.
        sampling_frequency (float): The frequency of the sampled signal.

    Returns:
        duration (float): duration of signal (by default in seconds if
        frequency is expressed in cycles per second).
    """
    return samples / sampling_frequency


def append_instances(dfs_list):
    """
    Creates a new dataframe with the acc and gyroscope axes of all the
    instances in the list.
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
