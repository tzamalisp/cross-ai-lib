from scipy.signal import butter, filtfilt, stft, sosfilt, sosfiltfilt
from scipy.ndimage import gaussian_filter1d, median_filter
from crossai.ts.processing.signal import filters
import numpy as np
import logging
import pandas as pd

from crossai.ts.processing.signal.filters import butterworth_filter


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


def remove_gravity(df, kernel, cutoff=None, sampling_freq=None, order=None,
                   new_axes=None):
    """

    Args:
        df (pandas.Dataframe): The dataframe with the accelerometer axes data that the gravity will be removed from
        kernel (int): Corresponds to the size argument of the scipy.ndimage.median_filter function.Kernel size gives
        the shape that is taken from the input array, at every element position, to define the input to the filter
        function. The input dimension is 1, thus, the second argument of the tuple size is already defaulted to 1.
        (kernel,1).
        cutoff (int): The cutoff  frequency of the butterworth_filter.
        sampling_freq (int): The sampling frequency of the signals.
        order (int): The order of the butterworth_filter.
        new_axes (list of strs): A list of the names of the new axes that will be created after the gravity removal. The
        list should have length equal to number of signals contained in the df.

    Returns:
        new_df (pandas.Dataframe): A new dataframe that contains the signals without the gravity component. The column
        names are after the new_axes argument.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Argument df must be in pandas.Dataframe format")
    new_values = []
    med_filt_values = median_filter(df.values, size=(kernel, 1))
    gravity = butterworth_filter(med_filt_values,
                                 cutoff,
                                 sampling_freq,
                                 order=order,
                                 filter_type="lowpass")
    values = df.values - gravity
    new_values.append(values)

    new_df_values = new_values[0]
    for value_array in new_values[1:]:
        if len(value_array.shape) == 1:
            value_array = value_array[:, np.newaxis]
        # print("value_array shape : ", value_array.shape)
        new_df_values = np.hstack((new_df_values, value_array))
        # print(new_df_values.shape)
    new_df = pd.DataFrame(data=new_df_values, columns=new_axes)

    return new_df
