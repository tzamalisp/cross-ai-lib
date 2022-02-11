from scipy.signal import butter, filtfilt, stft, sosfilt, sosfiltfilt
from scipy.ndimage import gaussian_filter1d, median_filter
from processing.signal import filters
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
        df:
        kernel:
        cutoff:
        sampling_freq:
        order:
        new_axes:

    Returns:

    """
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
