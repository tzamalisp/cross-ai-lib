from scipy.signal import butter, filtfilt, stft, sosfilt, sosfiltfilt
from scipy.ndimage import gaussian_filter1d, median_filter
import numpy as np


def apply_gaussian_filter(df, sigma):
    """
    Apply lowpass gaussian filter to entire dataframe.
    `scipy.ndimage.gaussian_filter1d` is used.
    Args:
        df: A dataframe where all columns correspond to a signal
        sigma: width of gaussian distribution that is convolved with
         the signal.

    Returns: A new dataframe (dataframe.copy is used) with the values after
     convolution.

    """
    df_smooth = df.copy()
    df_smooth = df_smooth.apply(lambda x: gaussian_filter1d(x, sigma))
    return df_smooth


def butterworth_filter(data, cutoff, fs, order=4, filter_type="low", sos=None):
    """
    Lowpass butterworth filter
    Args:
        data: numpy array (vector)
        cutoff: highest pass frequency
        fs: sampling frequency
        order: filter order
        filter_type: lowpass(low) or highpass(high)
        method: Method used for applying filter to data (usage of
         `scipy.filtfilt`)
    Returns: filtered array

    """
    nyq = 0.5 * fs
    if isinstance(cutoff, list):
        cutoff = np.array(cutoff)
    normal_cutoff = cutoff / nyq
    if sos is None:
        sos = butter(N=order, Wn=normal_cutoff, btype=filter_type,
                     analog=False, output="sos")
    y = sosfiltfilt(sos, data, axis=0)
    return y
