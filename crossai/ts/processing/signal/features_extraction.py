"""
Features extraction module for a single signal
"""
from scipy.stats import kurtosis
from scipy.stats import skew
import antropy as ant
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft
import peakutils
import numpy as np
import logging
from PyAstronomy import pyaC
from utilities.lists_common_utils import find_list_in_list


def calculate_rms(data):
    """
    Calculates the root mean square (RMS) of a signal.
    Args:
        data (np.array): The input signal.

    Returns:
        (float): The RMS of the input
    """
    if len(data.shape) != 1:
        msg = "Expected 1-dimensional signal for RMS calculation but shape found is {}".format(len(data.shape))
        logging.error(msg)
        raise Exception(msg)
    square_signal = np.power(data, 2)
    return np.sqrt(square_signal.sum().mean())


def signal_features_extraction(data, feature_categories, sampling_frequency, features_extraction_parameters,
                               postfix=None):
    """
    Calculate features for a signal.
    Args:
        data (np.array or list): Input timeseries/signal to calculate features.
        feature_categories (list): Contains the name of the features to calculate.
        sampling_frequency (int): sampling frequency of the signal.
        features_extraction_parameters (dict):
        postfix (str or None, optional): When used, the feature name is expanded with the
            postfix value. E.g. when feature `max` is calculated, and postfix is set to
            `abc`, then the features dictionary key that would be created is `max_abc`.

    Returns:
        features: a dictionary with features of the input.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    features = dict()
    keywords_stat_features = ["statistical", "all", "time"]
    keywords_other_time_domain = ["other_time_domain", "all", "time"]
    keywords_frequency_domain = ["all", "frequency"]

    if find_list_in_list(["max"] + keywords_stat_features, feature_categories):
        f_name = "max" if "postfix" is None else "max_" + postfix
        features[f_name] = data.max()
    if find_list_in_list(["min"] + keywords_stat_features, feature_categories):
        f_name = "min" if "postfix" is None else "min_" + postfix
        features[f_name] = data.min()
    if find_list_in_list(["range"] + keywords_stat_features, feature_categories):
        f_name = "range" if "postfix" is None else "range_" + postfix
        features[f_name] = np.abs(data.max() - data.min())
    if find_list_in_list(["mean"] + keywords_stat_features, feature_categories):
        f_name = "mean" if "postfix" is None else "mean_" + postfix
        features[f_name] = data.mean()
    if find_list_in_list(["std"] + keywords_stat_features, feature_categories):
        f_name = "std" if "postfix" is None else "std_" + postfix
        features[f_name] = data.std()
    if find_list_in_list(["variance"] + keywords_stat_features, feature_categories):
        f_name = "variance" if "postfix" is None else "variance_" + postfix
        features[f_name] = data.var()

    if find_list_in_list(["rms"] + keywords_stat_features, feature_categories):
        f_name = "rms" if "postfix" is None else "rms_" + postfix
        features[f_name] = calculate_rms(data)

    if find_list_in_list(["skew"] + keywords_stat_features, feature_categories):
        f_name = "skew" if "postfix" is None else "skew_" + postfix
        features[f_name] = skew(data)

    if find_list_in_list(["kurtosis"] + keywords_stat_features, feature_categories):
        f_name = "kurtosis" if "postfix" is None else "kurtosis_" + postfix
        features[f_name] = kurtosis(data)

    if find_list_in_list(["zero_crossings"] + keywords_other_time_domain, feature_categories):
        f_name = "zero_crossings" if "postfix" is None else "zero_crossings_" + postfix
        zc = pyaC.zerocross1d(np.arange(data.shape[0]), data, getIndices=False)
        features[f_name] = zc.shape[0]

    if find_list_in_list(["peaks"] + keywords_other_time_domain, feature_categories):
        f_name = "peaks" if "postfix" is None else "peaks_" + postfix
        thres = features_extraction_parameters["peak_utils_thres"]
        min_dist = features_extraction_parameters["peak_utils_min_dist"]
        peaks = peakutils.indexes(data, thres=thres, min_dist=min_dist)
        features[f_name] = len(peaks) if peaks is not None else 0

    if find_list_in_list(keywords_frequency_domain, feature_categories):
        signal_fft = rfft(data)
        signal_fft_square = np.square(signal_fft)
        # TODO verify if it is correct to use DC component this way.
        f_name = "dc" if "postfix" is None else "dc_" + postfix
        features[f_name] = signal_fft[0]

        f_name = "coeff_sum" if "postfix" is None else "coeff_sum_" + postfix
        features[f_name] = np.sum(signal_fft)

        f_name = "energy" if "postfix" is None else "energy_" + postfix
        features[f_name] = signal_fft_square.mean()

        f_name = "spectral_entropy" if "postfix" is None else "spectral_entropy_" + postfix
        method = features_extraction_parameters["spectral_entropy_method"]
        nperseg = 256
        if data.shape[0] < 64:
            nperseg = 32
        elif data.shape[0] < 128:
            nperseg = 64
        elif data.shape[0] < 256:
            nperseg = 128
        features[f_name] = ant.spectral_entropy(data,
                                                sampling_frequency,
                                                nperseg=nperseg,
                                                method=method, normalize=True)

        stft_window_length = features_extraction_parameters["stft_window_length"]
        stft_window = features_extraction_parameters["stft_window"]
        noverlap = features_extraction_parameters["stft_noverlap"]
        signal_max_ampl, signal_max_freq, max_freq_moment = stft_max_ampl_freq(data, sampling_frequency,
                                                                               stft_window=stft_window_length,
                                                                               window=stft_window,
                                                                               noverlap=noverlap)
        f_name = "stft_ampl" if "postfix" is None else "stft_ampl_" + postfix
        features[f_name] = signal_max_ampl

        f_name = "stft_freq" if "postfix" is None else "stft_freq_" + postfix
        features[f_name] = signal_max_ampl

        f_name = "stft_freq" if "postfix" is None else "stft_moment_" + postfix
        features[f_name] = signal_max_ampl

    return features


def stft_max_ampl_freq(data, sampling_frequency, stft_window, window, noverlap):
    try:
        f, t, zxx = stft(data, sampling_frequency, nperseg=stft_window,
                         window=window, noverlap=noverlap)
        peak = np.amax(zxx)
        ampl = np.real(peak)
        peakind = np.argmax(zxx)
        pos_row = peakind // zxx.shape[1]
        pos_col = peakind - pos_row * zxx.shape[1]
        m_time = t[pos_col]
        freq = f[pos_row]
    except ValueError as e:
        logging.error(e)
        ampl, freq, m_time = None, None, None

    return ampl, freq, m_time
