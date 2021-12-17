from src.processing.timeseries.motion.preprocessing import *
from src.processing.timeseries.signal.features_extraction \
    import signal_features_extraction
from utilities.lists_common_utils import find_list_in_list
import numpy as np

feature_taxonomy = dict()
feature_taxonomy["statistical"] = ["mean", "min", "max", "variance", "std", "rms", "range",
                                   "skew", "kurtosis"]


def motion_features_extraction(df, sampling_frequency, features_extraction_parameters, feature_categories=["all"]):
    """

    Args:
        df:
        feature_categories (list): Supported values are ["statistical", "all", "time", "frequency"]
        features_extraction_parameters (dict):
        sampling_frequency(int, optional) : Default 100.
    Returns:

    """
    df_raw = df.copy()
    samples = df_raw.shape[0]
    # Filter data with lowpass butterworth filter
    # lowpass_butterworth_filter_1df(df, sampling_frequency, settings.filter_cutoff_freq,
    #                                settings.filter_order)
    add_preprocessing_axes(df)
    # dictionary to keep features
    features = dict()
    # 3-axis accelerometer signals
    df_acc = df_raw[[axes_acc[0], axes_acc[1], axes_acc[2]]].copy()
    # Signal duration
    features["duration"] = np.round(samples / sampling_frequency, 3).astype(np.float32)
    # SMA for accelerometer signals
    df_acc["sum_axis"] = df_acc.apply(
        lambda x: np.abs(x[axes_acc[0]]) + np.abs(x[axes_acc[1]]) + np.abs(x[axes_acc[2]]), axis=1)
    features["acc_sma"] = df_acc["sum_axis"].mean()
    features["acc_dsma"] = df_acc["sum_axis"].diff().mean()
    features["acc_dsvm"] = df["acc_magnitude"].diff().mean()
    # 3-axis gyroscope features
    df_gyr = df_raw[[axes_gyro[0], axes_gyro[1], axes_gyro[2]]].copy()

    df_gyr["sum_axis"] = df_gyr.apply(
        lambda x: np.abs(x[axes_gyro[0]]) + np.abs(x[axes_gyro[1]]) + np.abs(x[axes_gyro[2]]), axis=1)
    features["gyr_sma"] = df_gyr["sum_axis"].mean()
    features["gyr_dsma"] = df_gyr["sum_axis"].diff().mean()
    features["gyr_dsvm"] = df["gyr_magnitude"].diff().mean()

    # Statistical features
    for axe_signal in axes_acc+axes_gyro+["acc_magnitude", "gyr_magnitude"]:
        features.update(signal_features_extraction(df[axe_signal],
                                                   feature_categories=feature_categories,
                                                   sampling_frequency=sampling_frequency,
                                                   features_extraction_parameters=features_extraction_parameters,
                                                   postfix=axe_signal))
    #######################################################################
    # Correlation
    if find_list_in_list(["corr", "all"], feature_categories):
        features["corr_acc_x_y"] = np.corrcoef(df[axes_acc[0]], df[axes_acc[1]])[0, 1].round(5)
        features["corr_acc_x_z"] = np.corrcoef(df[axes_acc[0]], df[axes_acc[2]])[0, 1].round(5)
        features["corr_acc_y_z"] = np.corrcoef(df[axes_acc[1]], df[axes_acc[2]])[0, 1].round(5)
        features["corr_acc_x_m"] = np.corrcoef(df[axes_acc[0]], df["acc_magnitude"])[0, 1].round(5)
        features["corr_acc_y_m"] = np.corrcoef(df[axes_acc[1]], df["acc_magnitude"])[0, 1].round(5)
        features["corr_acc_z_m"] = np.corrcoef(df[axes_acc[2]], df["acc_magnitude"])[0, 1].round(5)
        features["corr_gyr_x_y"] = np.corrcoef(df[axes_gyro[0]], df[axes_gyro[1]])[0, 1].round(5)
        features["corr_gyr_x_z"] = np.corrcoef(df[axes_gyro[0]], df[axes_gyro[2]])[0, 1].round(5)
        features["corr_gyr_y_z"] = np.corrcoef(df[axes_gyro[1]], df[axes_gyro[2]])[0, 1].round(5)
        features["corr_gyr_x_m"] = np.corrcoef(df[axes_gyro[0]], df["gyr_magnitude"])[0, 1].round(5)
        features["corr_gyr_y_m"] = np.corrcoef(df[axes_gyro[1]], df["gyr_magnitude"])[0, 1].round(5)
        features["corr_gyr_z_m"] = np.corrcoef(df[axes_gyro[2]], df["gyr_magnitude"])[0, 1].round(5)

        # Cross corellation
        features["crosscorr_acc_x_y"] = np.correlate(df[axes_acc[0]], df[axes_acc[1]])[0]
        features["crosscorr_acc_x_z"] = np.correlate(df[axes_acc[0]], df[axes_acc[2]])[0]
        features["crosscorr_acc_y_z"] = np.correlate(df[axes_acc[1]], df[axes_acc[2]])[0]
        features["crosscorr_acc_x_m"] = np.correlate(df[axes_acc[0]], df["acc_magnitude"])[0]
        features["crosscorr_acc_y_m"] = np.correlate(df[axes_acc[1]], df["acc_magnitude"])[0]
        features["crosscorr_acc_z_m"] = np.correlate(df[axes_acc[2]], df["acc_magnitude"])[0]
        features["crosscorr_gyr_x_y"] = np.correlate(df[axes_gyro[0]], df[axes_gyro[1]])[0]
        features["crosscorr_gyr_x_z"] = np.correlate(df[axes_gyro[0]], df[axes_gyro[2]])[0]
        features["crosscorr_gyr_y_z"] = np.correlate(df[axes_gyro[1]], df[axes_gyro[2]])[0]
        features["crosscorr_gyr_x_m"] = np.correlate(df[axes_gyro[0]], df["gyr_magnitude"])[0]
        features["crosscorr_gyr_y_m"] = np.correlate(df[axes_gyro[1]], df["gyr_magnitude"])[0]
        features["crosscorr_gyr_z_m"] = np.correlate(df[axes_gyro[2]], df["gyr_magnitude"])[0]

    for key in features:
        features[key] = float(np.round(features[key], decimals=5))
    return features
