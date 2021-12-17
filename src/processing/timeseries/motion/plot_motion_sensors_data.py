"""
Various functions for plotting the available motion instances.
# TODO alter functions to support:
# TODO dynamic plotting of the different domains in a unified or separate manner.
# TODO Additional axes. This should occur as plotting new axes (e.g. magnetometer, or actual acceleration) with
# TODO all axes in the same plot and additionally plotting the magnitude.
"""

import random
import logging
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
from utilities.visualizations import display_and_save_fig
from utilities.lists_common_utils import match_list_in_list
from configuration_functions.project_configuration_variables import project_configuration
import matplotlib.pyplot as plt
from matplotlib.mlab import window_none
from utilities.visualizations import save_fig
from predictions.timeseries_predictions_visualizations import LABELS_COLORS
from src.processing.timeseries.motion.project_processing_motion_variables import axes_acc, axes_gyro, spectrogram_default_params
# TODO change location of functions it does not only correspond to motion.


def specgram_window_selector(window_name, nfft):
    """

    Args:
        window_name:
        nfft:

    Returns:

    """
    window = None
    if window_name == "hanning":
        window = signal.get_window("hann", nfft)
    if window_name == "hamming":
        window = signal.get_window("hamming", nfft)
    if window_name == "bartlett":
        window = signal.get_window("bartlett", nfft)
    if window_name == "blackman":
        window = signal.get_window("blackman", nfft)
    if window_name == "none":
        print("selected is None")
        window = window_none(np.ones(nfft, ))
    return window


def plot_spectrogram(sig, ax=None, stft_params=None, colorbar=False, **kwargs):
    """
    Calculates and plots the STFT of signal as a spectrogram.
    Args:
        sig (ndarray): input signal. It is expected to be one-dimensional.
        ax (matplotlib.axe): matplotlib.axe to plot the spectrogram.
        stft_params (dict): Should include the keys `stft_window`, `window`, `sampling_frequency`, `ylim`, `noverlap`
                            `ylim`, `cmap`. If None uses the default values that are defined in
                            project_default_parameters.project_processing_variables.spectrogram_default_params.
                            Defaults to None.
        colorbar (boolean, optional): If True, a colorbar would be depicted in the side of the confusion matrix.
                            Defaults to False.
    Returns:
        None. Plots the spectrogram of the input signal

    """
    if stft_params is None:
        stft_params = spectrogram_default_params

    # Calculate STFT samples overlap according to the given percentage
    noverlap = int(np.floor(0.01 * stft_params["noverlap"] * stft_params["stft_window"]))
    # Calculate STFT
    # f, t, zxx = signal.stft(sig, stft_params["sampling_frequency"], nperseg=stft_params["stft_window"],
    #                         window=stft_params["window"], noverlap=noverlap)

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    mode = kwargs.get("mode", "psd")
    scale = kwargs.get("scale", "linear")
    cmap = kwargs.get("cmap", "jet")
    # freqs, t, spectrum = signal.spectrogram(sig,
    #                                         fs=stft_params["sampling_frequency"],
    #                                         nfft=stft_params["stft_window"],
    #                                         window=specgram_window_selector(stft_params["window"],
    #                                                                         stft_params["stft_window"],
    #                                                                         ),
    #                                         noverlap=noverlap,
    #                                         mode=mode,
    #                                         return_onesided=True,
    #                                         scaling="spectrum"
    #
    #                                         )

    # freqs, t, spectrum = signal.stft(sig,
    #                                  fs=stft_params["sampling_frequency"],
    #                                  nperseg=stft_params["stft_window"],
    #                                  nfft=stft_params["stft_window"],
    #                                  window=specgram_window_selector(stft_params["window"],
    #                                                                  stft_params["stft_window"],
    #                                                                  ),
    #                                  noverlap=noverlap,
    #                                  return_onesided=True
    #                                  )
    # spectrum = np.square(np.abs(spectrum))

    spectrum, freqs, t, im = ax.specgram(sig,
                                         Fs=stft_params["sampling_frequency"],
                                         sides="onesided",
                                         NFFT=stft_params["stft_window"],
                                         window=specgram_window_selector(stft_params["window"],
                                                                         stft_params["stft_window"],
                                                                         ),
                                         noverlap=noverlap,
                                         scale_by_freq=True,
                                         mode=mode, scale=scale, cmap=cmap)
    # im = ax.pcolormesh(t, freqs, spectrum, cmap=stft_params["cmap"], shading="auto")
    ylim = stft_params.get("ylim", stft_params["sampling_frequency"] // 2)
    ax.set_ylim([0, ylim])
    if colorbar:
        fig = plt.gcf()
        fig.colorbar(im, ax=ax)
    return spectrum, freqs, t, im


def plot_motion_sensor_df(df, title=None, path_to_save=None, **kwargs):
    """
    Plot only the fundamental axes, if they exist in the dataframe
    Args:
        df(pandas DataFrame): A DataFrame with the defined motion sensors" data.
        title (str):
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
            being depicted through `plt.show()`. Default None.

        **kwargs:
            figsize (tuple): Default value (6,6)
            orientation (str): It can be `vertical` or `matrix`. If `matrix`, the detected signals are plotted
                in a matrix formation with no more than 4 plots in a row. If vertical, all plots are placed below each
                other. Default `matrix`.
            segments (processing.timeseries.preprocessing.SegmentsCollection):
                If segments are passed as arguments, they are colored in the plots as vertical lines.
                When `segments` argument is passed, `labels` argument should also exist, otherwise it will have no
                 effect.
            labels (list): List of strings with the all the names of the segments.
    Returns:
        None. Plots the figure or saves it in the path, given as argument.
    """

    magnitudes_to_plot = 0
    list_of_data_to_plot = list()
    if match_list_in_list(axes_acc, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(axes_acc)
    if match_list_in_list(axes_gyro, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(axes_gyro)
    if "acc_magnitude" in df.columns:
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["acc_magnitude"])
    if "gyr_magnitude" in df.columns:
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["gyr_magnitude"])
    flt_acc_axes = ["flt_acc_x", "flt_acc_y", "flt_acc_z"]
    if match_list_in_list(flt_acc_axes, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(flt_acc_axes)
    flt_gyr_axes = ["flt_gyr_x", "flt_gyr_y", "flt_gyr_z"]
    if match_list_in_list(flt_gyr_axes, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(flt_gyr_axes)
    if "filter_acc_magnitude" in df.columns:
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["filter_acc_magnitude"])
    if "filter_gyro_magnitude" in df.columns:
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["filter_gyro_magnitude"])
    if "filter_acc_sum" in df.columns:
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["filter_acc_sum"])
    if "filter_gyro_sum" in df.columns:
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["filter_gyro_sum"])
    flt_acc_axes = ["filter_acc_x", "filter_acc_y", "filter_acc_z"]
    if match_list_in_list(flt_acc_axes, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(flt_acc_axes)
    flt_gyr_axes = ["filter_gyr_x", "filter_gyr_y", "filter_gyr_z"]
    if match_list_in_list(flt_gyr_axes, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(flt_gyr_axes)
    if match_list_in_list(["magn_filter_acc"], df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["magn_filter_acc"])
    if match_list_in_list(["magn_filter_gyr"], df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["magn_filter_gyr"])
    wgc_acc_axes = ["wgc_acc_x", "wgc_acc_y", "wgc_acc_z"]
    if match_list_in_list(wgc_acc_axes, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(wgc_acc_axes)
    wgc_gyr_axes = ["wgc_gyr_x", "wgc_gyr_y", "wgc_gyr_z"]
    if match_list_in_list(wgc_gyr_axes, df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(wgc_gyr_axes)
    if match_list_in_list(["magn_wgc"], df.columns):
        magnitudes_to_plot += 1
        list_of_data_to_plot.append(["magn_wgc"])
    orientation = kwargs.get("orientation", "matrix")
    if orientation == "matrix":
        fig_cols = magnitudes_to_plot // 2 if magnitudes_to_plot >= 2 else magnitudes_to_plot
        fig_rows = magnitudes_to_plot // fig_cols
    else:
        fig_rows = magnitudes_to_plot
        fig_cols = 1
    figsize = kwargs.get("figsize", (6, 6))

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=figsize, sharex=True, num=kwargs.get("num", None),
                            constrained_layout=True)
    if not (fig_rows == 1 and fig_cols == 1):
        for list_of_axes, ax_i in zip(list_of_data_to_plot, axs.flatten()):
            for ax_name in list_of_axes:
                ax_i.plot(df[ax_name], label=ax_name)
                ax_i.grid()
                ax_i.legend()
    else:
        for ax_name in list_of_data_to_plot:
            axs.plot(df[ax_name], label=ax_name)
            axs.grid()
            axs.legend()
    segments = kwargs.get("segments", None)
    labels = kwargs.get("labels", None)
    if segments is not None and labels is not None:
        for ax in fig.axes:
            for seg in segments:
                if isinstance(seg.label, str):
                    label_color = LABELS_COLORS[labels.index(seg.label)]
                else:
                    label_color = LABELS_COLORS[seg.label]
                ax.axvspan(seg.start, seg.stop, facecolor=label_color, alpha=0.3)
    if title is not None:
        fig.suptitle(title)
    display_and_save_fig(path_to_save)


def plot_dataframe_as_spectrogram(df, stft_params=None, colorbar=True, path_to_save=None):
    """
    Plots in one column all the numerical axes (dtype=float64) of the input dataframe as spectrogram.
    #TODO Change function to be used as more generic.
    Args:
        df (pandasDataframe): The dataframe with the signals to be plotted.
        stft_params (dict): Should include the keys `stft_window`, `window`, `sampling_frequency`, `ylim`, `noverlap`
                            `ylim`, `cmap`. If None uses the default values that are defined in
                            project_default_parameters.project_processing_variables.spectrogram_default_params.
                            Defaults to None.
        colorbar (boolean, optional): If True, a colorbar would be depicted in the side of each plot axe.
                            Defaults to True.
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
                                        being depicted through `plt.show()`. Default None.

    Returns:
        None. Plots the figure or saves it in the path, given as argument.
    """
    if stft_params is None:
        stft_params = spectrogram_default_params
    number_of_axes = len(df.select_dtypes(include=["float64"]).columns)
    _, axs = plt.subplots(number_of_axes, 1, figsize=[10, 20], constrained_layout=True, sharex=True)
    ax_names = df.select_dtypes(include=["float64"]).columns
    for ax_name, ax_ind in zip(ax_names, axs.flatten()):
        plot_spectrogram(df[ax_name].values, ax=ax_ind, stft_params=stft_params, colorbar=colorbar)
        ax_ind.set_title(ax_name)

    if path_to_save:
        save_fig(path_to_save)
        plt.close()
    else:
        plt.show()


def plot_motion_time_frequency(df, stft_params=spectrogram_default_params,
                               title=None,
                               figsize=(20, 20),
                               fig=None, colorbar=True, path_to_save=None):
    """
    Hardcoded plot that creates a complete plot of the fundamental motion axes (accelerometer x,y,z, accelerometer
    magnitude, gyroscope x,y,z, gyroscope magnitude).
    Args:
        title (string, optional): A figure title
        df: A dataframe that includes the required axes, named after the dfefault project variables naming (see
            project_default_parameters.project_processing_variables.axes_acc, axes_gyro).
        stft_params (dict): Should include the keys `stft_window`, `window`, `sampling_frequency`, `ylim`, `noverlap`
            `ylim`, `cmap`. If None uses the default values that are defined in
            project_default_parameters.project_processing_variables.spectrogram_default_params.
            Defaults to None.
        colorbar (boolean, optional): If True, a colorbar would be depicted in the side of each plot axe.
            Defaults to True.
        fig (matplotlib figure, optional): If defined, the plot would be created in an external figure.
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
            being depicted through `plt.show()`. Default None.
    Returns:
        None. Plots the figure or saves it in the path, given as argument.
    """

    if fig is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=2, nrows=8)
    f_ax0 = fig.add_subplot(spec[0:3, 0])
    f_ax0.plot(df["acc_x"], label="acc_x")
    f_ax0.plot(df["acc_y"], label="acc_y")
    f_ax0.plot(df["acc_z"], label="acc_z")
    f_ax0.legend()
    f_ax0.grid()
    f_ax3 = fig.add_subplot(spec[3, 0])
    f_ax3.plot(df["acc_magnitude"])
    f_ax3.set_title("acc_magnitude")
    f_ax01 = fig.add_subplot(spec[0, 1])
    _, _, _, _ = plot_spectrogram(df["acc_x"].values, ax=f_ax01, stft_params=stft_params, colorbar=colorbar)
    f_ax01.set_title("acc_x")
    f_ax11 = fig.add_subplot(spec[1, 1])
    _, _, _, _ = plot_spectrogram(df["acc_y"].values, ax=f_ax11, stft_params=stft_params, colorbar=colorbar)
    f_ax11.set_title("acc_y")
    f_ax21 = fig.add_subplot(spec[2, 1])
    _, _, _, _ = plot_spectrogram(df["acc_z"].values, ax=f_ax21, stft_params=stft_params, colorbar=colorbar)
    f_ax21.set_title("acc_z")
    f_ax31 = fig.add_subplot(spec[3, 1])
    _, _, _, _ = plot_spectrogram(df["acc_magnitude"].values, ax=f_ax31, stft_params=stft_params, colorbar=colorbar)
    f_ax31.set_title("acc_magnitude")
    f_ax4 = fig.add_subplot(spec[4:7, 0])
    f_ax4.plot(df["gyr_x"], label="gyr_x")
    f_ax4.plot(df["gyr_y"], label="gyr_y")
    f_ax4.plot(df["gyr_z"], label="gyr_z")
    f_ax4.legend()
    f_ax4.grid()
    f_ax7 = fig.add_subplot(spec[7, 0])
    f_ax7.plot(df["gyr_magnitude"])
    f_ax7.set_title("gyr_magnitude")
    f_ax41 = fig.add_subplot(spec[4, 1])
    _, _, _, _ = plot_spectrogram(df["gyr_x"].values, ax=f_ax41, stft_params=stft_params, colorbar=colorbar)
    f_ax41.set_title("gyr_x")
    f_ax51 = fig.add_subplot(spec[5, 1])
    _, _, _, _ = plot_spectrogram(df["gyr_y"].values, ax=f_ax51, stft_params=stft_params, colorbar=colorbar)
    f_ax51.set_title("gyr_y")
    f_ax61 = fig.add_subplot(spec[6, 1])
    _, _, _, _ = plot_spectrogram(df["gyr_z"].values, ax=f_ax61, stft_params=stft_params, colorbar=colorbar)
    f_ax61.set_title("gyr_z")
    f_ax71 = fig.add_subplot(spec[7, 1])
    _, _, _, _ = plot_spectrogram(df["gyr_magnitude"].values, ax=f_ax71, stft_params=stft_params, colorbar=colorbar)
    f_ax71.set_title("gyr_magnitude")
    if title is not None:
        fig.suptitle(title)
    if path_to_save is not None:
        save_fig(path_to_save)
        plt.close()
    else:
        plt.show()


def plot_instance_3d_scatterplot(df, title, ax=None, return_ax=False, path_to_save=None, **kwargs):
    figsize = kwargs.get("figsize", (10, 6))
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(df["acc_x"], df["acc_y"], df["acc_z"])
    ax.set_xlabel("acc X-axis")
    ax.set_ylabel("acc Y-axis")
    ax.set_zlabel("acc Z-axis")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(df["gyr_x"], df["gyr_y"], df["gyr_z"])
    ax.set_xlabel("gyr X-axis")
    ax.set_ylabel("gyr Y-axis")
    ax.set_zlabel("gyr Z-axis")
    if return_ax:
        return ax
    display_and_save_fig(path_to_save)


def plot_all_instances_3d_scatterplot(dfs_list, list_of_indexes, number_of_instances, title=None, path_to_save=None):
    columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
    list_of_indexes = random.sample(list_of_indexes, number_of_instances)
    gesture_df_all_instances = pd.DataFrame(columns=columns)
    for index in list_of_indexes:
        gesture_df_all_instances = gesture_df_all_instances.append(dfs_list[index])
    # print(gesture_df_all_instances)
    fig = plt.figure(figsize=(15, 7))
    plt.suptitle(title)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(gesture_df_all_instances["acc_x"], gesture_df_all_instances["acc_y"], gesture_df_all_instances["acc_z"])
    ax.set_xlabel("acc X-axis")
    ax.set_ylabel("acc Y-axis")
    ax.set_zlabel("acc Z-axis")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(gesture_df_all_instances["gyr_x"], gesture_df_all_instances["gyr_y"], gesture_df_all_instances["gyr_z"])
    ax.set_xlabel("gyr X-axis")
    ax.set_ylabel("gyr Y-axis")
    ax.set_zlabel("gyr Z-axis")
    plt.tight_layout()
    if path_to_save is None:
        plt.show()
    else:
        save_fig(path_to_save)
        plt.close()


def plot_motion_dataset_instances(input_instances, labels, n_samples=5, random_seed=42, path_to_save=None):
    """
    Plots a number of instances from each class.
    Args:
        input_instances (tuple or list of DataFrames): If Dataframe, it should have the common signals for motion as
        columns. The second case occurs when the dataset has already been transformed in a multidimensional matrix,
        hence the conversion to DataFrame should occur inside this function. The tupple has the
        following content: (<3-dimensional np.array>, <list of column names>).
        labels (list): List of integers
        n_samples (int): Number of samples
        random_seed (int): Random numbers generator seed.
        path_to_save (pathlib.Path or str):

    Returns:
        None
    """
    random.seed(random_seed)
    dataset_labels = project_configuration["processing"]["dataset_labels"]
    unique_labels = np.unique(labels)
    for label in unique_labels:
        class_indices = np.where(np.array(labels) == label)[0]
        # Get a random sample of these indices
        if len(class_indices) > n_samples:
            indices_to_plot = random.sample(range(0, len(class_indices)), n_samples)
        elif len(class_indices) > 0:
            indices_to_plot = class_indices
        else:
            indices_to_plot = list()
        for index_to_plot in indices_to_plot:
            dataset_index_to_plot = index_to_plot
            fig_name = "{}_{}".format(dataset_labels[label], dataset_index_to_plot)
            path_to_save_instance = None
            if path_to_save is not None:
                path_to_save_instance = Path(path_to_save).joinpath(fig_name)
            if isinstance(input_instances, list):
                # Case where input instances are already DataFrames
                data_df = input_instances[dataset_index_to_plot]
            elif isinstance(input_instances, tuple):
                data_df = pd.DataFrame(input_instances[0][dataset_index_to_plot],
                                       columns=input_instances[1])

            else:
                msg = "Saving plots of random instances of dataset failed. Unknown format of dataset."
                logging.error(msg)
                raise Exception(msg)
            plot_motion_sensor_df(data_df, title=fig_name, path_to_save=path_to_save_instance, figsize=(10, 10))


def plot_data_freq_dom(data, db=False, title=None, path_to_save=None, **kwargs):
    """

    Args:
        data (type): Description of parameter `data`.
        db (type): Description of parameter `db`.
        title (type): Description of parameter `title`.
        path_to_save (type): Description of parameter `path_to_save`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    rows = 0
    cols = 3
    nfft = kwargs.get("nfft", 128)
    fs = kwargs.get("sampling_frequency", 100.0)
    plotted_signals = list()
    for signal_name in data.columns:
        if data[signal_name].dtypes == np.float64:
            rows += 1
            plotted_signals.append(signal_name)
    fig, axs = plt.subplots(rows, cols, figsize=kwargs.get("figsize", (10, 10)), constrained_layout=True)
    for ind, signal_name in enumerate(plotted_signals):
        axs[ind, 0].plot(data[signal_name])
        axs[ind, 0].grid()
        axs[ind, 0].set_title(signal_name)
        fft_data = np.abs(np.fft.rfft(data[signal_name]))
        if db:
            fft_data = 20 * np.log10(fft_data/np.max(fft_data))
        freqs = np.fft.rfftfreq(len(data[signal_name]), d=1/fs)
        axs[ind, 1].plot(freqs, fft_data)
        ylim = np.max(fft_data[1:])
        axs[ind, 1].axvline(x=1, linewidth=1, color='r')
        axs[ind, 1].set_ylim([np.min(fft_data), ylim])
        axs[ind, 1].grid()
        axs[ind, 1].set_title("FFT (dBFS)" if db else "FFT")
        axs[ind, 1].set_xlabel("Frequency (Hz)")
        axs[ind, 1].set_ylabel("Amplitude (dBFS)" if db else "Amplitude")
        psd = axs[ind, 2].psd(data[signal_name], NFFT=nfft, Fs=fs)
        axs[ind, 2].axvline(x=1, linewidth=1, color='r')
        axs[ind, 2].set_title("Spectral power")
    if title is not None:
        fig.suptitle(title)
    if path_to_save is None:
        plt.show()
    else:
        save_fig(path_to_save)
        plt.close()
