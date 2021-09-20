"""
Functions for vizualizing and saving figures, using matplotlib and seaborn.
"""
# TODO discuss a proper way of using savefig and finalize the savefig arguments.

import logging
import re

import numpy as np
import seaborn as sns
from PIL import Image
from scipy import stats
from pathlib import Path
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from utilities.files_handling import load_csv_data
from configuration_functions.configurations import project_configuration
CURR_DIR = Path(__file__).parent.absolute()
plt.style.use(CURR_DIR.joinpath("publication_visualization_params.mplstyle"))

"""
List of standard colors to always use for the visualization of each of the
various labels at each project. Used to maintain the same color for each label
whenever it is plotted.
"""
LABELS_COLORS = ["#ff9900", "#66ff66", "#ffccff", "#009900", "#996633",
                 "#666699", "#00ffff", "#ff3300", "#ffff00", "#6600cc",
                 "#990033", "#006666", "#003300", "#ff6666", "#0000ff",
                 "#99ffcc"]

LABELS_COLORS_QUALITATIVE = sns.color_palette("tab10", 26)


def number_of_instances_barplot(instances_counts, title=None, ax=None, xlabel=None, ylabel=None, path_to_save=None,
                                **kwargs):
    """
    Creates a barplot with the items of the `instances_counts` dictionary.

    Args:
        y_label: (str, optional) The y_axis label
        x_label: (str, optional) The x_axis label
        instances_counts (dict): Dict in the format {<label_name>: <integer>}
        title (str, optional): When defined, a title is added to the exported plot. Default None.
        ax (matplotlib.axe, optional): When defined, the plot is created in the given axe. Default None.
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
                        being depicted through `plt.show()`. Default None.

    Returns:
        None. Plots a barplot or saves it in the path, given as argument.
    """
    labels = list(instances_counts.keys())
    labels.sort()
    figsize = kwargs.get("figsize", (16, 4))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(hspace=1, bottom=0.15, top=0.85)
    x = np.linspace(0, np.ceil(1.5 * len(labels)), len(labels))  # the xlabel locations, placed sparsely.
    counts = [instances_counts[label] for label in labels]
    bars = ax.bar(x, counts, width=1, color="#1f77b4", edgecolor="black", linewidth="1.0")
    ax.grid(axis="y")
    ax.bar_label(bars, padding=2, fontsize=6, fontweight="normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if len(labels) > 10:
        plt.xticks(rotation="vertical")
    elif len(labels) > 5:
        rotation = kwargs.get("rotation", "45")
        plt.xticks(rotation=rotation)
    else:
        plt.xticks(rotation=0)
    if title:
        ax.set_title(title)
    display_and_save_fig(path_to_save)


def number_of_instances_pie(instances_counts, title=None, ax=None, path_to_save=None, **kwargs):
    """
        Creates a piechart with the items of the `instances_counts` dictionary. The plot shows percentage.
        The percentage is calculated in reference to the sum of the dictionary values.

        Args:
            instances_counts (dict): Dict in the format {<label_name>: <integer>}
            title (str, optional): When defined, a title is added to the exported plot. Default None.
            ax (matplotlib.axe, optional): When defined, the plot is created in the given axe. Default None.
            path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead
                            of being depicted through `plt.show()`. Default None.

        Returns:
            None. Plots a barplot or saves it in the path, given as argument.
    """
    labels = list(instances_counts.keys())
    labels.sort()
    figsize = kwargs.get("figsize", (5, 5))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(left=0.3, right=0.7, bottom=0.2, top=0.8, hspace=1, wspace=2.0)
    counts = [instances_counts[label] for label in labels]
    counts_percentages = counts / np.sum(counts) * 100
    wedges, _ = ax.pie(counts_percentages,
                       colors=LABELS_COLORS_QUALITATIVE[:len(labels)],
                       startangle=90 if len(labels) < 4 else 0, textprops={"fontsize": 2}
                       )
    bbox_props = dict(boxstyle="square,pad=0.25", fc="w", ec="k", lw=0.2)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate("{0} : {1:.2f}%".format(labels[i], counts_percentages[i]),
                    xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)
    if title:
        ax.set_title(title, y=1.08)
    display_and_save_fig(path_to_save)


def plot_hist(data, title=None, ax=None, color=None, xlabel=None, ylabel=None, bins=30, path_to_save=None, **kwargs):
    """

    Args:
        data (list or np.ndarray):
        title (str, optional): The figure title
        ax (matplotlib.ax, optional): If provided, the plot would be depicted in the given axe, else a new plot
                                        would be created.
        color (str or list of strings, optional ): The color(s) to use for the histogram plot.
        xlabel (str, optional): If defined, it would be used as a description to the histogram.
        ylabel (str, optional): If defined, it would be used as a description to the histogram.
        bins (int, optional): bins to use for the histogram. Defaults to 10.
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
                                        being depicted through `plt.show()`. Default None.
        **kwargs:
            kde (boolean)
    Returns:
        None. Plots a histogram or saves it in the path, given as argument.
    """
    create_plot_in_function = False  # This variable handles whether to call plt.show() from inside
    # this function or the show() is called from the caller function. It depends on whether the ax
    # argument is None or not
    figsize = kwargs.get("figsize", (10, 5))
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        create_plot_in_function = True
    kde = kwargs.get("kde", True)

    n, bins_centers, patches = ax.hist(data,
                                       bins=bins,
                                       color=color,
                                       edgecolor="black",
                                       linewidth="1.0")
    ax.set_ylabel("Counts", rotation=90, labelpad=1)
    # print("{} : n : {}\n {}".format(title, n, len(bins_centers)))

    if kde:
        try:
            kde = stats.gaussian_kde(data)
            length = np.max(data)
            xx = np.linspace(0, length, length * bins)
            ax2 = ax.twinx()
            ax2.plot(xx,  kde(xx), color="orange", linewidth=3, label="KDE")
            kde_max = np.max(kde(xx))
            ax2.tick_params(axis='y', labelrotation=270, pad=3)
            ax2.set_ylabel("Density", rotation=270, labelpad=10)
            ax2.set_ylim([0, kde_max + kde_max/5])
            f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
            g = lambda x, pos: "${}$".format(("%1.1e" % x) if x !=0 else "0")
            scientific_formatter = FuncFormatter(g)
            yticks_ax2 = [0, np.max(kde(xx))]
            ax2.set_yticks(yticks_ax2)
            ax2.yaxis.set_major_formatter(scientific_formatter)
            ax2.legend()
        except ValueError as e:
            msg = "{} : {}".format(title, e)
            logging.warning(msg)
    nonzero_bins = np.nonzero(np.array(n))[0]
    # print("Nonzero bins : {}".format(nonzero_bins))
    if len(data) > 1:
        ax.set_xlim([np.min(data) - 2, np.max(data) + np.max(data) / 50])
    ax.set_xscale("linear")
    if len(nonzero_bins) <= 5:
        xticks = [np.floor(bins_centers[i]) for i in nonzero_bins]
    elif len(bins_centers) <= 10:
        xticks = [np.floor(bins_centers[i]) for i in range(0, len(bins_centers))]
    else:
        xticks = [np.floor(bins_centers[i]) for i in range(0, len(bins_centers), 3)]
    ax.set_xticks(xticks)
    ax.set_ylabel("Counts", rotation=90, labelpad=1)
    ax.tick_params(axis='y', labelrotation=90)
    ax.grid("y")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, horizontalalignment="center")
    if create_plot_in_function:
        display_and_save_fig(path_to_save)


def plot_distribution_of_data(instances_dict, description=None, title=None, xlabel=None, ylabel=None, bins=50,
                              path_to_save=None, **kwargs):
    """
    Creates a histogram for each of the input dictionary keys. Even in case of only one instance, the instances_dict
    should be in a dictionary format.
    Args:
        instances_dict (dict): Dict in the format {<label_name>: <list of values>}
        description: If defined, it would be used as a description in the xlabel of the histogram.
        title (str, optional): The figure title
        bins (int, optional): bins to use for the histogram. Defaults to 10.
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
                                        being depicted through `plt.show()`. Default None.
        **kwargs:
            kde (bool, optional): Add Kernel density distribution line on the plot. Default is True.

    Returns:
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
                                        being depicted through `plt.show()`. Default None.
    """
    fig_rows = 1
    fig_cols = 1
    kde = kwargs.get("kde", True)
    figsize = kwargs.get("figsize", (12, 4))
    if isinstance(instances_dict, dict):
        classes_names = list(instances_dict.keys())
        classes_names.sort()
        fig_rows = len(classes_names) // 2 if len(classes_names) >= 2 else len(classes_names)
        fig_cols = len(classes_names) // fig_rows
    else:
        error_message = "Unknown format of passed data for distribution plotting!"
        logging.error(error_message)
        raise Exception(error_message)
    if len(classes_names) > 1:
        fig, ax = plt.subplots(fig_rows, fig_cols, figsize=figsize)
        fig.tight_layout(pad=4.5, w_pad=3, h_pad=4.5, rect=(0, 0.05, 1, 0.97))
        for ind, (class_name, ax_i) in enumerate(zip(classes_names, ax.flatten())):
            data = instances_dict[class_name]
            color = "#1f77b4"
            plot_hist(data, title=class_name, color=color, xlabel=xlabel, ylabel=ylabel, ax=ax_i, bins=bins, kde=kde)
    else:
        fig, ax = plt.subplots(fig_rows, fig_cols, figsize=figsize)
        data = instances_dict[classes_names[0]]
        color = "#1f77b4"
        plot_hist(data, title=classes_names[0], xlabel=description, ylabel=ylabel, color=color, ax=ax, bins=bins,
                  kde=kde)
    if title:
        fig.suptitle(title, horizontalalignment="center")
    display_and_save_fig(path_to_save)


def display_and_save_fig(path_to_save):
    if path_to_save:
        save_fig(path_to_save)
    plt.show()
    plt.close()


def save_fig(fig_id, fig_extension="png", resolution=500):
    """
    Create and save a figure to the relevant path.

    Args:
        fig_id: (str) The name of the image.
        fig_extension: (str) The extension of the image file. (Default: "png")
        resolution: (int) The resolution in dpi. (Default: 300)

    Returns:
    """
    if isinstance(fig_id, Path):
        fig_id = str(fig_id)
    # Detect if the fig_id already contains a filetype extension
    extension_re = r"\.\w{3,4}$"
    if not re.search(extension_re, fig_id):
        fig_id += "." + fig_extension
    logging.debug("Saving figure: {}".format(fig_id))
    fig = plt.gcf()
    fig.savefig(fig_id, format=fig_extension, dpi=resolution)


def load_fig(path_to_figure, figsize=(15, 15), display=True):
    """

    Args:
        figsize: figure size in tuple
        path_to_figure (str or pathlib.Path):
        display (boolean): Whether to display the image or just return it
    Returns:

    """
    im = Image.open(path_to_figure)
    # show image
    if display:
        plt.figure(figsize=figsize)
        imshow(np.asarray(im))
    return im


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    #     print("MAKING IMAGE AND BUFFER DATA")
    # draw the renderer
    fig.canvas.draw()

    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)  # 3 because there is a RGB channels mode and not RGB
    #     print("W:", w)
    #     print("H:", h)
    #     print("BUFFER SHAPE:", buf.shape)

    # canvas.tostring_rgb gives pixmap in RGB mode.
    buf = np.roll(buf, 3, axis=2)
    #     print("BUF AFTER ROLL:", buf.shape)
    return buf


def fig2img(fig, fig_export):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param
    @return a Python Imaging Library ( PIL ) image
    :param fig:  a matplotlib figure
    :param fig_export: Boolean, if True, a PIL Image would be returned
    :return:
        buf : nparray corresponding to the image from matplotlib
        image: if fig_export is True a PIL.Image object else None
        w,h,d: Dimensions of exported ndarray
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    if fig_export is True:
        image = Image.frombytes("RGB", (w, h), buf.tostring())
    else:
        image = None
    return buf, image, w, h, d


def merge_pil_images(pil_images_sprectograms_list):
    imgs = [i for i in pil_images_sprectograms_list]

    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]

    # imgs_comb = list()
    # imgs_comb.append([np.asarray(i.resize(min_shape)) for i in imgs])
    imgs_comb = list()
    for i in imgs:
        imgs_comb.append(np.asarray(i.resize(min_shape)))
    imgs_comb = np.vstack(imgs_comb)
    # imgs_comb = np.array(imgs_comb)

    # save that beautiful picture
    imgs_comb = Image.fromarray(imgs_comb)
    #     imgs_comb.show()
    return imgs_comb


def boxplot_statistical_data(data, title=None, ax=None, path_to_save=None, **kwargs):
    """

    Args:
        path_to_save:
        ax:
        data(DataFrame or numpy.array): Array or dataframe to plot.
        title (str, optional): When defined, a title is added to the exported plot. Default None.
            ax (matplotlib.axe, optional): When defined, the plot is created in the given axe. Default None.
            path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead
                            of being depicted through `plt.show()`. Default None.
        *args (list): Access positional arguments
        **kwargs (list): Access categorical arguments. Can be used to specify various parameters for the
            plots such as figsize.

    Returns:

    """
    create_plot_in_function = False
    if ax is None:
        figsize = kwargs.get("figsize", (6, 4))
        _, ax = plt.subplots(figsize=figsize)
        create_plot_in_function = True
    data_descriptions = data.columns
    xticks = np.arange(len(data_descriptions)) + 1
    ax.boxplot(data.values)
    ax.set_xticks(xticks)
    ax.set_xticklabels(data_descriptions)
    if title:
        ax.set_title(title)
    if create_plot_in_function:
        if path_to_save:
            plt.savefig(path_to_save)
            plt.close()
        else:
            plt.show()


def plot_null_values(df, title=None, path_to_save=None):
    plt.figure(figsize=(16, 5))
    plt.title(title)
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    if path_to_save:
        plt.savefig(path_to_save)
        plt.close()
    else:
        plt.show()


def plot_scatter(data_list, xlabel, ylabel, zlabel=None, title=None, path_to_save=None, **kwargs):
    # TODO add documentation
    # TODO add check that list contains only two arrays of data
    figsize = kwargs.get("figsize", (14, 9))
    c = kwargs.get("c", None)
    cmap = kwargs.get("cmap", "plasma")
    plt.style.use("classic")
    fig = plt.figure(figsize=figsize)
    if len(data_list) == 2:
        ax = plt.gca()
        ax.scatter(data_list[0], data_list[1], c=c, cmap=cmap)
    elif len(data_list) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(data_list[0], data_list[1], data_list[2], c=c, cmap=cmap)
    else:
        msg = "Invalid data dimensions provided to create scatterplot. Data arrays found : {}".format(len(data_list))
        logging.error(msg)
        raise Exception(msg)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(data_list) == 3:
        if zlabel is not None:
            ax.set_zlabel(zlabel)
        else:
            logging.warning("3-Dimensional data provided, but no description for zlabel.")
    if title is not None:
        plt.title(title)
    if path_to_save is not None:
        save_fig(path_to_save)
        plt.close()
    else:
        plt.show()


def plot_dataframe_heatmap(data, title=None, path_to_save=None, **kwargs):
    figsize = kwargs.get("figsize", (24, 10))
    cmap = kwargs.get("cmap", "plasma")
    plt.figure(figsize=figsize)
    sns.heatmap(data, cmap=cmap)
    if title is not None:
        plt.title(title)
    if path_to_save is not None:
        save_fig(path_to_save)
        plt.close()
    else:
        plt.show()


def plot_all_ml_model_scores(path_to_csv):
    logging.info("Plotting all ml models' accuracy scores..")
    if path_to_csv.is_file():
        plt.rcParams["axes.labelsize"] = "x-large"
        plt.rcParams["xtick.labelsize"] = "medium"
        plt.rcParams["ytick.labelsize"] = "medium"
        scores_df = load_csv_data(
            "{}".format(Path(project_configuration["project_store_path"]).joinpath("reports/all_scores.csv")))
        scores_df.plot(x='model', y='score', kind='bar', figsize=(20, 12), color='#1f77b4',
                       edgecolor=(0, 0, 0), linewidth="1.0")
        plt.xticks(rotation=25)
        plt.savefig(Path(project_configuration["project_store_path"]).joinpath("reports/all_model_scores"))
    else:
        logging.error("The csv file does not exist!")


def plot_dataframe(df_plot, signals, num, title=None):
    """
    Plots the argument signals of the Dataframe. Used in jupyterlab noteboooks
    for interactively plotting the available signals.
    Args:
        df_plot (pandas DataFrame):
        signals (list): List of strings. Strings should be columns of the dataframe
        num (int or str): The identifier of the fgure that is created.

    Returns:

    """
    plt.figure(num=num)
    plt.clf()
    for sig_name in list(signals):
        plt.plot(df_plot[sig_name], label=sig_name)
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.legend()
