"""
Plots regarding learning models training performance and outcomes
"""
import logging
import re
import matplotlib.pyplot as plt
import numpy as np
import itertools
from pathlib import Path


def plot_model_performance(performance_history_dict, path_to_save=None):
    """Plots a dictionary with the performance of a trained model. This function is hardcoded to depict only loss,
    val_loss, accuracy, val_accuracy..

    Args:
        performance_history_dict (dict): Dictionary with the performance history according to keras  history dictionary,
                                        created by the `keras.model.fit()` method.
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
                                        being depicted through `plt.show()`. Default None.

    Returns:
        None. Plots the input dictionary or saves it in the path, given as argument.
        #TODO make the function more dynamic (to plot the history without hardcoded names) if possible.
    """
    _, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(np.arange(0, len(performance_history_dict["loss"])),
                performance_history_dict["loss"],
                color="darkorange",
                lw=2,
                label="Training loss")
    axs[0].plot(np.arange(0, len(performance_history_dict["val_loss"])),
                performance_history_dict["val_loss"],
                color="royalblue",
                lw=2,
                label="Validation loss")
    axs[0].set_ylim([0, 1])
    axs[0].set_title("Loss per Epoch")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].yaxis.grid()
    axs[0].legend()

    # Accuracy graph.
    axs[1].plot(np.arange(0, len(performance_history_dict["accuracy"])),
                performance_history_dict["accuracy"],
                color="darkorange",
                lw=2,
                label="Training accuracy")
    axs[1].plot(np.arange(0, len(performance_history_dict["val_accuracy"])),
                performance_history_dict["val_accuracy"],
                color="royalblue",
                lw=2,
                label="Validation accuracy")
    axs[1].set_title("Accuracy per Epoch")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].yaxis.grid()
    axs[1].legend()
    if path_to_save is not None:
        plt.savefig(path_to_save)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm, labels, title=None, figsize=(8, 8), normalize=False, colorbar=False, path_to_save=None):
    """
    Plots a confusion matrix, with the corresponding labels.
    Args:
        cm (np.ndarray): The confusion matrix to plot as a two dimensional np.ndarray
        labels (list): A list of strings with the project defined labels for plotting in the confusion matrix.
        title (str, optional):
        figsize (tuple) :
        normalize (boolean, optional): If the exported matrix would be normalized. Defaults to False.
        colorbar (boolean, optional): If True, a colorbar would be depicted in the side of the confusion matrix.
        path_to_save (pathlib.Path or str, optional): When defined, the plot is saved on the given path instead of
                                        being depicted through `plt.show()`. Default None.

    Returns:
        None. Plots the given confusion matrix or saves it in the path, given as argument.
    """
    if normalize:
        cm = 100 * (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])
        fmt = ".1f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    if colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    if title is not None:
        plt.title(title, fontsize=14, fontweight="bold")

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
        try:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        except Exception as e:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="black")

    plt.tight_layout(pad=2)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    display_and_save_fig(path_to_save)


def plot_fit_history(history, key, figsize=(14, 6), path_to_save=None):
    fig = plt.figure(figsize=figsize)
    if key == "accuracy":
        # summarize history for accuracy
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="validation")
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
    elif key == "loss":
        # summarize history for loss
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="validation")
        plt.title("Model loss")
        plt.ylabel("Loss")
    elif key == "lr":
        # summarize history for loss
        plt.figure(figsize=(14, 6))
        plt.plot(history.history["lr"], label="Learning rate")
        plt.title("model learning rate")
        plt.ylabel("Learning Rate")
    else:
        msg = "Invalid key name for history object!"
        raise Exception(msg)
    plt.xlabel("epoch")
    plt.grid()
    plt.legend(loc="upper left")
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