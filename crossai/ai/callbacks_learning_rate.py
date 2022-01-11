import numpy as np
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


def callback_lr_scheduler(config):
    """
    Learning Rate Scheduling: Calls the appropriate functionality or callback
    for the predefined scheduling task:
        a) tensorflow.keras.callbacks.LearningRateScheduler
        b) tensorflow.keras.callbacks.ReduceLROnPlateau

    Args:
        config: The configuration data.

    Returns:

    """
    if config["clb_lr_scheduler"]["mode"] == "exponential":
        scheduler_exp = exponential_decay(
            lr0=config["clb_lr_scheduler"]["exponential"]["lr0"],
            s=config["clb_lr_scheduler"]["exponential"]["steps"])
        clb = LearningRateScheduler(
            schedule=scheduler_exp, verbose=config["fit_params"]["verbose"])
    elif config["clb_lr_scheduler"]["mode"] == "piecewise":
        scheduler_pws = piecewise_constant(
            boundaries=config["clb_lr_scheduler"]["piecewise"]["boundaries"],
            values=config["clb_lr_scheduler"]["piecewise"]["values"])
        clb = LearningRateScheduler(
            schedule=scheduler_pws, verbose=config["fit_params"]["verbose"])
    elif config["clb_lr_scheduler"]["mode"] == "performance":
        clb = ReduceLROnPlateau(
            monitor=config["clb_lr_scheduler"]["clb_plateau"]["monitor"],
            factor=config["clb_lr_scheduler"]["clb_plateau"]["factor"],
            patience=config["clb_lr_scheduler"]["clb_plateau"]["patience"],
            min_lr=config["clb_lr_scheduler"]["clb_plateau"]["min_lr"])
    else:
        print("Not valid callback learning scheduler value/s is/are selected.")
        raise Exception

    return clb


def exponential_decay(lr0, s):
    """

    Args:
        lr0:
        s:

    Returns:

    """
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn


def piecewise_constant(config, boundaries, values):
    """

    Args:
        config:
        boundaries:
        values:

    Returns:

    """
    boundaries = np.array([0] + boundaries)
    values = np.array(values)
    if config["clb_lr_scheduler"]["piecewise"]["mode"] == "argmax":
        def piecewise_constant_fn(epoch):
            return values[np.argmax(boundaries > epoch) - 1]
    elif config["clb_lr_scheduler"]["piecewise"]["mode"] == "simple":

        def piecewise_constant_fn(epoch):
            if epoch < 5:
                return 0.01
            elif epoch < 15:
                return 0.005
            else:
                return 0.001
    else:
        print("Not valid piecewise constant mode value/s is/are selected.")
        raise Exception

    return piecewise_constant_fn
