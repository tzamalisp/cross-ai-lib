from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay,\
    PiecewiseConstantDecay

lr_schedule_list = ["tf_exponential", "tf_piecewise"]


def learning_rate_scheduler(opt_schedule, exp_initial_lr=0.001,
                            exp_decay_stp=20, exp_decay_rt=0.1,
                            pcw_boundaries=[5, 15],
                            pcw_values=[0.01, 0.005, 0.001]):
    """

    Args:
        opt_schedule:
        exp_initial_lr:
        exp_decay_stp:
        exp_decay_rt:
        pcw_boundaries: (list)
        pcw_values: (list)

    Returns:

    """
    if opt_schedule == "tf_exponential":
        learning_rate = ExponentialDecay(initial_learning_rate=exp_initial_lr,
                                         decay_steps=exp_decay_stp,
                                         decay_rate=exp_decay_rt)
    elif opt_schedule == "tf_piecewise":
        learning_rate = PiecewiseConstantDecay(boundaries=pcw_boundaries,
                                               values=pcw_values)
    else:
        print("Not valid learning rate scheduler value/s is/are selected.")
        raise Exception

    return learning_rate


def get_lr_metric(optimizer):
    """

    Args:
        optimizer: The optimizer object.

    Returns:
        The learning rate value.
    """
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
