from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay,\
    PiecewiseConstantDecay

lr_schedule_list = ["tf_exponential", "tf_piecewise"]


def compile_model(model, optimizer="adam", lr_rate=0.001, adam_epsilon=1e-07,
                  loss="sparse_categorical_crossentropy", opt_schedule=None):
    """
    TODO: add the learning rate scheduler arguments

    Args:
        model: The created model topology.
        optimizer: (str) The optimizer name. Accepted values: "adam"
        loss: (str) The loss function.
        Accepted values: "sparse_categorical_crossentropy"
        lr_rate:
        adam_epsilon:
        opt_schedule:

    Returns:

    """
    # set Learning Rate
    if opt_schedule is not None and opt_schedule in lr_schedule_list:
        lr_rate = learning_rate_scheduler(opt_schedule=opt_schedule)
    else:
        lr_rate = lr_rate

    if optimizer == "adam":
        optimizer = Adam(learning_rate=lr_rate, epsilon=adam_epsilon)
    else:
        print("Please provide a valid optimizer.")
        raise Exception

    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[["accuracy"], get_lr_metric(optimizer)])

    lr_metric = get_lr_metric(optimizer)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[["accuracy"], lr_metric]
                  )

    return model


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
