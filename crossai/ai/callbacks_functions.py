import os
import logging
from crossai.ai.callbacks_learning_rate import callback_lr_scheduler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


def callbacks_list(config, early_stop=True, lr_scheduler=True,
                   checkpoint=True, tensorboard=True):
    """

    Args:
        config: The configuration data.
        early_stop: Define if adding the Early Stopping callback to the
        callbacks list.
        lr_scheduler: Define if adding the corresponding Learning Rate
        Scheduling callback to the callbacks list.
        checkpoint: Define if adding the Model Checkpoint callback to the
        callbacks list.
        tensorboard: Define if adding the Tensorboard callback to the
        callbacks list.

    Returns:
        The callbacks list.
    """
    clb_list = []
    if early_stop:
        clb_list.append(callback_early_stop(config))

    if lr_scheduler:
        clb_list.append(callback_lr_scheduler(config))

    if checkpoint:
        clb_list.append(callback_checkpoint(config))

    if tensorboard:
        tb_callback = callback_tensorboard(config)
        if tb_callback is not None:
            clb_list.append(tb_callback)

    return clb_list


def callback_early_stop(config):
    """

    Args:
        config: The configuration data.

    Returns:

    """
    return EarlyStopping(monitor=config["clb_early_stopping"]["monitor"],
                         patience=config["clb_early_stopping"]["patience"])


def callback_checkpoint(config):
    """

    Args:
        config:

    Returns:

    """
    if config["clb_checkpoint"]["path"] is None:
        os.makedirs(os.path.join(os.getcwd(), "model_checkpoints"),
                    exist_ok=True)
        file_path = os.path.join(os.getcwd(), "model_checkpoints")
    else:
        file_path = config["fit_params"]["checkpoint_path"]

    return ModelCheckpoint(filepath=file_path,
                           monitor=config["clb_checkpoint"]["monitor"],
                           save_best_only=config["clb_checkpoint"]
                           ["save_best_only"])


def callback_tensorboard(config):
    tensorboard_config = config.get("tensorboard", None)
    if tensorboard_config is None:
        msg = "Tensorboard configuration is missing from the " \
              "configuration.\nSkipping tensorboard callback setup."
        logging.warning(msg)
        return None
    else:
        log_dir = tensorboard_config.get("log_dir", None)
        if log_dir is None:
            msg = "Tensorboard configuration is missing from" \
                  " the configuration."
            raise Exception(msg)
        return TensorBoard(log_dir=log_dir)
