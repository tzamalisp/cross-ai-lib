from tensorflow import keras
import logging


class MyModelCheckpoint(keras.callbacks.ModelCheckpoint):
    """
    Overloads tf.keras.callbacks.ModelCheckpoint and modifies it so it can
    call checkpoint depending on epochs run instead of data batches processed.
    Related to issue:
    [issue_link](https://github.com/tensorflow/tensorflow/issues/33163#issuecomment-829575078)
    Args:
        epoch_per_save (int): Number of epochs to elapse before calling checkpoint.
        *args (iterable): Positional arguments.
        **kwargs (iterable): keyword arguments.

    Attributes:
        epochs_per_save (int): Number of epochs to elapse before calling checkpoint.

    """

    def __init__(self, epoch_per_save=1, *args, **kwargs):
        logging.debug("MyModelCheckpoint called with epoch_per_save={}".format(epoch_per_save))
        self.epochs_per_save = epoch_per_save
        super().__init__(save_freq="epoch", *args, **kwargs)

    def on_epoch_end(self, epoch, logs):
        """
        Overloads `on_epoch_end` of super class.
        """

        if epoch % self.epochs_per_save == 0:
            super().on_epoch_end(epoch, logs)

