import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import logging
from crossai.models.nn1d.base_model import BaseModel

from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D


class CNN2D(BaseModel):

    def define_model(self, config=None):

        self.arch.num_of_classes = config["number_of_classes"]
        self.arch.kernel_size = config["kernel"]
        self.arch.conv_layers = config["conv_layers"]
        self.arch.filters = config["filters"]
        self.arch.dense_layers = config["dense_layers"]
        self.arch.dense_units = config["dense_units"]
        self.arch.input_shape = config["dataset_shape"]
        self.arch.dropout_percent = config["dropout"]

        # Model build
        # ks: kernel size of inputs
        # wide path (features path)
        self.model = tf.keras.models.Sequential(name=self.model_name)
        # print("Conv1D input shape:", shaping)
        debug_msg_str = "model name : {}\n".format(self.model_name)
        debug_msg_str = debug_msg_str + "shaping : {}\n".format(self.arch.input_shape)
        debug_msg_str = debug_msg_str + "kernel : {}\n".format(self.arch.kernel_size)
        debug_msg_str = debug_msg_str + "filters : {}\n".format(self.arch.filters)
        debug_msg_str = debug_msg_str + "conv layers : {}\n".format(self.arch.conv_layers)
        debug_msg_str = debug_msg_str + "Dense units : {}\n".format(self.arch.dense_units)
        logging.debug(debug_msg_str)
        print(debug_msg_str)
        self.model.add(
            layers.Input(shape=self.arch.input_shape)
        )
        self.model.add(
            BatchNormalization()
        )
        for conv_l in range(0, self.arch.conv_layers):
            self.model.add(
                Conv2D(
                    filters=self.arch.filters[conv_l],
                    kernel_size=(self.arch.kernel_size, self.arch.kernel_size),
                    strides=(1, 1),
                    activation="relu",  # activation = 'relu'
                    padding="same"
                )
            )
        self.model.add(
            MaxPooling2D()
        )
        self.model.add(
            Dropout(self.arch.dropout_percent)
        )
        self.model.add(
            Flatten()
        )
        for d in range(0, self.arch.dense_layers):
            self.model.add(
                Dense(
                    self.arch.dense_units[d],
                    activation="relu"  # activation = "relu"
                )
            )
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.arch.num_of_classes, activation="softmax"))

    def fit(self, train_dataset, validation_dataset, verbose=None):
        """
        Trains the self.model.
        Args:
            train_dataset (tupple): Tupple of numpy arrays (x_train, y_train)
            validation_dataset(tupple): Tupple of numpy arrays (x_validation, y_validation)
            verbose:
        Returns: None

        """
        if verbose is None:
            verbose = 1 if logging.root.level == logging.DEBUG else 0

        epochs = self.epochs
        batch_size = self.batch_size
        steps_per_epoch = train_dataset[-1].shape[0] // batch_size
        logging.debug("Model.fit : Steps per epoch : {}".format(steps_per_epoch))
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        val_dataset = tf.data.Dataset.from_tensor_slices(validation_dataset). \
            shuffle(batch_size, reshuffle_each_iteration=True). \
            batch(batch_size)

        self.history = self.model.fit(train_dataset.repeat().batch(batch_size),
                                      epochs=epochs,
                                      steps_per_epoch=steps_per_epoch,
                                      callbacks=[] if not self.callbacks else self.callbacks_init(),
                                      verbose=verbose,
                                      validation_data=val_dataset
                                      )

        if "loss" in self.history.history.keys():
            self.performance_history["loss"] += self.history.history["loss"]
        if "val_loss" in self.history.history.keys():
            self.performance_history["val_loss"] += self.history.history["val_loss"]
        if "accuracy" in self.history.history.keys():
            self.performance_history["accuracy"] += self.history.history["accuracy"]
        if "val_accuracy" in self.history.history.keys():
            self.performance_history["val_accuracy"] += self.history.history["val_accuracy"]
