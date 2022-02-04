import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from crossai.models.nn1d.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv1D, \
    Dense, Dropout, Flatten, LeakyReLU, LSTM, \
    MaxPooling1D, TimeDistributed
from tensorboard.plugins.hparams import api as hp


class CNN1D(BaseModel):
    """
    Short summary.

    Parameters
    ----------
    model_name : String
        Name to save current model as. Default value is in the form
        nn_type_name+ "_" + str(kernel_size) + "_" + str(filter_size[0])
    parameters : dictionary
        Contains all the parameters for the training of the NN. Fields which should
        be in the dictionary are:
        * input_shape : tupple with the input shape of the data.
        * num_of_classes : integer which holds the number of classes to recognise
        * n_layers : Number of layers for the NN.
        * kernel_size : integer with the kernel size used in the NN
        * filters : Array which holds the filters which would be used in each of the NN
                    layers. Must have length equal to n_layers.
        * dense_units : Number of layers used in the dense units.



    Attributes
    ----------
    num_of_classes : int
        number of classes to recognise.
    kernel_size : int
        kernel size used in the NN.
    filters : array of integers
        filters which would be used in each of the NN layers.
    n_layers : int
        Number of layers for the NN.
    input_shape : tupple
        input shape of the data.
    run_logdir : string
        Directory where metadata and other info are stored during run.
    model : tensorflow model
        Model object which holds the information of the Neural Network. It holds the coefficients
        which would be trained and used later.


    """

    def read_cnn1d_configuration(self, config):
        self.num_of_classes = config["number_of_classes"]
        self.kernel_size = config["kernel"]
        self.conv_layers = config["conv_layers"]
        self.filters = config["filters"]
        self.dense_layers = config["dense_layers"]
        self.dense_units = config["dense_units"]
        self.input_shape = config["dataset_shape"]
        self.dropout_percent = config["dropout"]

        # tensorboard initialization

        # Model build
        # ks: kernel size of inputs
        # wide path (features path)
        self.model = tf.keras.models.Sequential(name=self.model_name)
        # print('Conv1D input shape:', shaping)
        debug_msg_str = "model name : " + self.model_name + "\n"
        debug_msg_str = debug_msg_str + "shaping : " + repr(
            (None, self.input_shape[0], self.input_shape[1])) + "\n"
        debug_msg_str = debug_msg_str + "kernel : " + repr(
            self.kernel_size) + "\n"
        debug_msg_str = debug_msg_str + "filters : " + repr(
            self.filters) + "\n"
        debug_msg_str = debug_msg_str + "n_cnn : " + repr(
            self.conv_layers) + "\n"
        debug_msg_str = debug_msg_str + "Dense units : " + repr(
            self.dense_units) + "\n"
        logging.debug(debug_msg_str)
        print(debug_msg_str)

    def define_model(self, config=None):
        self.read_cnn1d_configuration(config)

        # Input layer
        self.model.add(
            Conv1D(
                filters=self.filters[0],
                kernel_size=self.kernel_size,
                input_shape=self.input_shape,
                kernel_initializer="he_uniform"
            )
        )
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling1D(pool_size=4))

        # Additional CONV layers
        for conv_l in range(1, self.conv_layers):
            self.model.add(
                Conv1D(
                    filters=self.filters[conv_l],  # filters = 64
                    kernel_size=self.kernel_size,
                    kernel_initializer="he_uniform"
                )
            )
            self.model.add(BatchNormalization())
            self.model.add(Activation("relu"))
            self.model.add(MaxPooling1D(pool_size=4))

        self.model.add(keras.layers.Flatten())
        for d in range(0, self.dense_layers):
            self.model.add(
                keras.layers.Dense(
                    self.dense_units[d],
                    activation="relu"
                )
            )
            self.model.add(keras.layers.Dropout(self.dropout_percent))

        # Dense layer - Output layer
        self.model.add(keras.layers.Dense(self.num_of_classes,
                                          activation="softmax"))  # activation = softmax

    def freeze_layers(self, n):
        """Short summary.

        Parameters
        ----------
        n : type
            Description of parameter `n`.

        Returns
        -------
        type
            Description of returned object.

        """
        """
        n : the number of layers to freeze
        """
        for layer in self.model.layers[:n]:
            layer.trainable = False


class AppleModel(BaseModel):
    def define_model(self, config=None):
        self.num_of_classes = config["number_of_classes"]
        self.kernel_size = config["kernel"]
        self.conv_layers = config["conv_layers"]
        self.filters = config["filters"]
        self.dense_layers = config["dense_layers"]
        self.dense_units = config["dense_units"]
        self.input_shape = config["dataset_shape"]
        self.dropout_percent = config["dropout"]
        self.branch = config["branch"]

        debug_msg_str = "model name : " + self.model_name + "\n"
        debug_msg_str = debug_msg_str + "shaping : " + repr(
            (None, self.input_shape[0], self.input_shape[1])) + "\n"
        debug_msg_str = debug_msg_str + "kernel : " + repr(
            self.kernel_size) + "\n"
        debug_msg_str = debug_msg_str + "filters : " + repr(
            self.filters) + "\n"
        debug_msg_str = debug_msg_str + "n_cnn : " + repr(
            self.conv_layers) + "\n"
        debug_msg_str = debug_msg_str + "Dense units : " + repr(
            self.dense_units) + "\n"
        debug_msg_str = debug_msg_str + "Branch separation : " + repr(
            self.branch) + "\n"
        logging.debug(debug_msg_str)
        print(debug_msg_str)
        if self.branch == "per_signal":
            branch_shape = (self.input_shape[0], 1)
            number_of_inputs = self.input_shape[1]
        elif self.branch == "per_sensor":
            branch_shape = (self.input_shape[0], 3)
            number_of_inputs = self.input_shape[1] // 3
        # TODO discuss/decide how the framework would take action in case of
        # magnitudes in the input (Hence the model input would opt be dividable by 3).

        input_branches = list()
        model_branches = list()
        for i in range(0, number_of_inputs):
            branch = None
            input_branches.append(branch)
            input_branches[i] = keras.layers.Input(shape=branch_shape,
                                                   name="{}".format(i))
            model = None
            model_branches.append(model)
            # Create model branch on input
            model_branches[i] = keras.layers.Conv1D(filters=self.filters[0],
                                                    # This is always the first conv
                                                    kernel_size=self.kernel_size,
                                                    activation="relu",
                                                    # activation = 'relu'
                                                    input_shape=branch_shape,
                                                    padding="valid",
                                                    # oned_padding = 'valid' --> default
                                                    strides=1
                                                    # strides = 1 --> default
                                                    )(input_branches[i])
            model_branches[i] = keras.layers.MaxPooling1D(pool_size=4,
                                                          padding="valid")(
                model_branches[i])

        hidden0 = keras.layers.concatenate(model_branches)

        top = keras.layers.Flatten()(hidden0)
        for d in range(0, self.dense_layers):
            top = keras.layers.Dense(
                self.dense_units[d],
                activation="relu"  # activation = "relu"
            )(top)
        softmax_output = keras.layers.Dense(self.num_of_classes,
                                            activation="softmax",
                                            name="output")(top)

        self.model = keras.models.Model(
            inputs=input_branches,
            outputs=[softmax_output],
            name=self.model_name)

    def fit(self, train_dataset, validation_dataset, verbose=None):
        """
        Overlaods the fit method in order to separate the inputs in different
        branches.
        Args:
            train_dataset:
            validation_dataset:
            verbose(int, optional):

        Returns:

        """

        # super().fit(train_slices_tuple, validation_tuple, verbose)
        train_slices_tuple = transform_to_apple_model_dataset(train_dataset,
                                                              self.branch)
        validation_tuple = transform_to_apple_model_dataset(validation_dataset,
                                                            self.branch)
        print("x dataset train/val : {} , {}".format(
            train_slices_tuple[0][0].shape, validation_tuple[0][0].shape))
        print("x dataset train/val : {} , {}".format(
            train_slices_tuple[0][1].shape, validation_tuple[0][1].shape))
        print(
            "y dataset train/val : {} , {}".format(train_slices_tuple[1].shape,
                                                   validation_tuple[1].shape))
        x = [np.vstack([train_slices_tuple[0][i], validation_tuple[0][i]]) for
             i in
             range(0, len(train_slices_tuple[0]))]
        x = tuple(x)
        y = np.vstack([train_dataset[1], validation_dataset[1]])
        print("x shape: {}".format(x[0].shape))
        print("y shape: {}".format(y.shape))
        self.history = self.model.fit(x=x,
                                      y=y,
                                      epochs=self.epochs,
                                      # steps_per_epoch=steps_per_epoch,z
                                      callbacks=[] if not self.callbacks else self.callbacks_init(),
                                      verbose=verbose,
                                      validation_split=0.2
                                      )
        if "loss" in self.history.history.keys():
            self.performance_history["loss"] += self.history.history["loss"]
        if "val_loss" in self.history.history.keys():
            self.performance_history["val_loss"] += self.history.history[
                "val_loss"]
        if "accuracy" in self.history.history.keys():
            self.performance_history["accuracy"] += self.history.history[
                "accuracy"]
        if "val_accuracy" in self.history.history.keys():
            self.performance_history["val_accuracy"] += self.history.history[
                "val_accuracy"]

    def evaluate(self, test_X, test_y):
        """
        Overloads evaluate to use apple model data transformation
        Args:
            test_X:
            test_y:

        Returns:

        """
        batch_size = self.batch_size
        logging.debug("Evaluate test_X shape : {}".format(test_X.shape))
        test_tuple = transform_to_apple_model_dataset((test_X, test_y),
                                                      self.branch)
        eval_results = self.model.evaluate(x=test_tuple[0], y=test_tuple[1],
                                           batch_size=batch_size,
                                           verbose=1 if logging.root.level == logging.DEBUG else 0)
        for ind, key in enumerate(self.model.metrics_names):
            self.evaluation_results["eval_" + key] = eval_results[ind]

        # Create the common filename to save the model exports
        self.exports_common_name = "{0:.3f}_{1}".format(
            self.evaluation_results["eval_accuracy"], self.model_name)

        return self.evaluation_results

    def predict(self, data, labels=None):
        """
        Overloads data to use apple model data
        Args:
            data (numpy.ndarray or pandas.DataFrame):
            labels (list): List of text, that contains the names of the predicted classes.

        Returns:
            DataFrame with the predictions. The DataFrame columns are the same number with the classes number
                of the model.
        """
        logging.debug("Model predictions..")
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = transform_to_apple_model_dataset(data, self.branch)
        predictions = self.model.predict(data)
        pd.options.display.float_format = "{:,.3f}".format
        predictions = pd.DataFrame(predictions, columns=labels)
        return predictions


def transform_to_apple_model_dataset(data, branch_config):
    """

    Args:
        data:
        branch_config:

    Returns:

    """
    if isinstance(data, tuple):
        if branch_config == "per_signal":
            data_slices = [data[0][:, :, i] for i in
                           range(0, data[0].shape[2])]
        elif branch_config == "per_sensor":
            data_slices = [data[0][:, :, i:i + 3] for i in
                           range(0, data[0].shape[2], 3)]
        else:
            msg = "Unknown branch split configuration {}".format(branch_config)
            logging.error(msg)
            raise Exception(msg)
        data_slices = tuple(data_slices)
        data_slices_tuple = (data_slices, data[1])
    else:
        if branch_config == "per_signal":
            data_slices = [data[:, :, i] for i in range(0, data.shape[2])]
        elif branch_config == "per_sensor":
            data_slices = [data[:, :, i:i + 3] for i in
                           range(0, data.shape[2], 3)]
        else:
            msg = "Unknown branch split configuration {}".format(branch_config)
            logging.error(msg)
            raise Exception(msg)
        data_slices_tuple = tuple(data_slices)
    return data_slices_tuple


class Multikernel(CNN1D):
    def define_model(self, config=None):
        self.read_cnn1d_configuration(config)

        self.n_branches = len(self.kernel_size)
        print("n_branches : {}".format(self.n_branches))
        input_branch = keras.layers.Input(shape=self.input_shape, name="input")
        model_branches = list()
        for i in range(0, self.n_branches):
            model = None
            model_branches.append(model)
            # Create model branch on input
            model_branches[i] = keras.layers.Conv1D(filters=self.filters[0],
                                                    # This is always the first conv
                                                    kernel_size=
                                                    self.kernel_size[i],
                                                    activation="relu",
                                                    # activation = 'relu'
                                                    input_shape=self.input_shape,
                                                    padding="valid",
                                                    # oned_padding = 'valid' --> default
                                                    strides=1
                                                    # strides = 1 --> default
                                                    )(input_branch)
            model_branches[i] = keras.layers.MaxPooling1D(pool_size=2,
                                                          padding="valid")(
                model_branches[i])
            model_branches[i] = keras.layers.Flatten()(model_branches[i])

            # Concatenated layer
        top = keras.layers.concatenate(model_branches)

        for d in range(0, self.dense_layers):
            top = keras.layers.Dense(
                self.dense_units[d],
                activation="relu"  # activation = "relu"
            )(top)
        softmax_output = keras.layers.Dense(self.num_of_classes,
                                            activation="softmax",
                                            name="output")(top)

        self.model = keras.models.Model(
            inputs=input_branch,
            outputs=[softmax_output],
            name=self.model_name)




