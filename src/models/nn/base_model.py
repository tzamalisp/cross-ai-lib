"""

"""
import time
import json
import logging
import numpy as np
import pandas as pd
from bunch import Bunch
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
from sklearn.metrics import classification_report, confusion_matrix
from training.project_training_variables import nn_default_parameters
from training.models_performance import plot_model_performance, plot_confusion_matrix
from configuration_functions.project_configuration_variables import project_configuration


class BaseModel:
    """
    Creates a BaseModel object, and it gives access to several functionalities.
    """
    def __init__(self, config, model_name=None, callbacks=False):
        """
        Constructor to initialize the model's architecture parameters.
        Args:
            config: the JSON configuration.
            model_name (str or None, optional): The model name. If not defined,
             current datetime is used
            callbacks (boolean): if callbacks will be used.
        """
        self.model_name = datetime.now().strftime("%Y-%m-%d-%H-%M") if\
            model_name is None else model_name
        self.hparams = dict()

        self.path_to_save_model = \
            Path(project_configuration["project_store_path"]).joinpath("tf")

        if not self.path_to_save_model.exists():
            logging.debug("Creating directory for saving model.".
                          format(self.path_to_save_model.relative_to(
                project_configuration["project_store_path"])))
            self.path_to_save_model.mkdir(parents=True, exist_ok=True)

        set_name = self.model_name + ".tf"
        self.path_to_save_model = self.path_to_save_model.joinpath(set_name)
        self.path_to_save_model.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.config["path_to_save_model"] = self.path_to_save_model

        self.exports_common_name = None


        log_dir = self.path_to_save_model
        self.path_to_tf_logs = log_dir

        logging.debug("Path to TF logs (usage with tensorboard)"
                      " : {}".format(self.path_to_tf_logs))

        self.callbacks = callbacks
        self.optimizer = None
        if config.get("optimizer"):
            self.optimizer = config["optimizer"]
            HP_OPTIMIZER = hp.HParam("optimizer",
                                     hp.Discrete([config["optimizer"]]))
            self.hparams[HP_OPTIMIZER] = config["optimizer"]
        self.learning_rate = None
        if config.get("learning_rate"):
            self.learning_rate = config["learning_rate"]
            HP_LEARNING_RATE = hp.HParam("learning_rate",
                                         hp.Discrete([config["learning_rate"]])
                                        )
            self.hparams[HP_LEARNING_RATE] = HP_LEARNING_RATE.domain.values[0]
        self.epochs = config.get("epochs", 0)
        self.batch_size = None
        if config.get("batch_size"):
            self.batch_size = config["batch_size"]
            HP_BATCH_SIZE = hp.HParam("batch_size",
                                      hp.Discrete([config["batch_size"]]))
            self.hparams[HP_BATCH_SIZE] = config["batch_size"]
        self.nn_metrics = nn_default_parameters["nn_metrics"] if "nn_metrics"\
            not in config.keys() else config["nn_metrics"]
        self.nn_loss_function = nn_default_parameters["nn_loss_function"] if\
            "nn_loss_function" not in config.keys() else\
            config["nn_loss_function"]

        # Initialize a model
        self.model = None
        self.iters_per_epoch = None
        # Default predictions behaviour. By default the MonteCarlo predictions
        # is deactivated.
        self.mcdropout = False
        self.mc_iterations = 1
        # Initialize NN architecture parameters. Set by the calling of define
        # model for each model arch.
        self.arch = Bunch()

        # Training time.
        self.train_time = 0

        # Predicted class labels.
        self.predictions = np.array([])

        # Initialize performance metrics. In these arrays the result of
        # history would be stored (from tensorflow.fit function). This
        # additional dictionary is used because history from fit is writting
        # data per epoch. Thus in case fit iscalled again (e.g. in case of
        # k-fold, only the last performance is viewed)
        # TODO automate the creation of the dictionary according to any metric
        # that wll be specified in the configuration file.
        self.history = None
        self.history = dict()
        # self.performance_history["loss"] = []
        # self.performance_history["accuracy"] = []
        # self.performance_history["val_loss"] = []
        # self.performance_history["val_accuracy"] = []
        self.evaluation_results = dict()
        return

    def calculate_number_of_filters(self):
        """
        Calculates the filter size for a given layer.
        Returns:

        """

        # Implement this method in the inherited class to calculate the filter
        # size.
        raise NotImplementedError

    def callbacks_init(self):
        """
        Initializes the callbacks of training with parameters from the
        configuration.
        Returns:
            callbacks_list (list): A list with callbacks.
        """
        callbacks_list = []
        calbacks_configuration = project_configuration.get("callbacks")
        if calbacks_configuration["early_stopping"]["active"]:
            es_monitor = calbacks_configuration["early_stopping"]["monitor"]
            es_mode = calbacks_configuration["early_stopping"]["mode"]
            es_patience = calbacks_configuration["early_stopping"]["patience"]
            callback_early_stopping = \
                keras.callbacks.EarlyStopping(monitor=es_monitor,
                                              mode=es_mode,
                                              patience=es_patience,
                                              verbose=1 if logging.root.level
                                                           == logging.DEBUG
                                                        else 0,
                                            restore_best_weights=True)
            callbacks_list.append(callback_early_stopping)

        if calbacks_configuration["checkpoint"]["active"]:
            checkpoint_monitor = \
                calbacks_configuration["checkpoint"]["monitor"]
            checkpoint_mode = calbacks_configuration["checkpoint"]["mode"]
            checkpoint_filepath =\
                Path(project_configuration["project_store_path"]). \
                joinpath("tf").joinpath("{}.tf".format(self.model_name))
            if not checkpoint_filepath.is_dir():
                checkpoint_filepath.mkdir(parents=True, exist_ok=True)
            save_freq = calbacks_configuration["checkpoint"]["save_freq"]
            logging.debug("Save freq checkpoint: {}".format(save_freq))
            callback_checkpoint = \
                MyModelCheckpoint(epoch_per_save=save_freq,
                                  filepath=str(checkpoint_filepath),
                                  monitor=checkpoint_monitor,
                                  verbose=1 if logging.root.level ==
                                               logging.DEBUG else 0,
                                  save_best_only=True,
                                  mode=checkpoint_mode
                                                    )
            callbacks_list.append(callback_checkpoint)
        if calbacks_configuration["tensorboard"]["active"]:
            tensorboard_callback = tf.keras.callbacks.\
                TensorBoard(log_dir=self.path_to_tf_logs)
            callbacks_list.append(tensorboard_callback)
        return callbacks_list

    def define_model(self, config=None):
        """
        Constructs the model.
        Args:
            config: configuration

        Returns:

        """
        # Implement this method in the inherited class to add layers to the
        # model.
        raise NotImplementedError

    def compile_model(self, optimizer=None, learning_rate=None, loss=None,
                      nn_metrics=None):
        """
        If optimizer and learning_rate are given as arguments, then they are
        used instead of the defaults given in the configuration. Useful if
        function is called from outside the main script (e.g. a Notebook)
        Args:
            optimizer: Keras optimizer
            learning_rate: float
            loss: One of the keras loss functions
            nn_metrics: Keras metrics for the performance of the trained model
        """

        if optimizer is None:
            optimizer = self.optimizer
        assert (optimizer in ["adam", "sgd"])

        # define learning_rate
        learning_rate = learning_rate if learning_rate is not None\
            else self.learning_rate
        if optimizer == "sgd":
            optimizer = keras.optimizers.SGD(lr=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(lr=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self.nn_loss_function if loss is None else loss,
            metrics=self.nn_metrics if nn_metrics is None else nn_metrics,
        )

    def fit(self, train_dataset, validation_dataset, verbose=None):
        """
        Trains the self.model.
        Args:
            train_dataset (tupple): Tupple of numpy arrays (x_train, y_train)
            validation_dataset(tupple): Tupple of numpy arrays (x_validation,
            y_validation)
            verbose:
        Returns: None

        """
        if verbose is None:
            verbose = 1 if logging.root.level == logging.DEBUG else 0

        epochs = self.epochs
        batch_size = self.batch_size
        steps_per_epoch = train_dataset[-1].shape[0] // batch_size
        logging.debug("Model.fit : Steps per epoch :"
                      " {}".format(steps_per_epoch))
        # train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        # val_dataset = tf.data.Dataset.
        # from_tensor_slices(validation_dataset).\
        #     shuffle(batch_size, reshuffle_each_iteration=True).\
        #     batch(batch_size)

        self.history = self.model.fit(x=np.vstack((train_dataset[0],
                                                   validation_dataset[0])),
                                      y=np.vstack((train_dataset[1],
                                                   validation_dataset[1])),
                                      epochs=epochs,
                                      # steps_per_epoch=steps_per_epoch,z
                                      callbacks=[] if not self.callbacks
                                                   else self.callbacks_init(),
                                      verbose=verbose,
                                      validation_split=0.2
                                      )
        # if "loss" in self.history.history.keys():
        #     self.performance_history["loss"] += self.history.history["loss"]
        # if "val_loss" in self.history.history.keys():
        #     self.performance_history["val_loss"] +=
        #     self.history.history["val_loss"]
        # if "accuracy" in self.history.history.keys():
        #     self.performance_history["accuracy"] +=
        #     self.history.history["accuracy"]
        # if "val_accuracy" in self.history.history.keys():
        #     self.performance_history["val_accuracy"] += self.history.
        #     history["val_accuracy"]

    def evaluate(self, test_X, test_y, **kwargs):
        """
        Evaluates the dataset according to testset.
        Args:
            test_X:
            test_y:
            **kwargs: keras evasluate arguments
        Returns: a dictionary with the performance results.

        """
        batch_size = kwargs.get("batch_size", self.batch_size)
        verbose = kwargs.get("verbose",
                             (1 if logging.root.level == logging.DEBUG else 0))
        if kwargs.get("batch_size"):
            _ = kwargs.pop("batch_size")
        if kwargs.get("verbose"):
            _ = kwargs.pop("verbose")
        logging.debug("Evaluate test_X shape : {}".format(test_X.shape))

        eval_results = self.model.evaluate(test_X, test_y,
                                           batch_size=batch_size,
                                           verbose=verbose,
                                           **kwargs)
        for ind, key in enumerate(self.model.metrics_names):
            self.evaluation_results["eval_" + key] = eval_results[ind]

        # Create the common filename to save the model exports
        self.exports_common_name =\
            "{0:.3f}_{1}".format(self.evaluation_results["eval_accuracy"],
                                                        self.model_name)

        return self.evaluation_results

    def predict(self, data, labels=None):
        """
        Predicts the class labels of unknown data.
        Args:
            data (numpy.ndarray or pandas.DataFrame):
            labels (list): List of text, that contains the names of the
            predicted classes.

        Returns:
            DataFrame with the predictions. The DataFrame columns are the same
            number with the classes number of the model.
        """
        logging.debug("Model predictions..")
        if isinstance(data, pd.DataFrame):
            data = data.values
        if self.mcdropout:
            print("Predictions using MC dropout"
                  " (MC iterations {}).".format(self.mc_iterations))
            predictions_all = np.empty(())
            predictions_i = self.model.predict(data)
            predictions_all =\
                np.empty(((self.mc_iterations,) + predictions_i.shape))
            predictions_all[0] = predictions_i
            for mc_it in range(1, self.mc_iterations):
                predictions_all[mc_it] = self.model.predict(data)
            predictions = predictions_all.mean(axis=0)
            predictions_std = predictions_all.std(axis=0)
        else:
            predictions = self.model.predict(data)
            predictions_std = np.zeros(predictions.shape)
        pd.options.display.float_format = "{:,.3f}".format
        predictions = pd.DataFrame(predictions, columns=labels)
        predictions_std = pd.DataFrame(predictions_std, columns=labels)
        return predictions, predictions_std

    def save_model(self):
        """
        Saves the  model to disk in tf format.
        :return none
        """

        if self.model is None:
            raise Exception("Model not configured and trained !")
        model_saved =\
            self.path_to_save_model.exists() and\
            self.path_to_save_model.joinpath("saved_model.pb").exists()
        if not model_saved:
            self.model.save(self.path_to_save_model, save_format="tf")
            logging.info("Model saved at "
                         "path: {}".format(self.path_to_save_model))
        else:
            logging.info("Model {} already exists. Skipping saving, in order"
                         " to avoid erasing best model save "
                         "by callbacks".format(self.path_to_save_model.name))
        return

    def load_model(self, path_to_saved_model=None):
        """
        Loads the saved model from the disk.
        Args:
            path_to_saved_model (str or pathlib.Path):

        Returns:
            None
        """
        if path_to_saved_model is None:
            path_to_saved_model = self.path_to_save_model
        logging.debug("Loading model from {} \n".format(path_to_saved_model))
        self.model = tf.keras.models.load_model(path_to_saved_model)


        return

    def plot_model(self, path_to_export=None):
        """

        Args:
            path_to_export (pathlib.Path): Path to the directory to save the
            plot.

        Returns:

        """
        if path_to_export is None:
            path_to_export = Path(project_configuration["project_store_path"])\
                .joinpath("reports")
        path_to_export = path_to_export.joinpath("{}_model.png"
                                                 "".format(self.model_name))
        if not path_to_export.parent.is_dir():
            path_to_export.parent.mkdir()
        return keras.utils.plot_model(self.model, path_to_export,
                                      show_shapes=True)

    def plot_performance(self, save=False):
        # Creating performance plot
        logging.info("Creating performance plot for model"
                     " {}".format(self.model_name))
        path_to_save_mp = None
        if save:
            set_name = self.exports_common_name + "_accuracy_loss" + ".png"
            path_to_save_mp = \
                Path(project_configuration["project_store_path"]).\
                joinpath("reports"). \
                joinpath(self.exports_common_name).joinpath(set_name)
        plot_model_performance(self.history,
                               path_to_save=path_to_save_mp if save else None)

    def plot_confusion_matrix(self, test_y, predictions, descriptions,
                              save=False, path_to_save=None):

        logging.info("Calculating and extracting confusion matrix for"
                     " model {}".format(self.model_name))
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        cm = confusion_matrix(test_y, predictions)
        path_to_save_cm = None
        if save:
            if path_to_save is None:
                set_name = self.exports_common_name + "_cm_test-data" + ".png"
                path_to_save_cm = \
                    Path(project_configuration["project_store_path"]).\
                    joinpath("reports").joinpath(self.exports_common_name).\
                    joinpath(set_name)
            else:
                path_to_save_cm = path_to_save
        plot_confusion_matrix(cm, descriptions, normalize=True,
                              path_to_save=path_to_save_cm if save else None)

    def generate_classification_report(self, test_y, predictions,
                                       output_dict=True):
        """

        Args:
            test_y: test_y vector (take care not to pass one-hot encoded
            vector!)
            predictions:
            output_dict: dict

        Returns:

        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        results = classification_report(test_y, predictions,
                                        output_dict=output_dict)
        if output_dict:
            logging.info(
                "Classification results for model {0}:\n\tPreci"
                "sion:{1:.3f}\n\tRecall:{2:.3f}\n\tF1"
                "-Score:{3:.3f}".format(self.model_name,
                                        results["macro avg"]["precision"],
                                        results["macro avg"]["recall"],
                                        results["macro avg"]["f1-score"]))
        return results

    def summary(self):
        """
        Implements Keras.model.summary()
        Returns:

        """
        self.model.summary()

    def save_run_summary(self):
        with tf.summary.create_file_writer(str(self.path_to_tf_logs)).\
                as_default():
            hp.hparams(self.hparams)  # record the values used in this trial
            accuracy = self.evaluation_results["eval_accuracy"]
            tf.summary.scalar("eval_accuracy", accuracy, step=1)

    def set_mc_predictions(self, mc_iterations=100):
        """
        Sets the number of MonterCarlo iterations.
        Args:
            mc_iterations:

        Returns:

        """
        self.mcdropout = True
        self.mc_iterations = mc_iterations
        # self.model = keras.models.Sequential([
        #     MCDropout(layer.rate) if isinstance(layer, keras.layers.Dropout)
        #     else layer
        #     for layer in self.model.layers]
        # )

    def unset_mc_predictions(self):
        """
        Unsets the Monte Carlo method.
        Returns:

        """
        self.mcdropout = False
        self.mc_iterations = 1
        # self.model = keras.models.Sequential([
        #     keras.layers.Dropout(layer.rate) if isinstance(MCDropout)
        #     else layer
        #     for layer in self.model.layers]
        # )


class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    Overloads tf.keras.callbacks.ModelCheckpoint and modifies it so it can
    call checkpoint depending on epochs run instead of data batches processed.
    Related to issue:
    [issue_link]
    (https://github.com/tensorflow/tensorflow/issues/
    33163#issuecomment-829575078)
    Args:
        epoch_per_save (int): Number of epochs to elapse before calling
                              checkpoint.
        *args (iterable): Positional arguments.
        **kwargs (iterable): keyword arguments.

    Attributes:
        epochs_per_save (int): Number of epochs to elapse before calling
                               checkpoint.

    """

    def __init__(self, epoch_per_save=1, *args, **kwargs):
        logging.debug("MyModelCheckpoint called with"
                      " epoch_per_save={}".format(epoch_per_save))
        self.epochs_per_save = epoch_per_save
        super().__init__(save_freq="epoch", *args, **kwargs)

    def on_epoch_end(self, epoch, logs):
        """
        Overloads `on_epoch_end` of super class.
        """

        if epoch % self.epochs_per_save == 0:
            super().on_epoch_end(epoch, logs)


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCSpatialDropout1D(keras.layers.SpatialDropout1D):
    def call(self, inputs):
        return super().call(inputs, training=True)
