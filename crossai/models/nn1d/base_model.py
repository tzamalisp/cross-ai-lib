"""

"""
import time
import json
import logging
import numpy as np
import pandas as pd
from bunch import Bunch
from pathlib import Path
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.layers import Layer
from tensorboard.plugins.hparams import api as hp
from training.models_performance import plot_fit_history
from sklearn.metrics import classification_report, confusion_matrix
from crossai.ai.callbacks_learning_rate import callback_lr_scheduler
from crossai.models.models_perfomance_utils import nn_default_parameters
from crossai.models.models_perfomance_utils import plot_confusion_matrix
from tensorflow.keras.optimizers.schedules import \
    ExponentialDecay, PiecewiseConstantDecay


lr_schedule_list = ["tf_exponential", "tf_piecewise"]


class BaseModel:
    def __init__(self, config, model_name=None):
        """
        Constructor to initialize the model's architecture parameters.
        Args:
            config: the JSON configuration.
            model_name (str or None, optional): The model name. If not defined,
            current datetime is used.
        """
        self.model_name = datetime.now().strftime(
            "%Y-%m-%d-%H-%M") if model_name is None else model_name

        self.path_to_save_model = Path(
            project_configuration["project_store_path"]).joinpath("tf")

        if not self.path_to_save_model.exists():
            logging.debug(
                "Creating directory for saving model.".format(
                    self.path_to_save_model.relative_to(
                        project_configuration["project_store_path"])))
            self.path_to_save_model.mkdir(parents=True, exist_ok=True)

        set_name = self.model_name + ".tf"
        self.path_to_save_model = self.path_to_save_model.joinpath(set_name)
        self.path_to_save_model.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.config["path_to_save_model"] = self.path_to_save_model

        self.exports_common_name = None

        # log_dir is used by tensorboard to log variables related
        # to the execution of the NN.
        log_dir = self.path_to_save_model
        self.path_to_tf_logs = log_dir

        logging.debug(
            "Path to TF logs (usage with tensorboard) : {}".format(
                self.path_to_tf_logs))

        self.optimizer = config.get("optimizer", None)
        self.adam_epsilon = config.get("adam_epsilon", 1e-7)
        self.opt_schedule = config.get("opt_schedule", None)
        self.learning_rate = config.get("learning_rate", None)
        self.epochs = config.get("epochs", 0)
        self.batch_size = config.get("batch_size", None)
        self.nn_metrics = nn_default_parameters["nn_metrics"] \
            if "nn_metrics" not in config.keys() \
            else config["nn_metrics"]
        self.nn_loss_function = nn_default_parameters["nn_loss_function"]\
            if "nn_loss_function" not in config.keys() \
            else config["nn_loss_function"]

        # Initialize a model
        self.model = None
        self.number_of_classes = config.get("number_of_classes", 1)
        self.dataset_shape = config.get("dataset_shape", None)
        self.iters_per_epoch = None
        # Default predictions behaviour. By default the MonteCarlo predictions
        # is deactivated.
        self.mcdropout = False
        self.mc_iterations = 1
        # Training time.
        self.train_time = 0

        # Predicted class labels.
        self.predictions = np.array([])
        self.history = None
        self.evaluation_results = dict()
        return

    def calculate_number_of_filters(self):
        """
        Calaculates the filter size for a given layer.
        :param none
        :return none
        :raises NotImplementedError: Implement this method.
        """

        # Implement this method in the inherited class to calculate the filter size.
        raise NotImplementedError

    def callbacks_init(self):
        """

        Returns:

        """
        callbacks_list = []
        calbacks_configuration = project_configuration.get("callbacks")
        if calbacks_configuration["early_stopping"]["active"]:
            es_monitor = calbacks_configuration["early_stopping"]["monitor"]
            es_mode = calbacks_configuration["early_stopping"]["mode"]
            es_patience = calbacks_configuration["early_stopping"]["patience"]
            es_delta = calbacks_configuration["early_stopping"].get("min_delta", 0)
            es_baseline = calbacks_configuration["early_stopping"].get("baseline", None)
            callback_early_stopping = keras.callbacks.EarlyStopping(monitor=es_monitor,
                                                                    mode=es_mode,
                                                                    patience=es_patience,
                                                                    min_delta=es_delta,
                                                                    baseline=es_baseline,
                                                                    verbose=1 if logging.root.level == logging.DEBUG
                                                                    else 0,
                                                                    restore_best_weights=True)
            callbacks_list.append(callback_early_stopping)
        if calbacks_configuration["checkpoint"]["active"]:
            checkpoint_monitor = calbacks_configuration["checkpoint"]["monitor"]
            checkpoint_mode = calbacks_configuration["checkpoint"]["mode"]
            checkpoint_filepath = Path(project_configuration["project_store_path"]). \
                joinpath("tf").joinpath("{}.tf".format(self.model_name))
            if not checkpoint_filepath.is_dir():
                checkpoint_filepath.mkdir(parents=True, exist_ok=True)
            save_freq = calbacks_configuration["checkpoint"]["save_freq"]
            logging.debug("Save freq checkpoint: {}".format(save_freq))
            callback_checkpoint = MyModelCheckpoint(epoch_per_save=save_freq,
                                                    filepath=str(checkpoint_filepath),
                                                    monitor=checkpoint_monitor,
                                                    verbose=1 if logging.root.level == logging.DEBUG else 0,
                                                    save_best_only=True,
                                                    mode=checkpoint_mode
                                                    )
            callbacks_list.append(callback_checkpoint)
        if calbacks_configuration["lr_scheduler"]["active"]:
            lr_clb = callback_lr_scheduler(calbacks_configuration["lr_scheduler"]["mode"])
            callbacks_list.append(lr_clb)
        if calbacks_configuration["tensorboard"]["active"]:
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.path_to_tf_logs)
            callbacks_list.append(tensorboard_callback)
            # hparams_callback = hp.KerasCallback(self.path_to_tf_logs, self.hparams)
        return callbacks_list

    def define_model(self, config=None):
        """
        Constructs the model.
        Args:
            config: configuration

        Returns:

        """
        # Implement this method in the inherited class to add layers to the model.
        raise NotImplementedError

    def compile(self, optimizer=None, learning_rate=None, loss=None, nn_metrics=None,
                adam_epsilon=None, opt_schedule=None):
        """
        If optimizer and learning_rate are given as arguments, then they are used insted of the defaults given in the
        configuration. Useful if function is called from outside the main script (e.g. a Notebook)
        Args:
            optimizer: Keras optimizer
            learning_rate: float
            loss: One of the keras loss functions
            nn_metrics: Keras metrics for the performance of the trained model
            adam_epsilon:
            opt_schedule:
        """

        if optimizer is None:
            optimizer = self.optimizer
        else:
            self.optimizer = optimizer
        assert (optimizer in ["adam", "sgd"])

        # define learning_rate
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            learning_rate = self.learning_rate
        if opt_schedule is not None and opt_schedule in lr_schedule_list:
            lr_rate = learning_rate_scheduler(opt_schedule=opt_schedule)
        else:
            lr_rate = learning_rate

        if optimizer == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=lr_rate)
        else:
            epsilon = adam_epsilon if adam_epsilon else self.adam_epsilon
            if isinstance(epsilon, str):
                epsilon = float(epsilon.replace("−", "-"))
            optimizer = keras.optimizers.Adam(learning_rate=lr_rate, epsilon=epsilon)

        metrics = [self.nn_metrics if nn_metrics is None else nn_metrics]
        if loss is not None:
            self.nn_loss_function = loss
        else:
            loss = self.nn_loss_function

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=[metrics, get_lr_metric(optimizer)])

    def fit(self, train_dataset, validation_dataset=None, validation_split=None, verbose=None):
        """
        Trains the self.model.
        Args:
            train_dataset (tupple): Tupple of numpy arrays (x_train, y_train)
            validation_dataset(tupple): Tupple of numpy arrays (x_validation, y_validation)
            verbose:
            validation_split:
        Returns: None

        """
        if verbose is None:
            verbose = 1 if logging.root.level == logging.DEBUG else 0

        epochs = self.epochs
        batch_size = self.batch_size
        steps_per_epoch = train_dataset[-1].shape[0] // batch_size
        logging.debug("Model.fit : Steps per epoch : {}".format(steps_per_epoch))
        callbacks_list = self.callbacks_init()
        # train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        # val_dataset = tf.data.Dataset.from_tensor_slices(validation_dataset).\
        #     shuffle(batch_size, reshuffle_each_iteration=True).\
        #     batch(batch_size)
        if validation_split:
            if validation_dataset is None:
                train_data = train_dataset[0]
                labels = train_dataset[1]
            else:
                msg = "Validation_split will be used in `fit` but a validation dataset has been provided too." \
                      "Concatenating validation with train dataset."
                logging.warning(msg)
                train_data = np.vstack((train_dataset[0], validation_dataset[0]))
                if len(train_dataset[1].shape) > 1:
                    labels = np.vstack((train_dataset[1], validation_dataset[1]))
                else:
                    labels = np.hstack((train_dataset[1], validation_dataset[1]))
            keras.backend.clear_session()
            self.history = self.model.fit(train_data, labels,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          callbacks=callbacks_list,
                                          verbose=verbose,
                                          validation_split=validation_split
                                          )
        else:
            if validation_dataset is None:
                msg = "Validation split is not defined and validation dataset is None."
                logging.error(msg)
                raise Exception(msg)
            else:
                self.history = self.model.fit(train_dataset[0],
                                              train_dataset[1],
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              callbacks=self.callbacks_init(),
                                              verbose=verbose,
                                              validation_data=validation_dataset
                                              )

        # if "loss" in self.history.history.keys():
        #     self.performance_history["loss"] += self.history.history["loss"]
        # if "val_loss" in self.history.history.keys():
        #     self.performance_history["val_loss"] += self.history.history["val_loss"]
        # if "accuracy" in self.history.history.keys():
        #     self.performance_history["accuracy"] += self.history.history["accuracy"]
        # if "val_accuracy" in self.history.history.keys():
        #     self.performance_history["val_accuracy"] += self.history.history["val_accuracy"]

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
        verbose = kwargs.get("verbose", (1 if logging.root.level == logging.DEBUG else 0))
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
        self.exports_common_name = "{0:.3f}_{1}".format(self.evaluation_results["eval_accuracy"], self.model_name)

        return self.evaluation_results

    def predict(self, data, labels=None):
        """
        Predicts the class labels of unknown data.
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
        if self.mcdropout:
            print("Predictions using MC dropout (MC iterations {}).".format(self.mc_iterations))
            predictions_all = np.empty(())
            predictions_i = self.model.predict(data)
            predictions_all = np.empty(((self.mc_iterations,) + predictions_i.shape))
            predictions_all[0] = predictions_i
            for mc_it in range(1, self.mc_iterations):
                predictions_all[mc_it] = self.model.predict(data)
            predictions = predictions_all.mean(axis=0)
            predictions_std = predictions_all.std(axis=0)
            predictions_var = predictions_all.var(axis=0)
        else:
            predictions = self.model.predict(data)
            predictions_std = np.zeros(predictions.shape)
            predictions_var = np.zeros(predictions.shape)
        pd.options.display.float_format = "{:,.3f}".format
        predictions = pd.DataFrame(predictions, columns=labels)
        predictions_std = pd.DataFrame(predictions_std, columns=labels)
        predictions_var = pd.DataFrame(predictions_var, columns=labels)
        return predictions, predictions_std, predictions_var

    def save_model(self):
        """
        Saves the  model to disk in tf format.
        :return none
        """

        if self.model is None:
            raise Exception("Model not configured and trained !")
        model_saved = self.path_to_save_model.exists() and self.path_to_save_model.joinpath("saved_model.pb").exists()
        if not model_saved:
            keras.models.save_model(
                self.model, self.path_to_save_model, overwrite=True, include_optimizer=True, save_format="tf",
                signatures=None, options=None, save_traces=True
            )
            logging.info("Model saved at path: {}".format(self.path_to_save_model))
        else:
            logging.info("Model {} already exists. Skipping saving, in order to avoid erasing best model save "
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
        self.model = keras.models.load_model(path_to_saved_model,
                                             compile=False)
        self.compile()
        return

    def plot_model(self, path_to_export=None):
        """

        Args:
            path_to_export (pathlib.Path): Path to the directory to save the plot.

        Returns:

        """
        if path_to_export is None:
            path_to_export = Path(project_configuration["project_store_path"]).joinpath("reports")
        path_to_export = path_to_export.joinpath("{}_model.png".format(self.model_name))
        if not path_to_export.parent.is_dir():
            path_to_export.parent.mkdir()
        return keras.utils.plot_model(self.model, path_to_export, show_shapes=True)

    def plot_performance(self, save=False):
        # Creating performance plot
        logging.info("Creating performance plot for model {}".format(self.model_name))
        path_to_save_mp = None
        for metric in ["accuracy", "loss", "lr"]:
            if metric in self.history.history.keys():
                if save:
                    set_name = self.exports_common_name + "_{}.png".format(metric)
                    path_to_save_mp = Path(project_configuration["project_store_path"]).joinpath("reports"). \
                        joinpath(self.exports_common_name).joinpath(set_name)
                plot_fit_history(self.history, metric, path_to_save=path_to_save_mp if save else None)

    def plot_confusion_matrix(self, test_y, predictions, descriptions, save=False, path_to_save=None):

        logging.info("Calculating and extracting confusion matrix for model {}".format(self.model_name))
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        cm = confusion_matrix(test_y, predictions)
        path_to_save_cm = None
        if save:
            if path_to_save is None:
                set_name = self.exports_common_name + "_cm_test-data" + ".png"
                path_to_save_cm = Path(project_configuration["project_store_path"]).joinpath("reports"). \
                    joinpath(self.exports_common_name).joinpath(set_name)
            else:
                path_to_save_cm = path_to_save
        plot_confusion_matrix(cm, descriptions, normalize=True,
                              path_to_save=path_to_save_cm if save else None)

    def generate_classification_report(self, test_y, predictions, target_names=None, output_dict=True):
        """

        Args:
            test_y: test_y vector (take care not to pass one-hot encoded vector!)
            predictions:
            output_dict: dict
            target_names:

        Returns:

        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        results = classification_report(test_y, predictions, target_names=target_names,
                                        output_dict=output_dict)
        if output_dict:
            logging.info(
                "Classification results for model {0}:\n\tPrecision:{1:.3f}\n\tRecall:{2:.3f}"
                "\n\tF1-Score:{3:.3f}".format(self.model_name,
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

    # def save_hparams_run_summary(self):
    #     with tf.summary.create_file_writer(str(self.path_to_tf_logs)).as_default():
    #         hp.hparams(self.hparams)  # record the values used in this trial
    #         accuracy = self.evaluation_results["eval_accuracy"]
    #         tf.summary.scalar("eval_accuracy", accuracy, step=1)

    def set_mc_predictions(self, mc_iterations=100):
        self.mcdropout = True
        self.mc_iterations = mc_iterations
        # self.model = keras.models.Sequential([
        #     MCDropout(layer.rate) if isinstance(layer, keras.layers.Dropout) else layer
        #     for layer in self.model.layers]
        # )

    def unset_mc_predictions(self):
        self.mcdropout = False
        self.mc_iterations = 1
        # self.model = keras.models.Sequential([
        #     keras.layers.Dropout(layer.rate) if isinstance(MCDropout) else layer
        #     for layer in self.model.layers]
        # )


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


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCSpatialDropout1D(keras.layers.SpatialDropout1D):
    def call(self, inputs):
        return super().call(inputs, training=True)


class DropoutLayer(Layer):
    def __init__(self, drp_rate=0.1, spatial=True):
        super(DropoutLayer, self).__init__()
        self.drp_rate = drp_rate
        self.spatial = spatial
        if spatial is True:
            self.drp = MCSpatialDropout1D(drp_rate)
        else:
            self.drp = MCDropout(drp_rate)

    def call(self, inputs):
        return self.drp(inputs)

    def get_config(self):
        return {"drp_rate": self.drp_rate,
                "spatial": self.spatial}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def dropout_layer(input_tensor, drp_on=True, drp_rate=0.1, spatial=True):
    """

    Args:
        input_tensor:
        drp_on:
        drp_rate:
        spatial:

    Returns:

    """
    if drp_on is True:
        if spatial is True:
            x = MCSpatialDropout1D(drp_rate)(input_tensor)
            # print("MC Spatial Dropout Rate: {}".format(drp_rate))
        else:
            x = MCDropout(drp_rate)(input_tensor)
            # print("MC Dropout Rate: {}".format(drp_rate))
    else:
        x = input_tensor

    return x


def learning_rate_scheduler(opt_schedule, exp_initial_lr=0.001, exp_decay_stp=20, exp_decay_rt=0.1,
                            pcw_boundaries=[5, 15], pcw_values=[0.01, 0.005, 0.001]):
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
