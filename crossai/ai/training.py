from pprint import pprint
# tensorflow
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# sklearn imports
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# cross ai imports
from crossai.models.nn1d.model_xceptiontime import xception_time
from crossai.models.nn1d.nn_inception_time_func import inception_time
from crossai.models.nn1d.nn_bi_lstm import bi_lstm
from crossai.ai.callbacks_functions import callbacks_list
from crossai.ai.argument_checker import model_arguments_check_transform
from sklearn.model_selection import ShuffleSplit


def train_keras_clf(config, train_data, labels, model_name="xception_time",
                    mode="randomized"):
    """

    Args:
        config: The configuration data.
        train_data:
        labels:
        model_name:
        mode: accepted values: "krs_clf", "tf_clf"

    Returns:

    """
    config = model_arguments_check_transform(config=config)
    param_grid = config["param_grid_train"]
    clf_params = config["clf_params"]
    print("CLF params:")
    pprint(clf_params)
    print("Parameters Grid:")
    pprint(param_grid)
    print()
    print("Grid Value checkers:")
    print("- Type of kernel constraint:",
          type(param_grid["kernel_constraint"][0]))
    print("- Type of kernel regularize:",
          type(param_grid["kernel_regularize"][0]))
    print("- Type of learning rate:", type(param_grid["lr_rate"][0]))
    print("- Type of epsilon:", type(param_grid["adam_epsilon"][0]))
    print()

    if model_name == "xception_time":
        # keras clf with XceptionTime
        model = KerasClassifier(build_fn=xception_time)
        model.set_params(**clf_params)
    elif model_name == "inception_time":
        # pop the non-needed Xception related keys
        clf_params.pop("xception_adaptive_size", None)
        clf_params.pop("xception_adapt_ws_divide", None)
        # pop the non-needed parameter grid keys
        param_grid.pop("drp_mid", None)
        # keras clf with InceptionTime
        model = KerasClassifier(build_fn=inception_time)
        model.set_params(**clf_params)
    elif model_name == "bi_lstm":
        # pop the non-needed Xception related keys
        clf_params.pop("xception_adaptive_size", None)
        clf_params.pop("xception_adapt_ws_divide", None)
        # pop the non-needed parameter grid keys
        param_grid.pop("spatial", None)
        # keras clf with Bi_LSTM
        model = KerasClassifier(build_fn=bi_lstm)
        model.set_params(**clf_params)
    else:
        print("Please provide a valid model name.")
        raise Exception

    # add 1 to the config params to set a single split.
    if config["fit_grid_params"]["cv"] == 1:
        config["fit_grid_params"]["cv"] = ShuffleSplit(test_size=0.20,
                                                       n_splits=1,
                                                       random_state=None)

    if mode == "randomized":
        clf_model = RandomizedSearchCV(estimator=model,
                                       param_distributions=param_grid,
                                       cv=config["fit_grid_params"]["cv"],
                                       verbose=config["fit_grid_params"]
                                       ["verbose"])
    elif mode == "gridsearch":
        clf_model = GridSearchCV(estimator=model,
                                 param_grid=param_grid,
                                 cv=config["fit_grid_params"]["cv"],
                                 verbose=config["fit_grid_params"]["verbose"])
    else:
        print("Please, provide a valid algorithm for fine-tuning")
        raise Exception

    callbacks_lst = callbacks_list(config=config, lr_scheduler=True,
                                   checkpoint=False)

    print("Model parameters:")
    print(model.get_params())

    clf_model.fit(train_data, labels, callbacks=callbacks_lst)

    model = clf_model.best_estimator_.model

    return {"clf_model": clf_model,
            "model": model,
            "best_params": clf_model.best_params_,
            "best_score": clf_model.best_score_,
            "cv_results": clf_model.cv_results_,
            "clf_params": clf_params
            }


def train_tf_clf(config, clf_params, grid_params, model, train_data, labels):
    """

    Args:
        config:
        clf_params:
        grid_params:
        model:
        train_data:
        labels:

    Returns:

    """
    print("Training a Tensorflow model..")

    callbacks_lst = callbacks_list(config=config, lr_scheduler=True,
                                   checkpoint=False)

    history = model.fit(train_data, labels,
                        validation_split=clf_params["validation_split"],
                        epochs=clf_params["epochs"],
                        batch_size=grid_params["batch_size"],
                        callbacks=callbacks_lst,
                        use_multiprocessing=clf_params["use_multiprocessing"],
                        verbose=clf_params["verbose"])

    return history, model


def load_tf_clf(clf_params, grid_params, model_selection="xception_time"):
    """

    Args:
        clf_params:
        grid_params:
        model_selection:

    Returns:
        The TF model
    """
    keras.backend.clear_session()
    print("Model selected: {}".format(model_selection))
    if model_selection == "xception_time":
        model = xception_time(
            number_of_classes=clf_params["number_of_classes"],
            train_data_shape=clf_params["train_data_shape"],
            xception_adaptive_size=clf_params["xception_adaptive_size"],
            xception_adapt_ws_divide=clf_params["xception_adapt_ws_divide"],
            kernel_initialize=grid_params["kernel_initialize"],
            kernel_regularize=grid_params["kernel_regularize"],
            kernel_constraint=grid_params["kernel_constraint"],
            drp_input=grid_params["drp_input"],
            drp_mid=grid_params["drp_mid"],
            drp_high=grid_params["drp_high"],
            spatial=grid_params["spatial"],
            optimizer=grid_params["optimizer"],
            lr_rate=grid_params["lr_rate"],
            adam_epsilon=grid_params["adam_epsilon"],
            activation=clf_params["activation"]
        )

    elif model_selection == "inception_time":
        model = inception_time(
            number_of_classes=clf_params["number_of_classes"],
            train_data_shape=clf_params["train_data_shape"],
            kernel_initialize=grid_params["kernel_initialize"],
            kernel_regularize=grid_params["kernel_regularize"],
            kernel_constraint=grid_params["kernel_constraint"],
            drp_input=grid_params["drp_input"],
            drp_high=grid_params["drp_high"],
            spatial=grid_params["spatial"],
            optimizer=grid_params["optimizer"],
            lr_rate=grid_params["lr_rate"],
            adam_epsilon=grid_params["adam_epsilon"],
            activation=clf_params["activation"]
        )

    elif model_selection == "bi_lstm":
        model = bi_lstm(number_of_classes=clf_params["number_of_classes"],
                        train_data_shape=clf_params["train_data_shape"],
                        kernel_initialize=grid_params["kernel_initialize"],
                        kernel_regularize=grid_params["kernel_regularize"],
                        kernel_constraint=grid_params["kernel_constraint"],
                        drp_input=grid_params["drp_input"],
                        drp_mid=grid_params["drp_mid"],
                        drp_high=grid_params["drp_high"],
                        optimizer=grid_params["optimizer"],
                        lr_rate=grid_params["lr_rate"],
                        adam_epsilon=grid_params["adam_epsilon"],
                        activation=clf_params["activation"]
                        )
    else:
        print("Select a valid classifier.")
        raise Exception

    print(model.summary())
    keras.utils.plot_model(model,
                           to_file="{}_final.png".format(model_selection),
                           show_shapes=True,
                           show_layer_names=True)

    return model


def print_default_nn_models(config, model_name="xception_time"):
    """

    Args:
        config:
        model_name:

    Returns:

    """
    if model_name == "xception_time":
        model = xception_time(
            number_of_classes=config["clf_params"]["number_of_classes"],
            train_data_shape=config["clf_params"]["train_data_shape"])
    elif model_name == "inception_time":
        model = inception_time(
            number_of_classes=config["clf_params"]["number_of_classes"],
            train_data_shape=config["clf_params"]["train_data_shape"])
    elif model_name == "bi_lstm":
        model = bi_lstm(
            number_of_classes=config["clf_params"]["number_of_classes"],
            train_data_shape=config["clf_params"]["train_data_shape"])
    else:
        print("Please provide a valid model.")
        raise Exception

    print(model.summary())

    return model
