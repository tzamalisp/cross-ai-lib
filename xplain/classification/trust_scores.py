import logging
from pathlib import Path
from pickle import dump, load
from alibi.confidence import TrustScore
from training.scaling import scale_dataset
from training.scaling import transform_dataset
from src.processing.dataset_process import load_dataset_npz
from training.dimensionality_reduction import Decomposer
from training.dimensionality_reduction import decompose
from configuration_functions.project_configuration_variables \
    import project_configuration

from sklearn.model_selection import train_test_split


def prepare_train_data(task, data=None, scaling_method="standard",
                       processing_uuid="untitled"):
    """
    Prepares the data and transforms it into the correct format for the
    trust_score model.
    Args:
        task (dict): The classification task dictionary.
        data (numpy.ndarray): An array with the data to be trained.
        scaling_method (str): The scaling method to use.
        processing_uuid (str): The processing UUID  of the classification task.

    Returns:
        train_data (tuple): train data X, y
    """

    if data is None:
        logging.debug("No train data have been provided. Assuming to use "
                      "preprocessing dataset's data.")
        preprocessing_uuid = task.get("preprocessing_uuid", None)
        if preprocessing_uuid is None:
            msg = "No dataset is available!"
            logging.error(msg)
            raise Exception(msg)
        logging.debug("Creating decomposer from"
                      " dataset {}".format(processing_uuid))
        logging.debug("Loading dataset {}".format(preprocessing_uuid))
        loaded_data = load_dataset_npz(preprocessing_uuid)
        try:
            train_data = loaded_data["train_X"]
            data_y = loaded_data["train_y"]
        except KeyError as e:
            logging.debug(e)
            data = loaded_data["X"]
            data_y = loaded_data["y"]
            train_data, data_y, train_y, test_y = \
                train_test_split(data, data_y, test_size=0.20,random_state=42,
                                 shuffle=True, stratify=data_y)

    else:
        train_data = data[0]
        data_y = data[1]

    logging.debug("Scaling data before decomposing.")
    scale_config = dict()
    scale_config["uuid"] = processing_uuid
    if task is not None:
        logging.debug("Scaling train data according to fitted stored scaler.")
        scale_config["scaling"] = task.get("scaling")
        transformed_data = transform_dataset(scale_config, train_data)
    else:
        logging.debug("Scaling train data with new scaler.")
        scale_config["scaling"] = scaling_method
        transformed_data, _ = scale_dataset(scale_config, train_data)
    return transformed_data, data_y


def ts_dimensionality_reduction_transformer_load(
        dimensionality_reduction_method,task=None, method_params=None,
        project_store_path=None):
    """
    This functions returns a transformer according to the input parameters.
    Args:
        dimensionality_reduction_method (str): The dimensionality reduction
         method.
        task (dict): The classification task dictionary.
        method_params (dict): The dictionary with the dimensionality reduction
         method's parameters.
        project_store_path (str, pathlib obj): The project store path.

    Returns:
    A transformer object.
    """
    transformer = None
    if dimensionality_reduction_method == "pca":
        processing_uuid = "untitled"
        if task is not None:
            processing_uuid = task["processing_uuid"]
        components = method_params.get("components", 0.95)
        decomposer_name = "{}_{}".format(processing_uuid,
                                         str(components).replace(".", "_"))
        transformer = load_ts_pca_transformer(decomposer_name,
                                              project_store_path
                                              =project_store_path)
    else:
        logging.debug("Method not implemented.")
    return transformer


def ts_dimensionality_reduction_fit(train_data,
                                    dimensionality_reduction_method,
                                    task=None, method_params=dict(),
                                    project_store_path=None):
    """
    Used only during training of dimensionality reduction transformers.
    Args:
        dimensionality_reduction_method (str): The mode of dimensionality
         reduction. Accepted values are `pca`, `ae`.
            If `pca`, a PCA transformer will be used. Else, an auto-encoder
            model will be used for dimensionality
            reduction (currently not implemented).
        task:
        train_data (tuple): Used if there is no previously created decomposer
        and a new one must be fit.
        method_params (dict): The method parameters
        project_store_path (str, pathlib obj): The project store path.

    Returns:
        transformer
        train_data_dr: The training data tuple after dimensionality reduction.
    """
    logging.debug("Fit of dimensionality reduction met"
                  "hod {}".format(dimensionality_reduction_method))
    transformer = None
    transformer_params = dict()
    if dimensionality_reduction_method == "pca":
        components = method_params.get("components", 0.95)
        transformer, train_data_dr = fit_pca_to_task_train_data(train_data,
                                        task,
                                        components=components,
                                        project_store_path=project_store_path)

    elif dimensionality_reduction_method == "ae":
        msg = "`ae` mode is not implemented yet"
        logging.error(msg)
        raise Exception(msg)
    else:
        msg = "Invalid dimensionality reduction " \
              "method `{}`".format(dimensionality_reduction_method)
        logging.error(msg)
        raise Exception(msg)

    return transformer, train_data_dr


def load_ts_pca_transformer(decomposer_name,
                            project_store_path=None):
    """
    This function loads a decomposer from a certain path.
    Args:
        decomposer_name (str): The name of the decomposer.
        project_store_path (str, pathlib obj): The project store path.
        task (dict): The dictionary with PCA parameters.

    Returns:
        decomposer (training.dimensionality_reduction.Decomposer)
    """
    decomposer = None
    if project_store_path is None:
        project_store_path = project_configuration["project_store_path"]
    decomposers_dir = Path(project_store_path).joinpath("decomposers")
    logging.debug("Attempting to locate decomposer {} in decomposers'"
                  " directory".format(decomposer_name))
    try:
        decomposer = Decomposer(decomposers_dir, decomposer_name)
        logging.debug("Decomposer loaded!")
    except FileNotFoundError as e:
        logging.debug("Decomposer {} does not exist".format(decomposer_name))
    return decomposer


def fit_pca_to_task_train_data(data, task, components=0.95,
                               project_store_path=None):
    """
    This functions "fits" PCA
    Args:
        data (tuple): Task training data (after scaling) to be used on fit
        task (dict):
        components (int, float): The dimensions of PCA algorithm to be
         exported (variance of the dataset).
        project_store_path
    Returns:
        train_data_dr (tuple): train data X after decomposer fit and transform,
         vector y
    """

    processing_uuid = "untitled"
    if task is not None:
        processing_uuid = task["processing_uuid"]
    decomposer_name = "{}_{}".format(processing_uuid,
                                     str(components).replace(".", "_"))
    transformed_data, data_y = data[0], data[1]
    decomposer = load_ts_pca_transformer(decomposer_name,
                                         project_store_path=project_store_path)
    if decomposer is None:
        logging.debug("Creating decomposer {}".format(decomposer_name))
        # Create and save decomposer
        _ , _, train_data_dr, _ = decompose(transformed_data, decomposer_name,
                                            components=components)
        # Load decomposer object
        decomposer = load_ts_pca_transformer(decomposer_name,
                                         project_store_path=project_store_path)
    else:
        train_data_dr = decomposer.transform(transformed_data)
    return decomposer, (train_data_dr, data_y)


def trust_score_model_fit(train_data_dr, ts_classes, path_to_save=None,
                          **trust_score_params):
    logging.debug("TrustScore fit.")
    ts = TrustScore(**trust_score_params)
    train_X = train_data_dr[0]
    train_y = train_data_dr[1]
    ts.fit(train_X, train_y, classes=ts_classes)
    dump(ts, open(path_to_save, "wb"))
    return ts


def trust_scores_model_score(data,
                             clf_predictions,
                             dimensionality_reduction_method,
                             transformer,
                             ts_model,
                             **params_score):
    logging.debug("Dimensionality reduction on predicted data.")
    if dimensionality_reduction_method == "pca":
        data_dr = transformer.transform(data)
    else:
        # autoencoder case
        data_dr = transformer.predict(data)
    logging.debug("Calculating Trust Score")
    logging.debug("type data_dr {}".format(type(data_dr)))
    logging.debug("shape data_dr {}".format(data_dr.shape))
    logging.debug("type clf_predictions {}".format(type(clf_predictions)))
    logging.debug("shape clf_predictions {}".format(clf_predictions.shape))
    logging.debug("clf_predictions {}".format(clf_predictions))
    logging.debug(params_score)
    score, closest_class = ts_model.score(data_dr,
                                          clf_predictions,
                                          **params_score)
    logging.debug("score len: {}".format(len(score)))
    logging.debug("closest class len: {}".format(len(closest_class)))
    logging.debug("score : {}".format(score))
    logging.debug("closest class : {}".format(closest_class))
    try:
        if len(closest_class) != len(score):
            msg = "Trust score closest_class has returned more elements than" \
                  " score elements. "
            logging.error(msg)
            raise Exception(msg)
    except Exception as e:
        closest_class = closest_class[:len(score)]
    return score, closest_class


def calc_trust_scores(data,
                      clf_predictions,
                      task=None,
                      train_data=None,
                      dimensionality_reduction_method="pca",
                      method_params=dict(),
                      trust_score_params_init=dict(),
                      trust_score_classes=2,
                      trust_score_params_score=dict(),
                      project_store_path=None):
    """

    Args:
        data:
        clf_predictions:
        task:
        train_data:
        dimensionality_reduction_method:
        method_params:
        trust_score_params_init:
        trust_score_classes:
        trust_score_params_score:
        project_store_path:

    Returns:

    """
    transformer = None  # The transformer that will be used to perform
    # dimensionality reduction on prediction data.
    if task is None:
        processing_uuid = "untitled"
    else:
        processing_uuid = task["processing_uuid"]
    logging.debug("Calculating TrustScore")
    if project_store_path is None:
        project_store_path = project_configuration["project_store_path"]
    trust_score_models_path = Path(project_store_path).joinpath("trust_sco"
                                                                "res_models")
    model_name = "{}.pkl".format(processing_uuid)
    path_to_model = trust_score_models_path.joinpath(model_name)
    if not trust_score_models_path.is_dir():
        trust_score_models_path.mkdir()
    if not path_to_model.exists():
        print("Creating a TrustScore model for "
              "processing task {}.".format(processing_uuid))
        logging.debug("Loading train data")
        train_data = prepare_train_data(task, data=train_data,
                                        processing_uuid=processing_uuid)
        transformer, train_data_dr = ts_dimensionality_reduction_fit(
                                         train_data,
                                         dimensionality_reduction_method,
                                         task=task,
                                         method_params=method_params,
                                         project_store_path=project_store_path)
        ts_model = trust_score_model_fit(train_data_dr, trust_score_classes,
                                         path_to_save=path_to_model,
                                         **trust_score_params_init)

    else:
        print("Loading a TrustScore model.")
        ts_model = load(open(path_to_model, "rb"))
        transformer = ts_dimensionality_reduction_transformer_load(
                                       dimensionality_reduction_method,
                                       task=task,
                                       method_params=method_params,
                                       project_store_path=project_store_path)

    score, closest_class = trust_scores_model_score(data,
                                            clf_predictions,
                                            dimensionality_reduction_method,
                                            transformer,
                                            ts_model,
                                            **trust_score_params_score)
    return score, closest_class
