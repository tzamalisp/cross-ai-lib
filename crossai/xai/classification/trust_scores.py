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
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, Any

from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)


class TrustScore(object):

    def __init__(self, k_filter: int = 10, alpha: float = 0., filter_type: str = None,
                 leaf_size: int = 40, metric: str = 'euclidean', dist_filter_type: str = 'point') -> None:
        """
        Initialize trust scores.

        Parameters
        ----------
        k_filter
            Number of neighbors used during either kNN distance or probability filtering.
        alpha
            Fraction of instances to filter out to reduce impact of outliers.
        filter_type
            Filter method; either 'distance_knn' or 'probability_knn'
        leaf_size
            Number of points at which to switch to brute-force. Affects speed and memory required to build trees.
            Memory to store the tree scales with n_samples / leaf_size.
        metric
            Distance metric used for the tree. See sklearn's DistanceMetric class for a list of available metrics.
        dist_filter_type
            Use either the distance to the k-nearest point (dist_filter_type = 'point') or
            the average distance from the first to the k-nearest point in the data (dist_filter_type = 'mean').
        """
        self.k_filter = k_filter
        self.alpha = alpha
        self.filter = filter_type
        self.eps = 1e-12
        self.leaf_size = leaf_size
        self.metric = metric
        self.dist_filter_type = dist_filter_type

    def filter_by_distance_knn(self, X: np.ndarray) -> np.ndarray:
        """
        Filter out instances with low kNN density. Calculate distance to k-nearest point in the data for each
        instance and remove instances above a cutoff distance.

        Parameters
        ----------
        X
            Data

        Returns
        -------
        Filtered data.
        """
        kdtree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        knn_r = kdtree.query(X, k=self.k_filter + 1)[0]  # distances from 0 to k-nearest points
        if self.dist_filter_type == 'point':
            knn_r = knn_r[:, -1]
        elif self.dist_filter_type == 'mean':
            knn_r = np.mean(knn_r[:, 1:], axis=1)  # exclude distance of instance to itself
        cutoff_r = np.percentile(knn_r, (1 - self.alpha) * 100)  # cutoff distance
        X_keep = X[np.where(knn_r <= cutoff_r)[0], :]  # define instances to keep
        return X_keep

    def filter_by_probability_knn(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out instances with high label disagreement amongst its k nearest neighbors.

        Parameters
        ----------
        X
            Data
        Y
            Predicted class labels

        Returns
        -------
        Filtered data and labels.
        """
        if self.k_filter == 1:
            logger.warning('Number of nearest neighbors used for probability density filtering should '
                           'be >1, otherwise the prediction probabilities are either 0 or 1 making '
                           'probability filtering useless.')
        # fit kNN classifier and make predictions on X
        clf = KNeighborsClassifier(n_neighbors=self.k_filter, leaf_size=self.leaf_size, metric=self.metric)
        clf.fit(X, Y)
        preds_proba = clf.predict_proba(X)
        # define cutoff and instances to keep
        preds_max = np.max(preds_proba, axis=1)
        cutoff_proba = np.percentile(preds_max, self.alpha * 100)  # cutoff probability
        keep_id = np.where(preds_max >= cutoff_proba)[0]  # define id's of instances to keep
        X_keep, Y_keep = X[keep_id, :], Y[keep_id]
        return X_keep, Y_keep

    def fit(self, X: np.ndarray, Y: np.ndarray, classes: int = None) -> None:
        """
        Build KDTrees for each prediction class.

        Parameters
        ----------
        X
            Data
        Y
            Target labels, either one-hot encoded or the actual class label.
        classes
            Number of prediction classes, needs to be provided if Y equals the predicted class.
        """
        self.classes = classes if classes is not None else Y.shape[1]
        self.kdtrees = [None] * self.classes  # type: Any
        self.X_kdtree = [None] * self.classes  # type: Any

        # KDTree and kNeighborsClassifier need 2D data
        if len(X.shape) > 2:
            logger.warning('Reshaping data from {0} to {1} so k-d trees can '
                           'be built.'.format(X.shape, X.reshape(X.shape[0], -1).shape))
            X = X.reshape(X.shape[0], -1)

        # make sure Y represents predicted classes, not one-hot encodings
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)  # type: ignore

        if self.filter == 'probability_knn':
            X_filter, Y_filter = self.filter_by_probability_knn(X, Y)

        for c in range(self.classes):

            if self.filter is None:
                X_fit = X[np.where(Y == c)[0]]
            elif self.filter == 'distance_knn':
                X_fit = self.filter_by_distance_knn(X[np.where(Y == c)[0]])
            elif self.filter == 'probability_knn':
                X_fit = X_filter[np.where(Y_filter == c)[0]]

            no_x_fit = len(X_fit) == 0
            if no_x_fit and len(X[np.where(Y == c)[0]]) == 0:
                logger.warning('No instances available for class %s', c)
            elif no_x_fit:
                logger.warning('Filtered all the instances for class %s. Lower alpha or check data.', c)

            self.kdtrees[c] = KDTree(X_fit, leaf_size=self.leaf_size, metric=self.metric)  # build KDTree for class c
            self.X_kdtree[c] = X_fit

    def score(self, X: np.ndarray, Y: np.ndarray, k: int = 2, dist_type: str = 'point') \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate trust scores = ratio of distance to closest class other than the
        predicted class to distance to predicted class.

        Parameters
        ----------
        X
            Instances to calculate trust score for.
        Y
            Either prediction probabilities for each class or the predicted class.
        k
            Number of nearest neighbors used for distance calculation.
        dist_type
            Use either the distance to the k-nearest point (dist_type = 'point') or
            the average distance from the first to the k-nearest point in the data (dist_type = 'mean').

        Returns
        -------
        Batch with trust scores and the closest not predicted class.
        """
        # make sure Y represents predicted classes, not probabilities
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)  # type: ignore

        # KDTree needs 2D data
        if len(X.shape) > 2:
            logger.warning('Reshaping data from {0} to {1} so k-d trees can '
                           'be queried.'.format(X.shape, X.reshape(X.shape[0], -1).shape))
            X = X.reshape(X.shape[0], -1)

        d = np.tile(None, (X.shape[0], self.classes))  # init distance matrix: [nb instances, nb classes]

        for c in range(self.classes):
            d_tmp = self.kdtrees[c].query(X, k=k)[0]  # get k nearest neighbors for each class
            if dist_type == 'point':
                d[:, c] = d_tmp[:, -1]
            elif dist_type == 'mean':
                d[:, c] = np.mean(d_tmp, axis=1)

        sorted_d = np.sort(d, axis=1)  # sort distance each instance in batch over classes
        # get distance to predicted and closest other class and calculate trust score
        d_to_pred = d[range(d.shape[0]), Y]
        d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1])
        trust_score = d_to_closest_not_pred / (d_to_pred + self.eps)
        # closest not predicted class
        class_closest_not_pred = np.where(d == d_to_closest_not_pred.reshape(-1, 1))[1]
        return trust_score, class_closest_not_pred


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
                train_test_split(data, data_y, test_size=0.20, random_state=42,
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
        dimensionality_reduction_method, task=None, method_params=None,
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
        transformer = load_ts_pca_transformer(
            decomposer_name,
            project_store_path=project_store_path
        )
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
        transformer: A transformer object.
        train_data_dr: The training data tuple after dimensionality reduction.
    """
    logging.debug("Fit of dimensionality reduction met"
                  "hod {}".format(dimensionality_reduction_method))
    transformer = None
    transformer_params = dict()
    if dimensionality_reduction_method == "pca":
        components = method_params.get("components", 0.95)
        transformer, train_data_dr = fit_pca_to_task_train_data(
            train_data,
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
        _, _, train_data_dr, _ = decompose(transformed_data, decomposer_name,
                                           components=components)
        # Load decomposer object
        decomposer = load_ts_pca_transformer(
            decomposer_name,
            project_store_path=project_store_path
        )
    else:
        train_data_dr = decomposer.transform(transformed_data)
    return decomposer, (train_data_dr, data_y)


def trust_score_model_fit(train_data_dr, ts_classes, path_to_save=None,
                          **trust_score_params):
    """
    Fits a trust score model on the train data.
    Args:
        train_data_dr (tuple): train data X after decomposer fit and transform
        ts_classes (list): The list of the classes.
        path_to_save (str): The path to save the trust_score model after fit.
        **trust_score_params (dict): Parameters of the Trust Score Model.

    Returns:
        ts (trust_score model object):
    """
    logging.debug("TrustScore fit.")
    ts = TrustScore(**trust_score_params)
    train_X = train_data_dr[0]
    train_y = train_data_dr[1]
    ts.fit(train_X, train_y, classes=ts_classes)
    if path_to_save():
        dump(ts, open(path_to_save, "wb"))
    return ts


def trust_scores_model_score(data,
                             clf_predictions,
                             dimensionality_reduction_method,
                             transformer,
                             ts_model,
                             **params_score):
    """
    Computes and the trust_scores and the closest classes of predictions.
    Args:
        data:
        clf_predictions:
        dimensionality_reduction_method:
        transformer:
        ts_model:
        **params_score:

    Returns:

    """
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
                      train_data=None,
                      dimensionality_reduction_method="pca",
                      n_components=0.95,
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
                                         method_params=dm_params,
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
            project_store_path=project_store_path
        )
    score, closest_class = trust_scores_model_score(
        data,
        clf_predictions,
        dimensionality_reduction_method,
        transformer,
        ts_model,
        **trust_score_params_score
    )
    return score, closest_class
