import logging
from pathlib import Path
from pickle import dump, load
from alibi.confidence import TrustScore


def trust_score_model_fit(train_data_dr, ts_classes, path_to_save=None,
                          **trust_score_params):
    """
    Fits a trust score model on the train data.
    Args:
        train_data_dr (tuple): train data X after decomposer fit and transform
        ts_classes (list): The list of the classes.
        path_to_save (str): The path to save the trust_score model after fit.
        **trust_score_params (dict): Parameters of the Trust Score Model.
            -alpha: target fraction of instances to filter out.
            -filter_type: filter method; one of None (no filtering), distance_knn
            (first technique discussed in Overview) or probability_knn (second technique).
            -k_filter: number of neighbors used for the distance or probability
            based filtering method.
            -leaf_size: affects the speed and memory usage to build the k-d trees.
            The memory scales with the ratio between the number of samples and
            the leaf size.
            -metric: distance metric used for the k-d trees. Euclidean by default.
            -dist_filter_type: point uses th

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
                             ts_model,
                             **params_score):
    """
    Computes and the trust_scores and the closest classes of predictions.
    Args:
        data (np.array): The data to use Trust Score on (e.g windows)
        clf_predictions (list): The labels predictions that were computed by
        the classifier.
        ts_model (TrustScore object): The object created via the alibi library.
        **params_score:
        -k (int): The number of the nearest neighbors used for distance calculations.
        -dist_type (str): Options are "point"  (distance to k-nearest point)
        or "mean" (average distance from  the fist to the k-nearest point).


    Returns:
        score (list): A list of the calculated Trust Scores.
        closest_class (list): A list of the closest not predicted classes
        according to Trust Score.
    """
    logging.debug("Calculating Trust Score")
    logging.debug("type data {}".format(type(data)))
    logging.debug("shape data {}".format(data.shape))
    logging.debug("type clf_predictions {}".format(type(clf_predictions)))
    logging.debug("shape clf_predictions {}".format(clf_predictions.shape))
    logging.debug("clf_predictions {}".format(clf_predictions))
    logging.debug(params_score)
    score, closest_class = ts_model.score(data,
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
                      trust_score_classes=2,
                      trust_score_params_score=dict(),
                      scaler=None,
                      decomposer=None,
                      model_path=None):
    """

    Args:
        data (np.array): The data that the trust score model will be used for.
        clf_predictions (np.array): The classifier predictions.
        train_data (np.array): The train data for the training of the Trust Score
        model.
        trust_score_classes (int): The number of prediction classes.
        trust_score_params_score:
        scaler (sklearn Scaler Object): The scaler object to scale the
        train data.
        decomposer (sklearn Decomposer Object (PCA etc.)): The decomposer
        object that transforms the scaled trained data.
        model_path (str or Pathlib object): The path to save the trust_score
        model.
        trust_score_params_score (dict):
        -alpha: target fraction of instances to filter out.
        -filter_type: filter method; one of None (no filtering), distance_knn
        (first technique discussed in Overview) or probability_knn (second technique).
        -k_filter: number of neighbors used for the distance or probability
        based filtering method.
        -leaf_size: affects the speed and memory usage to build the k-d trees.
        The memory scales with the ratio between the number of samples and
        the leaf size.
        -metric: distance metric used for the k-d trees. Euclidean by default.
        -dist_filter_type: point uses th

    Returns:
        score (list): A list of the calculated Trust Scores.
        closest_class (list): A list of the closest not predicted classes
        according to Trust Score.
    """
    logging.debug("Calculating TrustScore")
    if scaler is not None:
        transformed_data = scaler.tranform(train_data)
        if decomposer is not None:
            transformed_data = decomposer.transform(transformed_data)
        ts_model = trust_score_model_fit(transformed_data, trust_score_classes,
                                         path_to_save=model_path)
    else:
        ts_model = trust_score_model_fit(train_data, trust_score_classes,
                                         path_to_save=model_path)

    score, closest_class = trust_scores_model_score(
        data,
        clf_predictions,
        ts_model,
        **trust_score_params_score
    )
    return score, closest_class
