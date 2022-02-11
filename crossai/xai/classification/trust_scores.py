import logging
from pathlib import Path
from pickle import dump, load
from alibi.confidence import TrustScore

from sklearn.model_selection import train_test_split


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

