"""
According to the arguments, it selects the corresponding model class.
"""

import logging
from crossai.models.nn1d.base_model import BaseModel
from crossai.models.nn1d.model_cnn1d import CNN1D
from crossai.models.nn1d.model_cnn1d import Multikernel
from crossai.models.nn2d.model_cnn2d import CNN2D
from crossai.models.nn1d.model_bilstm import BiLSTM
from crossai.models.nn1d.model_cnn1d import AppleModel
from crossai.models.nn1d.model_xceptiontime import XceptionTimeModel
from crossai.models.nn1d.model_inceptiontime import InceptionTimeModel
from crossai.models.nn1d.model_cnn_lstm import CNN1DLSTM, Conv1DLSTM


def model_select(task_configuration):
    if "nn_classifier" in task_configuration.keys():
        classifier_model_name = task_configuration["nn_classifier"]
        if classifier_model_name == "CNN-1D":
            classifier_model = CNN1D(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "CNN-2D":
            classifier_model = CNN2D(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "BiLSTM":
            classifier_model = BiLSTM(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "APPLE":
            classifier_model = AppleModel(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "MULTIKERNEL":
            classifier_model = Multikernel(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "CNN1DLSTM":
            classifier_model = CNN1DLSTM(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "CONV1DLSTM":
            classifier_model = Conv1DLSTM(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "XCEPTIONTIME":
            classifier_model = XceptionTimeModel(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        elif classifier_model_name == "INCEPTIONTIME":
            classifier_model = InceptionTimeModel(
                task_configuration,
                model_name=task_configuration.get("uuid", None))
        else:
            msg = "Unknown/Unspecified classifier {}. Using BaseModel as classifier".format(classifier_model_name)
            logging.warning(msg)
            classifier_model = BaseModel(task_configuration,
                                         model_name=task_configuration.get(
                                             "uuid", None))
        logging.info("Classifier : {}".format(classifier_model_name))
    else:
        msg = "nn_classifier field does ot exist in task configuration!"
        logging.error(msg)
        raise Exception(msg)
    return classifier_model
