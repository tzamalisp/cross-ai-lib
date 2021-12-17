from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pickle import dump
from pickle import load
from sklearn.svm import SVC
import pandas as pd
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datetime import datetime
from bunch import Bunch
import logging
from configuration_functions.project_configuration_variables import project_configuration


class MLClassifier:
    def __init__(self, config, model_name=None):
        self.model_name = datetime.now().strftime("%Y-%m-%d-%H-%M") if model_name is None else model_name
        self.config = Bunch(config)
        self.classifier = None

    def fit(self, train_dataset, validation_dataset):
        self.classifier.fit(train_dataset, validation_dataset)

    def load(self):
        path_to_models = Path(project_configuration.get("project_store_path")).joinpath("models")
        models_paths_list = [i for i in path_to_models.glob("*{}*".format(self.model_name))]
        if not models_paths_list:
            msg = "Model {} does not exist in models directory".format(self.model_name)
            logging.error(msg)
            raise Exception(msg)
        model_path = models_paths_list[0]
        self.classifier = load(open(str(model_path), "rb"))

    def predict(self, data, labels=None):
        logging.debug("Model predictions..")
        if isinstance(data, pd.DataFrame):
            data = data.values
        predictions = self.classifier.predict(data)
        predictions = pd.DataFrame(predictions, columns=labels)
        return predictions

    def hyperparameters_tuning(self):
        pass


class SVM(MLClassifier):
    def __init__(self, *args, **kwargs):
        gamma = kwargs.get("gamma", "auto")
        probability = kwargs.get("probability", True)
        self.classifier = SVC(gamma=gamma, probability=probability)
        super().__init__(*args, **kwargs)