import logging
from pathlib import Path
from pickle import dump, load

import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler,\
    QuantileTransformer
from sklearn.pipeline import Pipeline


class Scaler:
    """
        A class fitting a scaler object to the data and saving the scaler object.
    """

    def __init__(self, data, path_to_scaler_dir, scaler_name, method="standardization"):
        self.data = data
        self.scaler = None
        self.scaler_name = scaler_name
        self.scaler_path = Path(path_to_scaler_dir)
        self.method = method
        self.scaling()

    def scaling(self):
        # First pass
        # read the data in batches
        if self.method == "standardization":
            self.scaler = StandardScaler()
        elif self.method == "normalization":
            self.scaler = MinMaxScaler()
        elif self.method == "gaussianized":
            scale_pip = [('minmax', MinMaxScaler()), ('quantile', QuantileTransformer(n_quantiles=1000))]
            pipeline = Pipeline(steps=scale_pip)
            self.scaler = pipeline
        else:
            msg = "Unknown or yet unsupported scaling method {}!".format(self.method)
            logging.error(msg)
            raise Exception(msg)
        counter = 0
        if len(self.data.shape) <= 2:
            logging.debug("Scaling data with maximum dimension 2")
            self.scaler.fit(self.data)
        else:
            """
            https://stackoverflow.com/a/59601298/2793713
            """
            logging.debug("Scaling data with dimension > 2")
            # logging.warning("Dangerous to use for dataset dimension larger than 3.")
            self.scaler.fit(self.data.reshape(-1, self.data.shape[-1]))
        logging.debug("Scaling fit completed.")
        # save the scaler
        save_path = self.scaler_path.joinpath(self.scaler_name)
        dump(self.scaler, open(save_path, "wb"))
        logging.debug("Scaler saved successfully with the name: {}".format(self.scaler_name))


class Transformer:
    """
    A class for loading the scaler saved object and transforming the data with
    respect to the scaler train.
    """

    def __init__(self, data, path_to_scaler_dir, scaler_name):
        self.data = data
        self.scaler = None
        self.scaler_name = scaler_name
        self.scaler_path = Path(path_to_scaler_dir)

    def transforming(self):
        """ Load the scaler """
        logging.debug("Loading the scaler..")
        import_path = self.scaler_path.joinpath(self.scaler_name)
        self.scaler = load(open(import_path, "rb"))
        if len(self.data.shape) <= 2:
            logging.debug("Transforming data with maximum dimension 2")
            data_scaled = self.scaler.transform(self.data)
        else:
            """
            https://stackoverflow.com/a/59601298/2793713
            """
            logging.debug("Transforming data with dimension > 2")
            data_shape = self.data.shape
            data_scaled = self.scaler.transform(self.data.reshape(-1, self.data.shape[-1])) \
                .reshape(data_shape)
        logging.debug("Data scale transformation completed.")
        return data_scaled


def scale_image(image_array):
    """
    Scales the pixels of an image from [0..255] to [0..1]
    Args:
        image_array: numpy array corresponding to each pixel intensity

    Returns: scaled image_array

    """
    return image_array / 255


def scale_dataset(config, dataset, scaler_path, save=True):
    """

    Args:
        config: Should contain the keys `uuid` and `scaling`.
        dataset:
        scaler_path:
        save:

    Returns:

    """
    scaler_name = scaler_path
    if isinstance(scaler_name, int):
        scaler_name = str(scaler_name)
    if not scaler_path.joinpath("{}".format(scaler_name)).exists():
        scaler = Scaler(data=dataset, path_to_scaler_dir=scaler_path,
                        scaler_name=scaler_name, method=config["scaling"])
    else:
        scaler = Transformer(data=dataset, path_to_scaler_dir=scaler_path,
                             scaler_name=scaler_name)
    # Normalization of the data - transforming
    transformer = Transformer(data=dataset, path_to_scaler_dir=scaler_path,
                              scaler_name=scaler_name)
    scaled_dataset = transformer.transforming()

    if config.get("scaling") == "standardization":
        logging.debug("Scaled data mean value : {0:.5f}".format(scaled_dataset.mean()))
        logging.debug("Scaled data STD : {0:.5f}".format(scaled_dataset.std()))
    elif config.get("scaling") == "normalization":
        logging.debug("Scaled data max value : {0:.5f}".format(scaled_dataset.max()))
        logging.debug("Scaled data min value : {0:.5f}".format(scaled_dataset.min()))
    logging.debug("ndarray_gestures scaled data shape: {}".format(scaled_dataset.shape))
    logging.debug("ndarray_gestures scaled data type:".format(scaled_dataset.dtype))
    logging.debug("ndarray_gestures scaled data dimensions:".format(scaled_dataset.ndim))

    return scaled_dataset, scaler.scaler


def transform_dataset(config, dataset, scaler_path, transformer=None):
    """
    Loads a scaler object according to classification task and transforms the data.
    Args:
        config: classification task configuration
        dataset:
        transformer:
    Returns:

    """
    if transformer is None:

        scaler_name = scaler_path
        logging.debug("Loading scaler {}".format(scaler_name))
        transformer = Transformer(data=dataset, path_to_scaler_dir=scaler_path,
                                  scaler_name=scaler_name)
        logging.debug("Scaling with loaded scaler.")
        transformed_data = transformer.transforming()
    else:
        logging.debug("Scaling with scaler passed as argument.")
        transformed_data = transformer.transform(dataset)
    if config.get("scaling") == "standardization":
        logging.debug("Transformed data mean value : {0:.5f}".format(transformed_data.mean()))
        logging.debug("Transformed data STD : {0:.5f}".format(transformed_data.std()))
    elif config.get("scaling") == "normalization":
        logging.debug("Transformed data max value : {0:.5f}".format(transformed_data.max()))
        logging.debug("Transformed data min value : {0:.5f}".format(transformed_data.min()))
    return transformed_data


def scale_data(config, data, transf_to_df=None):
    """
    A function that performs standard scaling to the argument features
    Args:
        transf_to_df: (boolean) True if it is desired to transform the scaled features array back to dataframe.
        config: (dict) The dictionary of the configuration file
        data: (dataframe) The dataframe of the features to be scaled

    Returns:

    """
    logging.info("Scaling dataset...")
    project_dir = Path(config["project_directory"])
    scaler_name = config["scaler_name"]
    if scaler_name == "standardization":
        scaler = StandardScaler()
        scaler.fit(data)
        data_scaled = scaler.transform(data)
        if not project_dir.joinpath("/scaler_{}.pkl".format(scaler_name)).is_file():
            dump(scaler, open(project_dir.joinpath("/scaler_{}.pkl".format(scaler_name)), "wb"))
            logging.info("Scaler object stored successfully.")
        if transf_to_df is True:
            # Transform to dataframe
            data_scaled = pd.DataFrame(data=data_scaled, columns=data.columns)
    else:
        # todo: add more scalers in the future
        logging.error("Not a valid scaler has been declared")

    return data_scaled
