"""
This file is responsible for loading the dataset which is stored in the disk or
in memory,
in a format that allows its usage in an ML or DL model.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tqdm import tqdm
from configuration_functions.project_configuration_variables \
    import project_configuration
from crossai.ts.processing.motion.features_extraction \
    import motion_features_extraction
from training.scaling import scale_dataset, transform_dataset
import logging
from training.dimensionality_reduction import decompose

from utilities.visualizations import plot_scatter, plot_dataframe_heatmap


class Dataset:
    """

    """

    def __init__(self):
        self.path_to_dataset_directory = \
            Path(project_configuration["project_directories"]["dataset"])
        self.X = None
        self.y = None
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        # Test_y as a vector (used in classification report)
        self.test_y_vec = None
        self.uuid = None
        self.shape = None
        self.number_of_instances = None
        self.number_of_classes = None
        # User friendly descriptions of labels (used in the confusion-matrix)
        self.descriptions = None
        self.input_instance_length = None
        self.labels_descriptions = None

    def set_data(self, X, y):
        """
        Used when the dataset is passed by reference and is not imported from
         filesystem
        Args:
            X (np.ndarray): Matrix in the form <datainstances>x <features/data>
            y (np.ndarray): labels_vector

        Returns:

        """
        self.X = X
        self.y = y

    def load_dataset_in_path(self, path_to_npz):
        """
        Load a dataset from a path.
        Args:
            path_to_npz: Path to an npz file containing a dataset with X,y in
            saved arrays

        Returns: None
            ndarray of the dataset that is stored in the disk
        """
        # TODO with pathlib define the filename and set the filename as uuid
        self.uuid = "loaded_dataset"
        data = np.load(path_to_npz)
        self.X = data["X"]
        self.y = data["y"]
        self.shape = self.X.shape[1:]
        self.number_of_instances = self.X.shape[0]
        self.number_of_classes = len(np.unique(self.y))

    def save_dataset(self):
        # TODO save X,y in npz format
        pass

    def load_dataset_by_uuid(self, uuid):
        """

        Args:
            uuid:

        Returns: None
            ndarray of the dataset that is stored in the disk
        """
        path_to_dataset = self.path_to_dataset_directory.joinpath(uuid +
                                                                  ".npz")
        path_to_datalen_txt = self.path_to_dataset_directory.joinpath(uuid +
                                                                      ".txt")
        self.uuid = uuid
        data = np.load(path_to_dataset, allow_pickle=True)
        self.input_instance_length = int(path_to_datalen_txt.read_text())
        self.X = data["X"]
        self.y = data["y"].astype(np.int32)
        self.shape = self.X.shape[1:]
        self.number_of_instances = self.X.shape[0]
        self.number_of_classes = len(np.unique(self.y))

    def labels_to_timesteps(self):
        """
        When the used model is BiLSTM, y-dataset should correspond to one
        value per timestep. This function transforms y vector to a matrix
        with shape Number_of_samples x rolling_window_size.
        This method can be used ONLY when the data have occured with rolling
         window.
        :return: None
            Transforms self.y
        """
        # rolling_window_size = self.shape[0]
        self.y = self.y[:, np.newaxis] * np.ones((self.number_of_instances,
                                                  self.input_instance_length))

    def labels_one_hot(self):
        # n_values = np.max(self.y) + 1
        # self.y_one_hot = np.eye(n_values)[self.y]
        # TODO fix the all-zero bug
        if len(self.y.shape) == 1:
            self.y = np.eye(self.number_of_classes)[self.y]

    def shuffle_dataset(self):
        """shuffles the dataset X and y.
        Should be called right after the dataset loaading.

        Returns:
            None:
            Modifies self.X and self.y

        """
        self.X, self.y = shuffle(self.X, self.y, random_state=42)

    def last_n_percent(self, n=0.2):
        """Returns the last n percent of the dataset.
        Currently not used.
        Args:
            n (float): Percent as float (range 0 to 1) of the dataset values
             to return.

        Returns:
            type: Description of returned object.

        """
        size = np.ceil(n * self.number_of_instances).astype(np.int32)
        return self.X[-size:], self.y[-size:]

    def load_labels_descriptions(self, dataset_labels_dict):
        """From a given dict_label, it generates the description for each
        class. It is expected that the values in y have occurred from the given
        dict_label that should originate from the project configuration file
        in the corresponding directory.

        Args:
            dataset_labels_dict (dict): Description of parameter `dict_label`.

        Returns:
            type: Description of returned object.

        """
        self.descriptions = list(dataset_labels_dict.values())
        self.descriptions.sort()


class DatasetSpectrogram(Dataset):
    def resize_dataset(self):
        pass

    def stack_axes(self):
        pass


def save_dataset(saveid, train_X, test_X, train_y, test_y, path_to_save=None):
    logging.info("Saving dataset with configuration uuid {}".format(saveid))
    filename = saveid + ".npz"
    if path_to_save is None:
        path_to_save = Path(project_configuration["project_store_path"]). \
            joinpath("datasets").joinpath(filename)
    # Create a dictionary where each key name is one of the given arguments.
    save_dict = dict()

    np.savez_compressed(path_to_save,
                        train_X=train_X,
                        test_X=test_X,
                        train_y=train_y.astype(np.int32),
                        test_y=test_y.astype(np.int32))
    logging.info("")
    logging.info("Dataset {} saved!".format(path_to_save))
    logging.info("")
    logging.info("----------------")
    logging.info("")


def load_dataset_npz(preprocessing_uuid, path_to_datasets_dir=None):
    """
    Loads the dataset from the default project dataset directory. The dataset
     should be
    in NPZ format and contain the keys train_X, test_X, train_y, test_y.
    Args:
        preprocessing_uuid (str): The uuid of the preprocessing task, according
         to which the dataset
            was generated.
        path_to_datasets_dir(pathlib.Path or str, optional): Path to the
         datasets directory
    Returns:
        train_X, test_X, train_y, test_y np arrays.
    """
    dataset_file_name = "{}.npz".format(preprocessing_uuid)
    project_datasets_dir = Path(project_configuration["project_store_path"]). \
        joinpath("datasets")
    path_to_project_data = project_datasets_dir.joinpath(dataset_file_name)
    if not project_datasets_dir.is_dir():
        project_datasets_dir.mkdir(parents=True)
    if path_to_datasets_dir is None:
        path_to_data = Path(project_configuration["project_store_path"]). \
            joinpath("datasets").joinpath(dataset_file_name)
    else:
        path_to_data = Path(path_to_datasets_dir).joinpath(dataset_file_name)
        shutil.copy(str(path_to_data), str(path_to_project_data))
    if path_to_data.is_file():
        data = np.load(path_to_data)
    else:
        msg = "{} does not exist!". \
            format(path_to_data.relative_to(
            project_configuration["project_store_path"]))
        logging.error(msg)
        raise Exception(msg)
    return data


def dataset_postprocessing(train_X, test_X, configuration):
    if configuration.get("scaling"):
        scaling_configuration = {
            "uuid": configuration["processing_uuid"],
            "scaling": configuration["scaling"]
        }
        train_X, scaler = scale_dataset(scaling_configuration, train_X)
        test_X = transform_dataset(scaling_configuration, test_X)

    else:
        logging.warning("No scaling option found in configuration."
                        " Dataset will not be scaled!")
    if configuration.get("dimensionality_reduction") is not None:
        # TODO review configuration template for dimensionality reduction.
        decomposer, dimensions, train_X, test_X = decompose(
            train_data=train_X,
            process_uuid=configuration["processing_uuid"],
            components=configuration.get("n_components"),
            test_data=test_X)

    return train_X, test_X


def dataset_split(dataset_X, dataset_y, train_size=None):
    """
    Splits a dataset in train and test sets according to the configuration
    paramaters.
    Args:
        dataset_X (np.ndarray): The dataset features' values.
        dataset_y (np.ndarray): The dataset labels.
        train_size (float): The percentage of the split.

    Returns:
        train_X (np.ndarray): The train dataset features' values.
        test_X (np.ndarray): The test dataset features' values.
        train_y (np.ndarray): The train dataset labels.
        test_y (np.ndarray): The test dataset labels.

    """
    if train_size is None:
        train_size = project_configuration["processing"].get("train_size")
    shuffle_dataset = project_configuration["processing"].get("shuffle") if \
        project_configuration["processing"].get("shuffle") else True
    random_state = project_configuration.get("random_state", 42)
    stratify = True if shuffle_dataset else False
    logging.debug("Splitting dataset in train/test with parameters :\n\t"
                  "train_size : {}\n\tshuffle: {}\n\t"
                  "Random state: {}\n\tStratify: {}\n".
                  format(train_size, shuffle_dataset, random_state, stratify))
    train_X, test_X, train_y, test_y = train_test_split(
        dataset_X,
        dataset_y,
        shuffle=shuffle_dataset,
        train_size=train_size,
        random_state=random_state,
        stratify=dataset_y if shuffle_dataset else None)
    logging.info("train_X dimensions : {}".format(train_X.shape))
    logging.info("test_X dimensions : {}".format(test_X.shape))

    return train_X, test_X, train_y, test_y


def label_by_movement_id(labels_dict, labels, movement_id):
    """
    A dictionary matches each movement id to an accepted label.
    Then the returned label occurs according to the label index in the
    labels array.
    """
    train_label = labels_dict[movement_id]
    try:
        label_ind = labels.index(train_label)
    except ValueError:
        raise Exception("Invalid movementID!")
    return label_ind


def dataset_features_extraction(dfs_list, dfs_labels, features_categories,
                                dataset_domain,
                                features_extraction_parameters=None):
    """
    Calculates features according to the given categories from a list of
    DataFrames that have occurred from
    a motion dataset.
    Args:
        dfs_list (list):
        dfs_labels (numpy array):
        features_categories (list):
        dataset_domain (str): Description of the dataset category (motion,
         signal, timeseries, text, audio)
        features_extraction_parameters (dict):
    Returns:
        df_feats (pandas DataFrame): A dataframe that holds the selected
         features or categories of features
        and the label of the instance.
    """
    if features_extraction_parameters is None:
        features_extraction_parameters = \
            project_configuration["processing"].get(
                "features_extraction_parameters")
        if features_extraction_parameters is None:
            msg = "Attempting features extraction but no " \
                  "parameters have been defined in configuration."
            logging.error(msg)
            raise Exception(msg)
    features_dicts_list = list()
    for df, label in tqdm(zip(dfs_list, dfs_labels)):
        feats = None
        if df.shape[0] != 0:
            if dataset_domain == "motion":
                feats = motion_features_extraction(
                    df,
                    feature_categories=features_categories,
                    sampling_frequency=100,
                    features_extraction_parameters=
                    features_extraction_parameters)
            else:
                msg = "{} feature extraction is not yet" \
                      " implemented or supported.".format(dataset_domain)
                logging.error(msg)
                raise Exception(msg)
            feats.update({"label": label})
        else:
            msg = "Cannot calculate features. DataFrame has no data."
            logging.warning(msg)
        if feats is not None:
            features_dicts_list.append(feats)
    features_df = pd.DataFrame(features_dicts_list)
    return features_df


def pca_x_components_analysis(data, components, labels, path_to_save,
                              **kwargs):
    logging.info("Performing PCA - 2 components analysis")
    pca_x = PCA(components)
    components_str = str(components)
    if isinstance(components, float):
        components_str = components_str.replace(".", "_")
    pca_x.fit(data)
    pca_x_data = pca_x.transform(data)
    logging.info("Data shape after transformation "
                 ": {}".format(pca_x_data.shape))
    if components == 2:
        plot_scatter([pca_x_data[:, 0], pca_x_data[:, 1]],
                     xlabel="First principal component",
                     ylabel="Second Principal Component",
                     c=labels,
                     path_to_save=path_to_save.joinpath(
                         "pca_{}_components.png".format(components_str)))
    elif components == 3:
        plot_scatter([pca_x_data[:, 0], pca_x_data[:, 1], pca_x_data[:, 2]],
                     xlabel="First principal component",
                     ylabel="Second Principal Component",
                     zlabel="Third Principal Component",
                     c=labels,
                     path_to_save=path_to_save.joinpath(
                         "pca_{}_components.png".format(components_str)))
    else:
        logging.info("Components larger than 3. Cannot plot scatterplot.")
    logging.info("PCA - {} components explained"
                 " variance : {}".format(components_str,
                                         pca_x.explained_variance_))
    logging.info(
        "PCA - {} components explained variance"
        " ratio : {}".format(components_str, pca_x.explained_variance_ratio_))
    logging.info("")
    logging.info("Length of each component"
                 " (It should be the same as the features extracted):")
    for i in range(0, pca_x.n_components_):
        logging.info(
            'Component {} length: {}'.format((i + 1),
                                             len(abs(pca_x.components_[i]))))
    logging.info("PCA - {0} - "
                 "Noise variance : {1:.4f}".format(components_str,
                                                   pca_x.noise_variance_))
    columns = kwargs.get("columns", None)
    pca_x_components = pd.DataFrame(pca_x.components_, columns=columns)
    plot_dataframe_heatmap(data=pca_x_components,
                           path_to_save=path_to_save.joinpath(
                               "pca_{}_components_"
                               "data_frame.png".format(components_str)))
    logging.info("")
    logging.info("Component maximum / minimum values")
    for i in range(0, pca_x.n_components_):
        logging.info('Component {} maximum'
                     ' value {}'.format(i + 1, pca_x_components.iloc[i].max()))
        logging.info('Component {} minimum'
                     ' value {}'.format(i + 1, pca_x_components.iloc[i].min()))
    pca_x_components_abs = pd.DataFrame(abs(pca_x.components_),
                                        columns=columns)
    plot_dataframe_heatmap(data=pca_x_components,
                           path_to_save=path_to_save.joinpath(
                               "pca_{}_components_data_"
                               "frame_abs_values.png".format(components_str)))
    logging.info("")
    logging.info("Component maximum / minimum values")
    for i in range(0, pca_x.n_components_):
        logging.info('Component {} maximum absolute value {}'
                     ''.format(i + 1, pca_x_components_abs.iloc[i].max()))
        logging.info('Component {} minimum absolute value {}'
                     ''.format(i + 1, pca_x_components_abs.iloc[i].min()))


def components_analysis(dataset_X, dataset_y, path_to_save):
    logging.debug("Splitting dataset in train/test with train size {}".
                  format(project_configuration.get("train_size")))
    train_X, test_X, train_y, test_y = train_test_split(
        dataset_X,
        dataset_y,
        train_size=project_configuration.get("train_size"),
        random_state=project_configuration.get("random_state"),
        stratify=dataset_y)
    logging.debug("train_X shape : {}".format(train_X.shape))
    logging.debug("test_X shape : {}".format(test_X.shape))

    logging.info("Scaling dataset ..")
    scaler = StandardScaler()
    scaler.fit(train_X)

    x_train_scaler = scaler.transform(train_X)
    test_scaler = scaler.transform(test_X)

    logging.info("Ploting the relation between the Variance"
                 " and the Dimensions")

    pca = PCA()
    pca.fit(x_train_scaler)
    cumsum = np.cumsum(pca.explained_variance_ratio_) * 100
    d = [n for n in range(len(cumsum))]
    plt.figure(figsize=(10, 10))
    plt.plot(d, cumsum, color='red', label='cumulative explained variance')
    plt.title('Cumulative Explained Variance as a Function of the Number'
              ' of Components')
    plt.ylabel('Cumulative Explained variance')
    plt.xlabel('Principal components')
    plt.axhline(y=95, color='k', linestyle='--',
                label='95% Explained Variance')
    plt.legend(loc='best')
    plt.savefig(path_to_save.joinpath("explained_variance.png"))
    plt.close()

    pca_x_components_analysis(x_train_scaler, 2, train_y, path_to_save,
                              columns=dataset_X.columns)
    pca_x_components_analysis(x_train_scaler, 3, train_y, path_to_save,
                              columns=dataset_X.columns)
    pca_x_components_analysis(x_train_scaler, 0.95, train_y, path_to_save,
                              columns=dataset_X.columns)
