import logging

import numpy as np
import pandas as pd

from src.processing.timeseries.motion.project_processing_motion_variables\
    import axes_acc, axes_gyro, accepted_keys_to_generic


def get_motion_signals_data_from_document(dataset_document):

    """
    Accepts a mongodb document and returns a dataframe. The length of data is
    provided by `datalen` field which is expected to be calculated in mongo
    project querie. It converts the keys of the dictionary to generally
    accepted motion sensor key names.
    Args:
        dataset_document: dictionary document from mongodb

    Returns:
        Pandas dataframe with the available axes in the dataset_document.

    """
    dataset_dict = (dataset_document.get("data"))
    dataset_df = pd.DataFrame.from_dict(dataset_dict)
    datalen = dataset_document.get("datalen")
    if datalen is None:
        datalengths = list()
        for key in dataset_dict.keys():
            keylen = np.count_nonzero(~np.isnan(dataset_dict[key]))
            datalengths.append(keylen)
        datalen = np.min(datalengths)
    df_dict = dict()
    for key in dataset_dict.keys():
        if key in list(accepted_keys_to_generic.keys()) or key in\
                axes_acc+axes_gyro:
            # this way all document keys would be converted to generic keys
            if len(dataset_dict[key]) > datalen:
                # logging.warning("Disagreement on DataFrame axes length.
                # Rejecting values to obtain the accepted len.")
                dataset_dict[key] = dataset_dict[key][:datalen]
            if key not in axes_acc+axes_gyro:
                df_dict[accepted_keys_to_generic[key]] = dataset_dict[key]
            else:
                df_dict[key] = dataset_dict[key]
    df = pd.DataFrame.from_dict(df_dict)
    return df


def recreate_signal_column_names(axes):
    """
    Given a list of signal names regarding motion (accelerometer and/or
    gyroscope) the signal names are recreated by replacing the acc name in axes
    name with each of the accelerometer signals (x, y, z).E.g. if axes contains
    the axes category `filter_acc`, then the new list will contain
    `filter_acc_x`, `filter_acc_y`, `filter_acc_z`.
    Args:
        axes (list): Contains strings that should contain either `acc` or
        `gyr` substrings.

    Returns:
        A list with all the signals that occur from the categories.
    """
    # Recreate the axes column names
    axes_signals = list()
    for axes_category in axes:
        if "acc" in axes_category:
            for signal in axes_acc:
                axes_signals.append(axes_category.replace("acc", signal))
        if "gyr" in axes_category:
            for signal in axes_gyro:
                axes_signals.append(axes_category.replace("gyr", signal))
    return axes_signals


def recreate_dataframe_and_append_signals(instance, axes, axes_signals):
    """
    Function to recreate a dataframe from an instance of the dataset and
    further add the magnitude signal and the sum signal.
    """
    df = pd.DataFrame(instance, columns=axes_signals)
    accumulated_signals = list()
    for axes_category in axes:
        signals_cat = list()
        if "acc" in axes_category:
            for signal in axes_acc:
                signals_cat.append(axes_category.replace("acc", signal))
            accumulated_signals.append(signals_cat)
        if "gyr" in axes_category:
            for signal in axes_gyro:
                signals_cat.append(axes_category.replace("gyr", signal))
            accumulated_signals.append(signals_cat)
    for signals, signals_category in zip(accumulated_signals, axes):
        col_name = signals_category+"_magnitude"
        df[col_name] = np.apply_along_axis(lambda x:
                                           np.sqrt(np.power(x, 2).sum()), 1,
                                           df[signals].values)
        col_name = signals_category+"_sum"
        df[col_name] = np.apply_along_axis(lambda x: np.sum(x),
                                           1, df[signals].values)
    return df


def calculate_magnitude(array, axis=1):
    """
    Calculates the magnitude of a given ndarray.
    Args:
        array (numpy.ndarray): numpy array holding the data
        axis (int 0,1): axis of np array to calculate magnitude

    Returns:
        (numpy.ndarray) the magnitude of the values of the input array
    """
    return np.apply_along_axis(lambda x: np.sqrt(np.power(x, 2).sum()),
                               axis, array)


def calculate_sma(array, axis=1):
    """
    Calculates the sma (signal magnitude area) of a given ndarray.
    Args:
        array (numpy.ndarray): numpy array holding the data
        axis (int 0,1): axis of np array to calculate magnitude

    Returns:
        (numpy.ndarray) the magnitude of the values of the input array
    """
    return np.apply_along_axis(lambda x: np.abs(x).sum(), axis, array)


def calculate_signal_duration(samples, sampling_frequency):
    """
    Calculates the duration of a signal. Main hypothesis is that sampling is
    uniform in time.
    Args:
        samples (int): Number of values of signal.
        sampling_frequency (float): The frequency of the sampled signal.

    Returns:
        duration (float): duration of signal (by default in seconds if
        frequency is expressed in cycles per second).
    """
    return samples / sampling_frequency


def append_instances(dfs_list):
    """
    Creates a new dataframe with the acc and gyroscope axes of all the
    instances in the list.
    Args:
        dfs_list (list): List of DataFrames.

    Returns:
        type: Description of returned object.

    """
    new_df = dict()
    for signal_name in axes_acc + axes_gyro:
        new_df[signal_name] = list()
    for instance in dfs_list:
        for signal_name in axes_acc + axes_gyro:
            if signal_name in instance.columns:
                new_df[signal_name].append(instance[signal_name].values)
    for signal_name in axes_acc + axes_gyro:
        new_df[signal_name] = np.hstack(new_df[signal_name])
    new_df = pd.DataFrame.from_dict(new_df)
    return new_df


class Segment:
    def __init__(self, start, stop, label=None):
        """

        Args:
            start:
            stop:
            label:
        """
        self.start = int(start)
        self.stop = int(stop)
        self.data_segment = None
        self.label = label
        # Fields used by the prediction tool
        self.predictions = None
        self.prediction_label = None
        self.prediction_value = None

    def __repr__(self):
        if self.label is not None:
            s_str = "[ {} - {}: {}]".format(self.start, self.stop, self.label)
        else:
            s_str = "[ {} - {}]".format(self.start, self.stop)
        return s_str


class SegmentsCollection:
    """
    Fundamental class that represents information for a signal or collection
     of signals,
    regarding their label at specific parts of the waveforms.
    """

    def __init__(self):
        self._segments = list()
        self._size = 0
        self.data = None

    def set_data(self, data):
        """
        Sets the data to correspond to the SegmentsCollection. This is optional in most cases, however it is mandatory
            if it is needed to visualize the SegmentsCollection.
        Args:
            data:

        Returns:

        """
        self.data = data

    def sort_segments(self):
        """
        Classic bubblesort to order segments according to their start index

        """
        segmentsnr = len(self._segments)
        sort_times = 2
        for t in range(sort_times):
            for i in range(segmentsnr):
                for j in range(0, segmentsnr - i - 1):
                    if self._segments[j].start > self._segments[j + 1].start:
                        self._segments[j], self._segments[j + 1] = self._segments[j + 1], self._segments[j]

    def add(self, start, stop, label):
        """
        Adds a segment by giving as input 3 arguments, `start`, `stop`, `label`
        Args:
            start(int):
            stop(int):
            label(str or int):

        Returns:

        """
        seg = Segment(start, stop, label)
        self._segments.append(seg)
        self._size += 1

    def add_segment(self, seg):
        self._segments.append(seg)
        self._size += 1

    def export_to_array(self):
        exported_array = list()
        for seg in self._segments:
            seg_element = list()
            seg_element.append(seg.start)
            seg_element.append(seg.stop)
            seg_element.append(str(seg.label))
            exported_array.append(seg_element)
        return exported_array

    def export_segments_labels_to_csv(self, filename):
        """

        Args:
            filename (pathlib.Path or str):

        Returns:

        """

        segments_labels_list = list()

        for seg in self._segments:
            segment_dict = dict()
            segment_dict["start"] = seg.start
            segment_dict["stop"] = seg.stop
            segment_dict["label"] = seg.label
            segments_labels_list.append(segment_dict)
        logging.debug("Creating labels dataframe")
        df = pd.DataFrame(segments_labels_list)
        df.to_csv(filename, sep=";")

    def import_segments_from_csv(self, filename):
        df = pd.read_csv(filename, sep=";")
        for ind, row in df.iterrows():
            self.add(row[1], row[2], row[3])

    def to_df(self, columns=None, labels_names=None):
        """
        Converts the segment collection to pandas.DataFrame
        Args:
            columns (list): Which fields of the collection to be added as columns to the DataFrame
            labels_names (list, optional): If defined, the label names will be added in the corresponding column.
                Used in case of a previous conversion of labels to integers.

        Returns:
            collection_df (pandas.DataFrame): A DataFrame that by default has columns `start`, `stop`, `label`.
        """
        collection_dict = dict()
        collection_dict["start"] = list()
        collection_dict["stop"] = list()
        collection_dict["label"] = list()
        if columns:
            for col_name in columns:
                if col_name in list(vars(self._segments[0]).keys()):
                    collection_dict[col_name] = list()

        for seg in self._segments:
            collection_dict["start"].append(seg.start)
            collection_dict["stop"].append(seg.stop)

            seg_label = seg.label
            if isinstance(seg.label, int):
                if labels_names:
                    seg_label = labels_names[seg.label]
            collection_dict["label"].append(seg_label)
            for col_name, col_val in vars(seg):
                if col_name in list(collection_dict.keys()):
                    collection_dict[col_name].append(col_val)

        collection_df = pd.DataFrame.from_dict(collection_dict)
        return collection_df

    def drop_unused_labels(self, labels):
        """
        Given a list of labels, recreate the segments list with only the segments instances that
        have a label included in the `labels` argument.
        Args:
            labels (list): List of the accepted labels that should be maintained. Each element should
                contain the descriptive name of the label (str), not the integer index.

        Returns:

        """
        new_segments = list()
        for seg in self._segments:
            if seg.label in labels or isinstance(seg.label, int):

                new_segments.append(seg)
        self._size = len(new_segments)
        self._segments = new_segments

    def convert_labels_to_indices(self, labels):
        """
        Given a list of labels, convert the labels of the collection to the corresponding integer index
        of each class. This prerequisites to have the segment collection labels as descriptive names.
        Args:
            labels (list): List of the accepted labels that should be maintained. Each element should
                contain the descriptive name of the label (str), not the integer index.

        Returns:

        """
        for seg in self._segments:
            seg.label = labels.index(seg.label)

    def add_segment_collection(self, labeled_list):
        """
        Creates a SegmentsCollection object from a labeled segments list.
        Args:
            labeled_list:

        Returns:

        """

        for segment in labeled_list:
            self.add(segment[0], segment[1], segment[2])
            self._size += 1