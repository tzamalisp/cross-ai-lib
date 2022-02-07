import logging
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from processing.motion.plot_motion_sensors_data import plot_spectrogram
from utilities.visualizations import fig2img, merge_pil_images


def pad_list_to_length(array, length, pad_to="both", mode=""):
    if mode == "expand":
        top = array[-1]
    else:
        top = 0
    array_added_elements = length - len(array)
    array += [top for i in range(0, array_added_elements)]
    return array


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

    def __iter__(self):
        """
        Returns the iterator object.
        """
        return SegmentsCollectionIterator(self)

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

    #  The rest of the functions below are used for compatibility with SegmentationTool.
    def add_plot_selection(self, start, stop):
        """ Creates/Updates the plot output. """

        fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        axs[0].plot(self.data)
        for seg in self._segments:
            axs[0].plot(self.data.iloc[seg.start:seg.stop])
        axs[1].clear()
        axs[1].plot(self.data)
        axs[1].plot(self.data.iloc[start:stop])

    def split(self, index, value, label1, label2):
        self.add(value, self._segments[index].stop, label1)
        self._segments[index].stop = value
        self._segments[index].label = label2

    def split_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        x = self._segments[index].start,
        y = self._segments[index].stop
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segments')

        # Resulting segments
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        ax2.plot(self.data.iloc[self._segments[index].start:value])
        ax2.plot(self.data.iloc[value:self._segments[index].stop])
        plt.title('Resulting segments')

    def pop(self, ind):
        print('Segment ', ind, ' deleted.')
        self._segments.pop(ind)
        self._size -= 1

    def pop_plot_selection(self, index):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

    def modify(self, index, start, stop, label):
        self._segments[index].start = start
        self._segments[index].stop = stop
        self._segments[index].label = label

    def modify_plot_selection(self, index, start, stop):
        """ Creates/Updates the plot output. """
        fig, axs = plt.subplots(2, 1, figsize=(14, 12))
        axs[0].plot(self.data)
        for seg in self._segments:
            axs[0].plot(self.data.iloc[seg.start:seg.stop])
        axs[1].clear()
        axs[1].plot(self.data)
        axs[1].plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        axs[1].plot(self.data.iloc[start:stop])

    def shift_left(self, ind, value):

        max_index_bound = self._segments[ind].start
        # Check for invalid value
        if max_index_bound - value < 0:
            return False
        else:
            self._segments[ind].start -= value
            self._segments[ind].stop -= value

    def shift_left_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Shifted segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start - value
        y = self._segments[index].stop - value
        if x > 0 and y > 0:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def shift_right(self, ind, value):

        max_index_bound = self._segments[ind].stop
        # Check for invalid value
        if max_index_bound + value > self.data.shape[0]:
            return False
        else:
            self._segments[ind].start += value
            self._segments[ind].stop += value

    def shift_right_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Shifted segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start + value
        y = self._segments[index].stop + value
        if x < self.data.shape[0] and y < self.data.shape[0]:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def expand_left(self, ind, value):

        max_index_bound = self._segments[ind].start
        # Check for invalid value
        if max_index_bound - value < 0:
            return False
        else:
            self._segments[ind].start -= value

    def expand_left_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Expanded segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start - value
        y = self._segments[index].stop
        if x > 0:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def expand_right(self, ind, value):

        max_index_bound = self._segments[ind].stop
        # Check for invalid value
        if max_index_bound + value > self.data.shape[0]:
            return False
        else:
            self._segments[ind].stop += value

    def expand_right_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Expanded segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start
        y = self._segments[index].stop + value
        if y < self.data.shape[0]:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def expand_both(self, ind, value):

        max_index_bound = self._segments[ind].stop
        min_index_bound = self._segments[ind].start
        # Check for invalid value
        if (max_index_bound + value > self.data.shape[0] or
                min_index_bound - value < 0):
            return False
        else:
            self._segments[ind].start -= value
            self._segments[ind].stop += value

    def expand_both_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Expanded segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start - value
        y = self._segments[index].stop + value
        if x > 0 and y < self.data.shape[0]:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def reduce_left(self, ind, value):

        max_index_bound = self._segments[ind].start
        # Check for invalid value
        if max_index_bound + value > self._segments[ind].stop:
            return False
        else:
            self._segments[ind].start += value

    def reduce_left_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Reduced segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start + value
        y = self._segments[index].stop
        if x < y:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def reduce_right(self, ind, value):

        max_index_bound = self._segments[ind].stop
        # Check for invalid value
        if max_index_bound - value < self._segments[ind].start:
            return False
        else:
            self._segments[ind].stop -= value

    def delete_segment(self):
        print("TODO")

    def reduce_right_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Reduced segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start
        y = self._segments[index].stop - value
        if x < y:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def reduce_both(self, ind, value):

        max_index_bound = self._segments[ind].stop
        min_index_bound = self._segments[ind].start
        mean = (min_index_bound + max_index_bound) / 2

        # Check for invalid value
        if (max_index_bound - value < mean or
                min_index_bound + value > mean):
            return False
        else:
            self._segments[ind].start += value
            self._segments[ind].stop -= value

    def reduce_both_plot_selection(self, index, value):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index].start:self._segments[index].stop])
        plt.title('Selected segment')

        # Reduced segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        # Color value
        x = self._segments[index].start + value
        y = self._segments[index].stop - value
        if x < self._segments[index].stop and y > self._segments[index].start:
            ax2.plot(self.data.iloc[x:y])
        plt.title('Segment after action')

    def concat(self, ind1, ind2, label):

        self.add(min(self._segments[ind1].start, self._segments[ind2].start),
                 max(self._segments[ind1].stop, self._segments[ind2].stop),
                 label)
        self.modify(ind1, self._segments[-1].start, self._segments[-1].stop, label)
        self._segments.pop(ind2)
        self._segments.pop()

    def concat_plot_selection(self, index1, index2):
        """ Creates/Updates the plot output. """

        # Selected segment
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax1.clear()
        ax1.plot(self.data)
        ax1.plot(self.data.iloc[self._segments[index1].start:self._segments[index1].stop])
        ax1.plot(self.data.iloc[self._segments[index2].start:self._segments[index2].stop])
        plt.title('Selected segments')

        # Concatenated segment
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        ax2.clear()
        ax2.plot(self.data)
        x = min(self._segments[index1].start, self._segments[index2].start)
        y = max(self._segments[index1].stop, self._segments[index2].stop)
        ax2.plot(self.data.iloc[x:y])
        plt.title('Resulting segments')


def restore_signals_from_rw(data, overlap_percent):
    """
    Given a matrix with shape (<N> x <window_size> x <signals' number>) recreate a timeseries ndarray with the original
        signals, without the overlap.
    Args:
        data (numpy.ndarray): Matrix with shape (<N> x <window_size> x <signals' number>) that has occured by performing
            rolling window on a collection of signals.
        overlap_percent: The percentage of overlap according to which the rolling window has been performed.

    Returns:
        restored_data (numpy.ndarray): A matrix with shape (<signals' number> X <~original_signals_length>). The
            restored signals' length may not be the exact length of the original, since the last part of a signal
            is discarded if the signal's length is not exactly divisible by the rolling window step.
    """
    window_size = data.shape[1]
    advance_step = window_size - np.ceil(window_size * (overlap_percent / 1e2)).astype(np.int32())
    data_restored = data[0, ...]
    for window_segment in range(1, data.shape[0]):
        data_restored = np.vstack([data_restored, data[window_segment, -advance_step:, :]])
    return data_restored


def get_overlap(a, b, percent=True):
    """
        Function to calculate the overlap between two consecutive segments.
    Args:
        a,b : list in form [start, end] or Segment

    Returns:
        Percentage of overlap (float) if percent = True.
             It is calculated as a percentage of the overall space from a to b.
            Otherwise, the number of overlapping samples (int).

    """
    if isinstance(a, Segment):
        a = [a.start, a.stop]
    if isinstance(b, Segment):
        b = [b.start, b.stop]
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap > 1:
        max_value = b[1]
        min_value = a[0]
        length = len(list(range(min_value, max_value)))
        if percent:
            return overlap * 100 / length
        else:
            return overlap
    else:
        return 0.0


def generate_instance_spectrogram(plot_params, stft_params, data):
    """
    For the given collection of signals, returns the spectrogram of each signal.
    Args:
        plot_params (dict): The parameters for plotting the image (cmap, image height, width, stretching,
            option to save or not)
        stft_params (dict): The parameters for the spectrogram calculation
        data (numpy.array): Array with shape number_of_instances x samples x signals.

    Returns:
        axes_stft_list: numpy array from each axis STFT. It has shape FrequencyBins x stft_time_bins x <Number of signals>
        images_comb_vertical: PIL images from each axis spectrogram stacked vertically
        image_dims : Tupple with the image dimensions
    """
    axes_number = data.shape[1]
    axes_stft_list = []  # List which holds the result of spectrogram
    images_sprecto_list = []  # List which holds PIL.Image objects

    # In case of generating images of the spectrograms of each input instance, the size of the figure can be provided
    # from the configuration.
    image_height = plot_params["final_image_height"]

    # Since the images would be concatenated vertically, each image that is produced by the
    # STFT, should have height equal to final_image_height//axes_number
    if isinstance(plot_params["final_image_height"], int):
        image_height = plot_params["final_image_height"] // axes_number
    else:
        if "auto" in plot_params["final_image_height"]:
            image_height = plot_params["dpi"] * 4.8  # Default matplotlib value in inches
    if "default" in plot_params["image_ratio"]:
        # Default matplotlib fig has size
        # 6.4 / 4.8 inches (ratio 1.33)
        image_width = int(np.ceil(1.33 * image_height))

    elif "stretch" in plot_params["image_ratio"]:
        image_width = plot_params["final_image_height"]
    else:
        if isinstance(plot_params["image_ratio"], float):
            image_width = int(np.ceil(plot_params["image_ratio"] * image_height))
        else:
            raise Exception("Unknown ratio type")
    im = None
    for axis in range(0, axes_number):
        # Calculate STFT
        fig = plt.figure()
        ax = fig.gca()
        spectrogram, f, t, im_plot = plot_spectrogram(data[:, axis], ax=ax, stft_params=stft_params, colorbar=False,
                                                      **plot_params)
        axes_stft_list.append(spectrogram)
        # plt.tight_layout(pad = -0.25)
        fig_export = plot_params["figure_export"]

        if fig_export:
            ax.axis("tight")
            plt.axis("off")

            plt.subplots_adjust(0, 0, 1, 1, 0, 0)

            # Resize figure to have specific size
            fig.set_size_inches(image_width / plot_params["dpi"], image_height / plot_params["dpi"])

            # numpy plot to image conversion
            buffer, im, w, h, d = fig2img(fig, plot_params["figure_export"])
            #         print("Image Value:", im)
            if im is not None:
                # close the plot
                plt.close()
                images_sprecto_list.append(im)
        else:
            images_sprecto_list = None
            # close the plot
            plt.close()

    axes_stft_list = np.array(axes_stft_list)

    instance_spectro_array = np.vstack([i[np.newaxis, :, :] for i in axes_stft_list])
    instance_spectro_array = np.reshape(instance_spectro_array, (instance_spectro_array.shape[1],
                                                                 instance_spectro_array.shape[2],
                                                                 instance_spectro_array.shape[0])
                                        )

    if im is not None:
        images_comb_vertical = merge_pil_images(images_sprecto_list)
        image_dims = (w, h, d)
    else:
        images_comb_vertical = None
        image_dims = instance_spectro_array.shape
    return instance_spectro_array, images_comb_vertical, image_dims


def calculate_sampling_frequency(df):
    """

    :param df:
    :return:
    """
    # get number of samples
    samples = df.shape[0]
    # Calculate epochs in milliseconds
    # epoch_end = pd_data["epoch"].iloc[0]
    epoch_end = datetime.fromtimestamp(df["epoch (ms)"].iloc[samples - 1] / 1e3)
    epoch_start = datetime.fromtimestamp(df["epoch (ms)"].iloc[0] / 1e3)
    # Calculate time difference in seconds
    dt = (epoch_end - epoch_start).total_seconds()
    # Calculate sampling frequency
    try:
        f_s_calculated = samples / dt
    except ZeroDivisionError:
        logging.warning("Too short dataframe. Duration less than seconds to calculate sampling frequency.")
        f_s_calculated = 0.0
    return f_s_calculated


def generate_spectrograms_dataset(config, dataset_X, dataset_y=None, save=True, project_dir=None):
    """
    For a given dataset that contains different signals, produces a new dataset where each instance is a collection
    of spectrograms of the initial dataset.
    Args:
        config:
        dataset_X:
        dataset_y:
        save:
        project_dir:

    Returns:

    """
    spectrogram_dataset = []
    if save and project_dir is None:
        msg = "Saving option is activated but no directory specified."
        logging.error(msg)
        raise Exception(msg)

    stft_params = config["STFT"]
    spectrograms_params = config["spectrograms_fig"]
    path_to_save_figures = None

    if save and spectrograms_params["figure_export"]:
        project_dir = Path(project_dir)
        path_to_save_figures = project_dir.joinpath(spectrograms_params["path_to_save"])
        path_to_save_figures = path_to_save_figures.joinpath(config["uuid"])
        path_to_save_figures.mkdir(parents=True, exist_ok=True)
    # Dictionary used for keeping a count of each figure class label. Used only when exporting images to directory
    count_export_dict = None

    # TODO separate figures extraction for the case that the present function is used in testset.
    if dataset_y is not None:
        num_of_labels = len(np.unique(dataset_y))
        count_export_dict = dict()
        for movementID in range(0, num_of_labels):
            count_export_dict[movementID] = 0

    for ind in tqdm(range(0, dataset_X.shape[0])):
        instance = dataset_X[ind]

        instance_spectro_array, instance_spectro_image, image_dims = generate_instance_spectrogram(spectrograms_params,
                                                                                                   stft_params,
                                                                                                   instance)
        spectrogram_dataset.append(instance_spectro_array[np.newaxis, :])

        if save and instance_spectro_image is not None:
            if dataset_y is not None:
                label_id = dataset_y[ind]
                filename_to_save = "{}_{}.png".format(dataset_y[ind], count_export_dict[label_id])
                count_export_dict[label_id] += 1
            else:
                filename_to_save = "{}.png".format(ind)
            instance_image_save_path = path_to_save_figures.joinpath(filename_to_save)
            instance_spectro_image.save(instance_image_save_path)
    spectrogram_dataset = np.vstack(spectrogram_dataset)
    return spectrogram_dataset


