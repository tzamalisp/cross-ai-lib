import logging
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from crossai.ts.processing.motion.preprocessing import Segment, \
    SegmentsCollection


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


class SlidingWindowHandler:
    def __init__(self, data, overlap, window_size):
        """

        Args:
            data (numpy.ndarray):
            overlap(int):
            window_size(int):
        """

        self.data = data
        # print("data shape : ", data.shape)
        self.length = data.shape[0]
        # assert(overlap in [25,50,75,99])
        self.overlap = overlap
        self.window_size = window_size
        if self.length < self.window_size:
            # # pad data
            # self.data = pad_ndarray(self.data, self.window_size)
            # self.length = self.window_size
            raise Exception("Instance shorter than accepted rolling window size. Cannot continue with segmentation.")
        adv = self.window_size - np.ceil(self.window_size * (self.overlap / 1e2)).astype(np.int32())
        self.segments_array = []
        self.segments_number = 0
        self.segments_ndarray = []
        it = 0  # iterator

        while (it + self.window_size) <= self.length:
            # print(it)
            start = it
            stop = it + self.window_size
            seg = Segment(start, stop)
            seg.data_segment = self.data[start:stop]
            self.segments_ndarray.append(self.data[np.newaxis, start:stop])
            self.segments_array.append(seg)
            self.segments_number += 1
            it = it + adv
        self.segments_ndarray = np.vstack(self.segments_ndarray)
        self.timeseries_len = self.segments_array[
            -1].stop  # the length of the timeseries which corresponds to number of windows

    def get_windows_values_to_array(self):
        """

        :return: a list which contains all segment values in the form [<segment_nd array>, <segment_nd array>, ..]
        where <segment_nd array> is an ndarray with shape 1X<window_sizex>X<input_axes>
        """
        arr = []
        for seg in self.segments_array:
            arr.append(seg.data_segment)
        return arr

    def label_segments(self, labels):
        """
        Adds labels to the segments that have occured after Rolling Window procedure. There are two cases (currently)
        that require the rolling window method and the corresponing labels. First, when creating a dataset, each dataset
        instance that is segmented in rolling window. On this case, the entire data instance has one label and
        each produced segment has the same label.
        The second case occurs when performing rolling window segmentation on an entire session. Then, the labels define
        parts of the overall session. For this case, each rolling window segment is labelled after the label of the
        session part that overlaps most with it. E.g. a rolling window instance that occurs from index 50 to 100
        overlaps with the session instance(a gesture for example) labeled 3 and is defined from 62 to 143. In case that
        a rolling window segment instance does not overlap with any initial session instance, the label remains None.
        Args:
            labels: Either a label integer (preferably) or string that define all the segments that have occured,
                or a SegmentCollection instance.

        Returns:

        """
        if isinstance(labels, SegmentsCollection):

            for seg in self.segments_array:
                labels_index = 0
                labels_found_list = list()
                labels_overlaps_list = list()
                while labels_index < labels._size and labels._segments[labels_index].start <= seg.stop:
                    segments_overlap_samples = get_overlap(seg, labels._segments[labels_index], percent=False)
                    # Define a threshold of accepted overlaped samples. Otherwise, for a few samples, a window
                    # will be mislabeled.
                    # TODO remove hardcoded value
                    labeling_coef = 0.5
                    segments_overlap_labeling_threshold = np.ceil(labeling_coef * self.window_size)
                    if segments_overlap_samples > segments_overlap_labeling_threshold:
                        label_value = labels._segments[labels_index].label
                        if label_value not in labels_found_list:
                            labels_found_list.append(label_value)
                            labels_overlaps_list.append(0)
                        label_index = labels_found_list.index(label_value)
                        labels_overlaps_list[label_index] += segments_overlap_samples

                    labels_index += 1
                # Case where no overlaping segments have been found. The seg label will remain None.
                if labels_found_list:
                    # Find the label with the most overlapping samples with the Segment.
                    most_overlap_samples_label_index = np.argmax(np.array(labels_overlaps_list))
                    seg.label = labels_found_list[most_overlap_samples_label_index]
        else:
            for seg in self.segments_array:
                seg.label = labels

    def get_labels(self):
        """
        Returns all the segments labels as a vector.
        Returns:
            labels (numpy.array)
        """
        labels = list()
        for seg in self.segments_array:
            if seg.label is not None:
                labels.append(seg.label)
            else:
                labels.append(np.nan)
        return np.array(labels)


def apply_sw(accepted_dfs, sw_size, overlap_percent, labels=None):
    """
    Perform rolling window segmentation on a given 2-D dataframe. If provided, the labels
    are also produced for each rolling window segment.
    Args:
        accepted_dfs: (list) A list of pandas dataframes that correspond to the downloaded data.
        sw_size (int): samples
        overlap_percent (int): percentage
        labels: (optional) if given, for each rolling window segment the labels are produced.

    Returns:
        dataset_X: numpy nd array where each row (axis 0) corresponds to a rolling window segment.
        dataset_y (optional): numpy axis with shape 1D which holds the index of the label of each of the dataset
                    segment.
        sw_size: The calculated rolling window size in accordance with the percentage (given from configuration)
                    and the calculated average gestures length from the dataset.
    """

    dataset_X = list()
    dataset_y = list()

    logging.debug("Transformation with Rolling Window and unification of all data to one ndarray.")
    for ind, df in tqdm(enumerate(accepted_dfs)):
        rwh = SlidingWindowHandler(df.values, overlap_percent, sw_size)
        segments_array = rwh.segments_ndarray
        segments_array_length = segments_array.shape[0]
        dataset_X.append(segments_array)
        if labels is not None:
            if isinstance(labels, np.ndarray) or isinstance(labels, list):
                rwh.label_segments(labels[ind])
            else:
                rwh.label_segments(labels)
            dataset_y.append(rwh.get_labels())
    logging.debug("concatenating all segments of dataset_X")
    dataset_X = np.vstack(dataset_X)
    if dataset_y:
        dataset_y = np.hstack(dataset_y)
    logging.info("Dataset created with dimensions {}".format(dataset_X.shape))
    if labels is not None:
        logging.info("Labels created with dimensions {}".format(dataset_y.shape))
    return dataset_X, dataset_y
