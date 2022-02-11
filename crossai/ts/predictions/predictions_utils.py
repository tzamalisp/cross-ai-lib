import logging
import pandas as pd
from crossai.ts.processing.motion.preprocessing import SegmentsCollection


def windows_to_segments(windows_df, sw_size, op, min_accepted_gesture_len=None):
    """
    This function takes as input a Dataframe and returns a SegmentsCollection
    object that contains the prediction segments that are created by
    consecutive windows with same class name.
    Args:
        windows_df: (df) The dataframe of the window predictions. The shape of
        the Dataframe is (x, >2) where the names of the columns should be Class 
        and model_confidence, representing the window predictions of a model.
        sw_size: (int) The sliding window size
        op: (int) The overlap percentage of the windows. Range 0-100.
        min_accepted_gesture_len (int, optional): The minimum accepted length 
        of a gesture.
    Returns:
        predictions: (SegmentCollection object) The segment collection object 
        with the gestures predictions segments.
    """
    non_op_step = round(sw_size - sw_size * (op / 100))
    print(non_op_step)
    windows_df = calculate_windows_positions(windows_df, non_op_step, sw_size)
    windows_df = windows_df[["model_confidence", "class", "wind_start", "wind_end"]]
    windows_df = windows_df.rename(columns={"class": "Class"})
    windows_df = find_consecutive_windows(windows_df)
    list_approved = []
    for i, row in windows_df.iterrows():
        if row["Class"]:
            if min_accepted_gesture_len < row["Length"]:
                list_approved.append(1)
            else:
                list_approved.append(0)
    windows_df["approved"] = list_approved
    windows_df = windows_df.drop(windows_df[windows_df["approved"] == 0].index)
    windows_df.pop("approved")
    windows_df.pop("Length")
    print(windows_df)
    windows_df.reset_index(inplace=True)
    del windows_df["index"]
    predictions_segments = SegmentsCollection()
    predictions_list = windows_df.values.tolist()
    for pred in predictions_list:
        predictions_segments.add(pred[0], pred[1], pred[2],
                                 prediction_value=pred[3])
    return predictions_segments


def find_consecutive_windows(windows_df):
    """
    This function takes as input a Dataframe with the shape of the extracted
    Dataframe of the function calculate_windows_positions(). It finds
    consecutive windows with the same Class label and organizes them into
    segments:
    Args:
        windows_df: The dataframe of the window predictions. The shape of
        the Dataframe is (x, >2) where the names of the columns should be Class
        and model_confidence, representing the window predictions of a model.
        sw_size (int): The sliding window size.

    Returns:
        A dataframe with the columns:
    "wind_start": The start of a segment.
    "wind_end": The end of a segment.
    "Class": The label of the segment.
    "Confidence": The mean confidence of the windows that created the segment.
    "Length": The length in samples of the segment.
    """
    windows_df["disp"] = (windows_df.Class != windows_df.Class.shift()).cumsum()
    windows_df = pd.DataFrame(
        {"wind_start": windows_df.groupby("disp").wind_start.first(),
         "wind_end": windows_df.groupby("disp").wind_end.last(),
         "Class": windows_df.groupby("disp").Class.first(),
         "Confindence": windows_df.groupby("disp").model_confidence.mean()}).reset_index(
        drop=True)
    windows_df["Length"] = windows_df["wind_end"] - windows_df["wind_start"]
    return windows_df


def calculate_windows_positions(windows_df, non_op_step, sw_size):
    """
    This functions takes as input a dataframe as described in
    windows_to_segments where two columns are added which indicate the start
    and the end of a window in samples.
    Args:
        windows_df (df): The dataframe of the window predictions. The shape of
        the Dataframe is (x, >2) where the names of the columns should be Class
        and model_confidence, representing the window predictions of a model.
        sw_size (int): The sliding window size.
        non_op_step (int): The non overlapping step between two windows.

    Returns:
    A Dataframe with two extra Columns:
    "wind_start": The starting position of a window in samples.
    "wind_end": The ending position of a window in samples.
    """
    starts_list = []
    ends_list = []
    for i in range(0, len(windows_df)):
        window_start = i * non_op_step
        starts_list.append(window_start)
        ends_list.append(window_start + sw_size)
    windows_df.loc[:, "wind_start"] = starts_list
    windows_df.loc[:, "wind_end"] = ends_list
    return windows_df
