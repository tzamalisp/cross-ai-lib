import logging
import pandas as pd
from crossai.ts.processing.preprocessing import SegmentsCollection


def convert_windows_to_gestures(windows_df, rw_size, op, gestures_stats_df=None, min_accepted_gesture_len=None):
    """

    Args:
        windows_df: (df) The dataframe of the window predictions.
        gestures_stats_df: (df) The dataframe of the gestures statistics.
        rw_size: (int) The rolling window size
        op: (int) The overlap percentage of the windows
        min_accepted_gesture_len (int, optional): The minimum accepted length of a gesture. If not specified,
            it is considered the mean length of the corresponding detected gesture.
    Returns:
        predictions: (SegmentCollection object) The segment collection object with the gestures predictions
    """
    non_op_step = round(rw_size - rw_size * (op / 100))
    print(non_op_step)
    windows_df = calculate_windows_positions(windows_df, non_op_step, rw_size)
    windows_df = windows_df[["model_confidence", "class", "wind_start", "wind_end"]]
    windows_df = windows_df.rename(columns={"class": "Class"})
    windows_df = find_consecutive_windows(windows_df)
    list_approved = []
    for i, row in windows_df.iterrows():
        if row["Class"]:
            if gestures_stats_df is not None:
                gest = gestures_stats_df.loc[gestures_stats_df.GestureID == row["Class"]]
                if min_accepted_gesture_len is None:
                    min_accepted_gesture_len = gest["Mean Length (samples)"].values[0]
                if min_accepted_gesture_len < row["Length"]:
                    list_approved.append(1)
                else:
                    logging.warning("No stats file was provided, accepting all predictions,"
                                    " independent of length.")
                    list_approved.append(0)
            else:
                list_approved.append(1)
        else:
            list_approved.append(0)
    windows_df["approved"] = list_approved
    windows_df = windows_df.drop(windows_df[windows_df["approved"] == 0].index)
    # windows_df = find_spaces_between_predictions(windows_df)
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
    windows_df["disp"] = (windows_df.Class != windows_df.Class.shift()).cumsum()
    windows_df = pd.DataFrame({"wind_start": windows_df.groupby("disp").wind_start.first(),
                               "wind_end": windows_df.groupby("disp").wind_end.last(),
                               "Class": windows_df.groupby("disp").Class.first(),
                               "Confindence": windows_df.groupby("disp").model_confidence.mean()}).reset_index(
        drop=True)
    windows_df["Length"] = windows_df["wind_end"] - windows_df["wind_start"]
    # print(windows_df)
    return windows_df


def calculate_windows_positions(windows_df, non_op_step, rw_size):
    """

    Args:
        windows_df:
        rw_size:
        non_op_step:

    Returns:

    """
    starts_list = []
    ends_list = []
    for i in range(0, len(windows_df)):
        window_start = i * non_op_step
        starts_list.append(window_start)
        ends_list.append(window_start + rw_size)
    windows_df.loc[:, "wind_start"] = starts_list
    windows_df.loc[:, "wind_end"] = ends_list
    return windows_df
