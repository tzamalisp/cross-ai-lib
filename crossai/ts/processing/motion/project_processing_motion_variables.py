"""
Default variables regarding processing.
"""
# Define global names for the input signals, and categorize them according to their source sensor.
axes_acc = ["acc_x", "acc_y", "acc_z"]
axes_gyro = ["gyr_x", "gyr_y", "gyr_z"]
# Dictionary to convert axes names to generic names
accepted_keys_to_generic = {
    "x-axis (g)": "acc_x",
    "y-axis (g)": "acc_y",
    "z-axis (g)": "acc_z",
    "x-axis (deg/s)": "gyr_x",
    "y-axis (deg/s)": "gyr_y",
    "z-axis (deg/s)": "gyr_z"
}


def rename_to_common_motion_signals_names(df):
    """
    Renames a pandas dataFrame with motion signal columns to the project defined names.
    Args:
        df (pandas dataFrame):

    Returns:

    """
    rename_dict = dict()
    for col_name in df.columns:
        if col_name in accepted_keys_to_generic.keys():
            rename_dict[col_name] = accepted_keys_to_generic[col_name]
    df = df.rename(columns=rename_dict, inplace=True)
