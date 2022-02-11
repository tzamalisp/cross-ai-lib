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

