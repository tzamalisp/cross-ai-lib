import numpy as np


def restore_signals_from_rw(data, overlap_percent):
    """
    Given a matrix with shape (<N> x <window_size> x <signals' number>)
    recreate a timeseries ndarray with the original signals, without the overlap.
    Args:
        data (numpy.ndarray): Matrix with shape (<N> x <window_size> x
        <signals' number>) that has occured by performing rolling window on a
        collection of signals.
        overlap_percent: The percentage of overlap according to which the
        rolling window has been performed.

    Returns:
        restored_data (numpy.ndarray): A matrix with shape (<signals' number> X
         <~original_signals_length>). The restored signals' length may not be
         the exact length of the original, since the last part of a signal
         is discarded if the signal's length is not exactly divisible by the
         sliding window step.
    """
    window_size = data.shape[1]
    advance_step = window_size - np.ceil(window_size * (overlap_percent / 1e2)).astype(np.int32())
    data_restored = data[0, ...]
    for window_segment in range(1, data.shape[0]):
        data_restored = np.vstack([data_restored, data[window_segment, -advance_step:, :]])
    return data_restored



