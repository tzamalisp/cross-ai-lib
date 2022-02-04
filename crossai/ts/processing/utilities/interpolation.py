from scipy import interpolate
import numpy as np


def interpolate_matrix(matrix, length, axis=0):
    """
    Produces a matrix of a desired size with interpolated values per column
    axis=0 or per row axis=1.
    Args:
        matrix:
        axis (int): The axis that the interpolation will be applied. A column
                    axis=0 / row axis=1.matrix (numpy.ndarray): An np.array
                    that the interpolation will be applied to.
        length (int):  The desired size of the matrix.

    Returns:
        matrix_interp (numpy.ndarray): The matrix with the interpolated values.
    """
    x = np.arange(0, matrix.shape[0])
    fit = interpolate.interp1d(x, matrix, axis=axis, kind="cubic")
    print(length)
    matrix_interp = fit(np.linspace(0, matrix.shape[0] - 1, length))

    return matrix_interp
