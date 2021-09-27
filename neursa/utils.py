import torch
import numpy as np
import os

def create_folder(path_to_folder):
    """Creates all folders contained in the path.

    Parameters
    ----------
    path_to_folder : str
        Path to the folder that should be created

    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)


def get_windows(arr, window_size=200, step_size=50):
    """Transforms time series to a matrix of windows

    Parameters
    ----------
    arr : array-like of shape (n, )
        Array-like object.
    window_size : int, optional (default=200)
        Size of int in term of measurements.
    step_size : int, optional (default=50)
        Size of shift of the window.

    Returns
    -------
    np.ndarray of shape (n // step, window_size)
        Slices of the passed `arr`.

    """
    indexes = (np.arange(0, arr.shape[0], step_size).astype(int)[:, None]
               + np.arange(0, window_size, 1).astype(int).reshape(1, -1))
    cutoff = indexes[:, -1] < arr.shape[0]
    ids = indexes[cutoff]
    if type(arr) == torch.Tensor:
        ids = torch.LongTensor(ids)
    windows = arr[ids]

    return windows