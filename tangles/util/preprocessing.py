import numpy as np

def standardize(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Standardize data such that, along the specified axis, the mean is 0 and the standard
    deviation is 1. If along the axis the data is the same, then the data is converted to zeros.

    Parameters
    ----------
    data
        the data to standardize.
    axis
        the axis along which to standardize.

    Returns
    -------
    np.ndarray
        standardized data.
    """
    data_mean_0 = data - np.mean(data, axis=axis, keepdims=True)
    std = np.std(data_mean_0, axis=axis, keepdims=True)
    std[std == 0] = 1
    standardized_data = data_mean_0/ std
    return standardized_data

def balance(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Each data vector, along the specified axis, gets its median subtracted from it.
    This causes half of the values to be positive and half to be negative.

    Parameters
    ----------
    data
        the data to balance.
    axis
        the axis along which to balance.

    Returns
    -------
    np.ndarray
        balanced_data.
    """
    return data - np.median(data, axis=axis, keepdims=True)

def normalize_length(data: np.ndarray, axis: int) -> np.ndarray:
    """
    Normalize the euclidean length of the data along the specified axis.

    Parameters
    ----------
    data
        the data to normalize.
    axis
        the axis along which to normalize.

    Returns
    -------
    np.ndarray
        normalized data.
    """
    return data / np.sqrt(np.sum(np.square(data), axis=axis, keepdims=True))
