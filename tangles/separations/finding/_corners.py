import numpy as np
from tangles.util import faster_uniquerows

def add_all_corners_of_features(feat: np.ndarray) -> np.ndarray:
    """
    Calculates the four corners of every pair of features from an input array. 
    
    Returns an array containing every corner after removing duplicates and inverses as well
    as the original features.

    Parameters
    ----------
    feat : np.ndarray
        The input features, given by columns encoded as -1/1 indicator vectors.

    Returns
    -------
    np.ndarray
        The array of corners.
    """

    all_corners = _add_all_corners(feat)
    all_corners *= all_corners[[0], :]
    return faster_uniquerows(all_corners.T).T

def _add_all_corners(features: np.ndarray) -> np.ndarray:
    """
    Calculates all of the corners of the features. 
    
    Returns a larger array of features which contains
    the original features as well as the corners of the original features.

    Parameters
    ----------
    features : np.ndarray
        The features of which to calculate the corners.

    Returns
    -------
    np.ndarray
        The original features together with the new features.
    """

    return np.c_[features, _add_corner_one_orientation(features, 1, 1),
          _add_corner_one_orientation(features, -1, 1),
          _add_corner_one_orientation(features, 1, -1),
          _add_corner_one_orientation(features, -1, -1)]

def _add_corner_one_orientation(features:np.ndarray, orientation_new:int, orientation_old:int):
    sep_1 = features * orientation_new
    sep_2 = features * orientation_old
    corners = np.ones((features.shape[0], features.shape[1], features.shape[1]), dtype=np.int8)
    corners[(sep_1[:, :, np.newaxis] == -1) * (sep_2[:, np.newaxis, :] == -1)] = -1
    return corners.reshape((sep_1.shape[0], -1))