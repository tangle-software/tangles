import numpy as np
from ._util import _threshold_partitions
from typing import Optional, Union

def pca_features(M: np.ndarray, k: Optional[int] = None, use_J: bool = False) -> Union[np.ndarray , tuple[np.ndarray, np.ndarray]]:
    """
    Generate features using a method inspired by Principal Component Analysis (PCA).

    In principal component analysis we identify principal components:
    orthogonal vectors which describe directions of high covariance in the data set.

    If we interpret these orthogonal vectors as a coordinate system, we can assign to
    every data-point a score for each of the orthogonal values, namely what the coordinate
    of the data-point is with regards to this axis. The feature for a particular component
    then is the set of points which have a positive score with regard to that vector.

    Parameters
    ----------
    M : np.ndarray
        A matrix of shape (:math:`n`, :math:`p`), where :math:`n` is the number of measurements, and :math:`p` is the number
        of dimensions of each measurement.
    k : int, optional
        The number of eigenvectors to return. The eigenvectors with the lowest value
        (i.e. the greatest magnitude) get returned first.
        Defaults to None in which case every eigenvector gets returned.
        Accepts negative values, where :math:`-k` leads to the exclusion of the last :math:`k` eigenvectors.
    use_J : bool
        There is a shortcut for calculating these features by directly calculating the
        eigenvalues of :math:`J = -MM^T`. This makes sense if the dimension :math:`p` is large
        or the number of data points :math:`n` is small.
        Defaults to False.

    Returns
    -------
    np.ndarray of dtype int
        The PCA features.
    """
    I = -M.T @ M if not use_J else -M @ M.T
    _, U = np.linalg.eigh(I)
    if k is not None:
        k = min(U.shape[0], k)
        bips = _threshold_partitions(U[:,:k])
    bips *= bips[0:1,:]
    return M @ bips if not use_J else bips
