import numpy as np

def _threshold_partitions(U: np.ndarray, threshold: float = 0, small_side_le:bool = True) -> np.ndarray:
    """
    Turn real valued vectors into partition indicator vectors by assigning indices to sides according to a threshold

    Parameters
    ----------
    U : np.ndarray
        matrix containing the vectors

    threshold : float
        the threshold: entries of the vectors smaller will be on the small side of the oriented bipartition

    small_side_le : bool
        if True, zero entries are on small side of the bipartition

    Returns
    -------

    np.array
        resulting partitions

    """
    S = -np.ones(U.shape, dtype=np.int8)
    if small_side_le:
        S[U > threshold] = 1
    else:
        S[U >= threshold] = 1
    return S