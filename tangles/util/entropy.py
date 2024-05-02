import numpy as np


def entropy(x: np.ndarray) -> np.ndarray:
    """Compute the discrete entropy of every column of `x`.

    For :math:`x = (x_1,...,x_k)` the result is :math:`(h(x_1),...,h(x_k))`.

    Parameters
    ----------
    x : np.ndarray
        The data.

    Returns
    -------
    np.ndarray
        The entropies of the rows of `x`.
    """

    if len(x.shape) < 2:
        x = x[:, np.newaxis]
    x_sorted = np.sort(x, axis=0)
    uni_indicator = np.ones((x.shape[0] + 1, x.shape[1]), dtype=bool)
    uni_indicator[1:-1, :] = x_sorted[1:, :] != x_sorted[:-1, :]
    ent = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        uni_idcs = np.flatnonzero(uni_indicator[:, i])
        p = (uni_idcs[1:] - uni_idcs[:-1]) / x.shape[0]
        ent[i] = -(p * np.log(p)).sum()
    return ent


def joint_entropy(x: np.ndarray) -> float:
    """Compute the discrete joint entropy of `x`.

    For :math:`x=(x_1, ..., x_k)` the result is :math:`H(x) = H(x_1,x_2,...,x_k)`.

    Parameters
    ----------
    x
        The data.

    Returns
    -------
    float
        The joint entropy of (the rows of) `x`.
    """

    if x.shape[0] == 0 or (len(x.shape) == 2 and x.shape[1] == 0):
        return 0
    x_sorted = x[np.lexsort(x.T), :]
    uni_indicator = np.ones(x.shape[0] + 1, dtype=bool)
    uni_indicator[1:-1] = np.any((x_sorted[1:, :] != x_sorted[:-1, :]), axis=1)
    uni_idcs = np.flatnonzero(uni_indicator)
    p = (uni_idcs[1:] - uni_idcs[:-1]) / x.shape[0]
    return -(p * np.log(p)).sum()


def colsplit_mutual_information(
    data: np.ndarray, partitions: np.ndarray, combined_entropy="joint_entropy"
) -> np.ndarray:
    """Mutual information in the two sides of partitions. The partitions are of the column vectors of a data matrix.

    Parameters
    ----------
    data : np.ndarray
        The data.
    partitions : np.ndarray
        A matrix with partition-indicator-vectors in its columns. Each column represents a partition.
        It has shape :math:`(k, l)`, where :math:`k` is the number of columns in `data` and :math:`l` is the number of partitions.
        We assume that the partition-indicator-vectors split the sides by negative vs. non-positive entries.
    combined_entropy : {'joint_entropy', 'max_entropy'}
        How to calculate the entropy of both sides of the partition together.
        Either 'joint_entropy' (of the entire data) or 'max_entropy' (of both sides).

    Returns
    -------
    np.ndarray
        Orders of the partitions in `partitions`.
    """

    if len(partitions.shape) == 1:
        partitions = partitions[:, np.newaxis]

    o = np.empty(partitions.shape[1])
    combined_entropy = joint_entropy(data)
    for s in range(partitions.shape[1]):
        h_x = joint_entropy(data[:, partitions[:, s] > 0])
        h_y = joint_entropy(data[:, partitions[:, s] <= 0])
        o[s] = (
            h_x + h_y - combined_entropy
            if combined_entropy == "joint_entropy"
            else min(h_x, h_y)
        )
    return o[0] if o.size == 1 else o


def pairwise_mutual_information(data: np.ndarray) -> np.ndarray:
    """Compute a matrix that contains the pairwise mutual information between the columns of `data`.

    Parameters
    ----------
    data : np.ndarray
        The data.

    Returns
    -------
    np.ndarray
        A matrix of shape :math:`(k, k)`, where :math:`k` is the number of columns in `data`.
        The entry at :math:`(i, j)` is the mutual information between columns :math:`i` and :math:`j` of `data`.
    """

    h = entropy(data)
    i_mat = h[:, np.newaxis] + h[np.newaxis, :]
    np.fill_diagonal(i_mat, 0)
    for i in range(data.shape[1] - 1):
        for j in range(i + 1, data.shape[1]):
            i_mat[j, i] -= joint_entropy(data[:, [i, j]])
            i_mat[i, j] = i_mat[j, i]
    return i_mat


def information_gain(data: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Order function based on information gain by adding each feature."""

    e = entropy(feats) + joint_entropy(data)
    for s in range(feats.shape[1]):
        e[s] -= joint_entropy(np.c_[data, feats[:, [s]]])
    return e


def datapointwise_information_gains(data: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """Compute information gains between `feats` and every single column of `data`.

    Parameters
    ----------
    data : np.ndarray
        The data.
    feats : np.ndarray
        A matrix with partition-indicator-vectors in its columns. Each column represents a feature.

    Returns
    -------
    np.ndarray
        A matrix with one column per feature and one row per column of `data`.
    """

    e = entropy(feats)[np.newaxis, :] + entropy(data)[:, np.newaxis]
    for s in range(feats.shape[1]):
        for c in range(data.shape[1]):
            e[c, s] -= joint_entropy(np.c_[data[:, [c]], feats[:, [s]]])
    return e
