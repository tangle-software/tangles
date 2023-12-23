import numpy as np

def entropy(X: np.ndarray) -> np.ndarray:
    """Compute the discrete entropy of every column of `X`.

    For :math:`X = (X_1,...,X_k)` the result is :math:`(h(X_1),...,h(X_k))`.

    Parameters
    ----------
    X : np.ndarray
        The data.

    Returns
    -------
    np.ndarray
        The entropies of the rows of `X`.
    """

    if len(X.shape) < 2:
        X = X[:,np.newaxis]
    X_sorted = np.sort(X, axis=0)
    uni_indicator = np.ones((X.shape[0]+1,X.shape[1]), dtype=bool)
    uni_indicator[1:-1,:] = X_sorted[1:,:] != X_sorted[:-1,:]
    ent = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        uni_idcs = np.flatnonzero(uni_indicator[:,i])
        p = (uni_idcs[1:]-uni_idcs[:-1])/X.shape[0]
        ent[i] = -(p*np.log(p)).sum()
    return ent

def joint_entropy(X:np.ndarray) -> float:
    """Compute the discrete joint entropy of `X`.

    For :math:`X=(X_1, ..., X_k)` the result is :math:`H(X) = H(X_1,X_2,...,X_k)`.

    Parameters
    ----------
    X
        The data.

    Returns
    -------
    float
        The joint entropy of (the rows of) `X`.
    """

    if X.shape[0] == 0 or (len(X.shape) == 2 and X.shape[1] == 0):
        return 0
    X_sorted = X[np.lexsort(X.T), :]
    uni_indicator = np.ones(X.shape[0] + 1, dtype=bool)
    uni_indicator[1:-1] = np.any((X_sorted[1:, :] != X_sorted[:-1, :]),axis=1)
    uni_idcs = np.flatnonzero(uni_indicator)
    p = (uni_idcs[1:]-uni_idcs[:-1])/X.shape[0]
    return -(p*np.log(p)).sum()

def colsplit_mutual_information(data: np.ndarray, partitions: np.ndarray, combined_entropy='joint_entropy') -> np.ndarray:
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
        partitions = partitions[:,np.newaxis]

    o = np.empty(partitions.shape[1])
    combined_entropy = joint_entropy(data)
    for s in range(partitions.shape[1]):
        h_X = joint_entropy(data[:,partitions[:,s]>0])
        h_Y = joint_entropy(data[:,partitions[:,s]<=0])
        o[s] = h_X + h_Y - combined_entropy if combined_entropy == 'joint_entropy' else min(h_X,h_Y)
    return o[0] if o.size==1 else o

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
    I_mat = h[:, np.newaxis] + h[np.newaxis, :]
    np.fill_diagonal(I_mat, 0)
    for i in range(data.shape[1] - 1):
        for j in range(i + 1, data.shape[1]):
            I_mat[j, i] -= joint_entropy(data[:, [i,j]])
            I_mat[i, j] = I_mat[j, i]
    return I_mat

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
            e[c,s] -= joint_entropy(np.c_[data[:,[c]], feats[:,[s]]])
    return e

def information_gain_2(q, p_1, p_2): #todo description and better name
    p = q * p_1 + (1-q)* p_2
    return q * p_1 * np.log(p_1) + q* (1-p_1) * np.log(1-p_1) + (1-q)* p_2 * np.log(p_2) + (1-q) * (1-p_2) * np.log(1-p_2) - p * np.log(p) - (1-p) * np.log(1-p)