import numpy as np

def tangle_score(tangle_matrix:np.ndarray, feat_ids:np.ndarray, feat_sys, normalize_rows=False, normalize_cols=False) -> np.ndarray:
    """
    Calculates the tangle scores, a measure of how much a point is 'contained' within a tangle.

    It is calculated by considering, for every point and every tangle,
    every feature of the tangle and then calculating

        number of features which contain the point - number of feature which do not contain the point.

    Parameters
    ----------
    tangle_matrix : np.ndarray
        A (tangles x features)-matrix encoding whether a tangle contains a certain feature or its inverse.
        Contains 1 if the tangle contains the feature and otherwise 0.
    feat_ids : np.ndarray
        The feature ids corresponding to the columns of the `tangle_matrix`.
    feat_sys : FeatureSystem
        A feature system.

    Returns
    -------
    np.ndarray
        A (points x tangles)-matrix encoding the tangle score of each point in the dataset for each tangle.
    """

    if len(feat_ids) != tangle_matrix.shape[1]:
        raise ValueError("""Number of feature ids must be equal to the number of columns in tangle_matrix. If tangle_matrix
                         and feature ids come directly from the TangleSearchTree, use feat_ids[:tangle_matrix.shape[1]].""")
    scores = feat_sys[feat_ids]@tangle_matrix.astype(float).T if tangle_matrix.dtype != float else feat_sys[feat_ids]@tangle_matrix.T

    if normalize_rows:
        scores -= scores.min(axis=1)[:, np.newaxis]
        score_norm = scores.sum(axis=1)
        scores[score_norm > 0, :] /= score_norm[score_norm > 0][:, np.newaxis]
    if normalize_cols:
        scores -= scores.min(axis=0)[np.newaxis,:]
        score_norm = scores.sum(axis=0)
        scores[:, score_norm > 0] /= score_norm[score_norm > 0][np.newaxis,:]

    return scores



