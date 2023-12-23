import numpy as np
from tangles._typing import Callable
from tangles.analysis._get_subtrees import get_subtrees

def hard_clustering_tuebingen(tangle_matrix:np.ndarray, feat_ids:np.ndarray, feat_sys) -> np.ndarray:
    """
    Implements hard clustering as described in the Tuebingen Paper.

    For each tangle, a corresponding cluster is calculated.

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
        A (points x tangles)-matrix with values 0 and 1.
        For each point, it encodes the membership of the point to the cluster that corresponds to the tangle.
    """

    def prob_calculator(_, distinguisher_idx:int, feat_ids:np.ndarray) -> np.ndarray:
        return ((feat_sys[feat_ids[distinguisher_idx]]+1)/2).astype(int)[np.newaxis, :]

    return _clustering_tuebingen(tangle_matrix, feat_ids, feat_sys.datasize, prob_calculator).T

def soft_clustering_tuebingen(tangle_matrix:np.ndarray, feat_ids:np.ndarray, feat_sys) -> np.ndarray:
    """
    Implements soft clustering as described in the Tuebingen Paper.

    For each tangle, a corresponding cluster is calculated.

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
    clustering_matrix : np.ndarray
        A (point x tangles)-matrix encoding the clustering score of each point in the dataset for each tangle.
        The value represents the proportion of the point contained within the cluster.
    """

    def prob_calculator(tangle_matrix: np.ndarray, distinguisher_idx:int, feat_ids:np.ndarray) -> np.ndarray:
        tangle_matrix_for_split = tangle_matrix * tangle_matrix[:, [distinguisher_idx]]
        pos_ids = np.all(tangle_matrix_for_split==1, axis=0)
        neg_ids = np.all(-tangle_matrix_for_split==1, axis=0)
        probability_split = (((feat_sys[feat_ids[pos_ids]].astype(float)+1)/2).sum(axis=1) + ((-feat_sys[feat_ids[neg_ids]].astype(float)+1)/2).sum(axis=1)) / (np.sum(pos_ids) + np.sum(neg_ids))
        return probability_split

    return _clustering_tuebingen(tangle_matrix, feat_ids, feat_sys.datasize, prob_calculator).T

def _clustering_tuebingen(tangles:np.ndarray, feat_ids:np.ndarray, num_datapoints:int, prob_calculator:Callable) -> np.ndarray:
    if tangles.shape[0] <= 1:
        return np.ones((1, num_datapoints))
    distinguisher_idx, left, right = get_subtrees(tangles)
    prob_vector = prob_calculator(tangles, distinguisher_idx, feat_ids)
    left_clustering_matrix = _clustering_tuebingen(tangles[left][:, distinguisher_idx:], feat_ids[distinguisher_idx:], num_datapoints, prob_calculator) * (1-prob_vector)
    right_clustering_matrix = _clustering_tuebingen(tangles[right][:, distinguisher_idx:], feat_ids[distinguisher_idx:], num_datapoints, prob_calculator) * (prob_vector)
    result = np.ones((tangles.shape[0], left_clustering_matrix.shape[1]))
    result[left] = left_clustering_matrix
    result[right] = right_clustering_matrix
    return result
