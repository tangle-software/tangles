from typing import Optional
import numpy as np
from tangles.analysis._get_subtrees import get_subtrees

def visibility(tangle_matrix:np.ndarray, column_orders:np.ndarray, coherence:Optional[np.ndarray]=None, complexity:Optional[np.ndarray] = None):
    """
    Calculate the visibility of the tangles in the tangle matrix. 
    
    The visibility is the difference between the order at which a tangle ceases to exist (its coherence) and the order
    at which it can first be distinguished from all other tangles it is not contained in (its complexity).

    Parameters
    ----------
    tangle_matrix : np.ndarray
        A (tangles x separations)-matrix encoding how a tangle orients a separation. 
        Contains the orientation of the separation if the tangle orients the separation and otherwise 0.
    column_orders : np.ndarray
        The orders of the separations in the columns of the `tangle_matrix`.
    coherence : np.ndarray, optional
        A parameter for precomputed coherence levels, which are the
        coherence of a tangle represented as the column index in the `tangle_matrix`.
    complexity : np.ndarray, optional
        A parameter for precomputed complexity levels, which are the
        complexity of a tangle represented as the column index in the `tangle_matrix`.

    Returns
    -------
    np.ndarray
        The visibilities of the tangles.
    """

    if coherence is None:
        coherence_lvl = coherence_levels(tangle_matrix)
    if complexity is None:
        complexity_lvl = complexity_levels(tangle_matrix)
    coherence = column_orders[coherence_lvl]
    complexity = column_orders[complexity_lvl]
    return coherence-complexity

def coherence_levels(tangle_matrix:np.ndarray):
    """
    Calculate the coherence levels of tangles. 
    
    The coherence level of a tangle, given a tangle matrix, 
    is the smallest index of a column where it has a zero. 
    This corresponds to the level at which the tangles ceases to exist for the first time.
    In case a tangle does not have such a zero its coherence level is -1 instead.

    Parameters
    ----------
    tangle_matrix : np.ndarray
        A (tangles x separations)-matrix encoding how a tangle orients a separation. 
        Contains the orientation of the separation if the tangle orients the separation and otherwise 0.

    Returns
    -------
    np.ndarray
        The coherence levels of the tangles.
    """

    max_index = np.argmax(tangle_matrix == 0, axis=1)
    max_index[np.all(tangle_matrix != 0, axis=1)] = -1
    return max_index


def complexity_levels(tangle_matrix:np.ndarray):
    """
    Calculates the complexity levels of tangles. 
    
    The complexity level of a tangle, given a tangle_matrix, is the smallest index such that if the tangle matrix were
    sliced until this index the tangle would be distinguishable from every other tangle it is not contained in.
    This corresponds to the level at which the tangle starts to exist for the first time.

    Parameters
    ----------
    tangle_matrix : np.ndarray
        A (tangles x separations)-matrix encoding how a tangle orients a separation. 
        Contains the orientation of the separation if the tangle orients the separation and otherwise 0.

    Returns
    -------
    np.ndarray
        The complexity levels of the tangles.
    """

    result = np.zeros(tangle_matrix.shape[0], dtype=int)
    if tangle_matrix.shape[0] == 1:
        return result
    distinguisher_idx, left_ids, right_ids = get_subtrees(tangle_matrix)
    if np.sum(left_ids) + np.sum(right_ids) > 1:
        result[left_ids] = complexity_levels(tangle_matrix[left_ids][:, distinguisher_idx:]) + distinguisher_idx
        result[right_ids] = complexity_levels(tangle_matrix[right_ids][:, distinguisher_idx:]) + distinguisher_idx
    return result
