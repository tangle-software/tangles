"""
Calculate and analyse guiding sets.

A guiding set of a tangle is a set such that, for every partition of the tangle, more
than half of the elements of the guiding set lie in the side of the partition contained
in the tangle.
"""

import numpy as np
from tangles.search import TangleSearchTree
from tangles import Tangle

def guided_tangle(tree: TangleSearchTree, feat_sys, subset: np.ndarray, min_agreement: int = 0) -> Tangle:
    """
    For a given subset, find the maximal guided tangle.

    Parameters
    ----------
    tree : TangleSearchTree
        The tangle search tree in which to look for the guided tangle.
    feat_sys
        The feature system.
    subset : np.ndarray
        The subset to guide the tangle.
    min_agreement : int
        Return a tangle of at least this agreement value.

    Returns
    -------
    Tangle
        The maximal tangle guided by the subset taken from the set of tangles of the tangle search tree
        which have agreement at least `min_agreement`.
    """

    subset_seps = feat_sys[tree.sep_ids][subset]
    guided_path = np.sign(np.sum(subset_seps, axis=0))
    potential_zero_index = np.argmax(guided_path == 0)
    if guided_path[potential_zero_index] == 0:
        guided_path[potential_zero_index:] = 0
    return get_tangle_by_path(tree, guided_path, min_agreement)

def get_tangle_by_path(tree: TangleSearchTree, path: np.ndarray, min_agreement: int = 0) -> Tangle:
    """
    Method for finding a tangle in the tangle search tree by path. The path contains -1/1
    values indicating whether to take the left or right child. The value 0 means that the path stops.

    If the tree does not contain the tangle specified by the path, then the tangle of the longest
    subpath which is contained in the tree is returned.

    Parameters
    ----------
    tree : TangleSearchTree
        The tangle search tree in which we look for the tangle.
    path : np.ndarray
        The path of which to find the tangle.
    min_agreement : int, optional
        Only the tangles of at least this agreement value are considered.

    Returns
    -------
    Tangle
        The specified tangle.
    """

    node = tree.root
    for direction in path:
        if direction == 0:
            return node
        candidate = node.left_child if direction == -1 else node.right_child
        if not candidate or candidate.agreement < min_agreement:
            return node
        node = candidate
    return node

def is_guiding(subset: np.ndarray, tangle: Tangle, feature_system) -> bool:
    """
    Whether the specified subset is a guiding subset for the given tangle.

    Parameters
    ----------
    subset : np.ndarray
        The subset to check for whether it guides. The subset is encoded by boolean values.
    tangle : Tangle
        The tangle to check.
    feature_system
        The feature system.

    Returns
    -------
    bool
        Whether the subset guides the tangle.
    """
    
    core_ids = np.array([sep_id for sep_id, _ in tangle.core])
    core_ori = np.array([ori for _, ori in tangle.core])
    features_of_tangle = feature_system[core_ids][subset] * core_ori[np.newaxis, :]
    return np.all(np.sum(features_of_tangle, axis=0) > 0)
