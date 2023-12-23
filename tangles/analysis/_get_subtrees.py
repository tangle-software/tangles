import numpy as np

def get_subtrees(tangle_matrix:np.ndarray):
    """
    Consider two subtrees of the TST and return the column index of the separation which separates the two subtrees
    and whether a tangle is in the left subtree, in the right subtree or in neither subtree.

    The two subtrees considered are those rooted at the children of the first node in the TST that distinguishes some
    two tangles.

    Parameters
    ----------
    tangle_matrix : np.ndarray
        A tangle matrix containing distinguished tangles.

    Returns
    -------
    distinguisher_idx : int
        The column index of the distinguishing separation, or -1 if there is no distinguishing separation.
    left_leaf_ids : np.ndarray
        A boolean array of whether a tangle id is contained in the left subtree.
    right_leaf_ids : np.ndarray
        A boolean array of whether a tangle id is contained in the right subtree.
    """

    max_order_tangle = np.argmax(np.sum(np.abs(tangle_matrix), axis=1))
    tangle_matrix_made_equal = tangle_matrix * tangle_matrix[[max_order_tangle]]
    column_equal = np.all(tangle_matrix_made_equal >= 0, axis=0)
    if np.all(column_equal):
        return -1, np.array([False] * tangle_matrix.shape[0]), np.array([False] * tangle_matrix.shape[0])
    distinguisher_idx = np.argmax(column_equal == 0)
    right_leaf_ids = tangle_matrix[:, distinguisher_idx] > 0
    left_leaf_ids = tangle_matrix[:, distinguisher_idx] < 0
    return distinguisher_idx, left_leaf_ids, right_leaf_ids
