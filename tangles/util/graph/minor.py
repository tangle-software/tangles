import numpy as np
import scipy.sparse as sparse

def contract_graph(A: np.ndarray, bag_indicator: sparse.csc_matrix) -> np.ndarray:
    """ Calculate the minor of a graph given a matrix of bag indicators.

    Parameters
    ----------
    A : np.ndarray
        The adjacency matrix to find the minor of.
    bag_indicator : sparse.csc_matrix
        A (number of vertices in the graph, number of bags) sparse.csc_matrix indicating what vertices
        are contained in which bag.

    Returns
    -------
    np.ndarray
        The minor.
    """

    bag_num = bag_indicator.shape[1]
    A_reduced = np.zeros((bag_num, bag_num))

    for bag_index in range(bag_indicator.shape[1]):
        indices_in_bag = bag_indicator.indices[bag_indicator.indptr[bag_index]:bag_indicator.indptr[bag_index+1]]
        weight = A[indices_in_bag,:].sum(axis=0).T
        weight[indices_in_bag,0] = 0
        A_reduced[bag_index,:] = np.squeeze(bag_indicator @ weight)
    return A_reduced