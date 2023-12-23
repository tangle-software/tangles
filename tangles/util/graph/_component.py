from typing import Union
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csg

def connected_component_indicators(A: Union[sparse.spmatrix, np.ndarray]) -> np.ndarray:
    """
    Compute a matrix containing indicator vectors of the connected components of a graph as columns.

    Parameters
    ----------
    A : sparse.spmatrix or np.ndarray
        Adjacency matrix of a graph.

    Returns
    -------
    np.array
        Resulting component indicators (bool-matrix containing the indicator vectors as columns).
    """

    cc, comp_labels = csg.connected_components(A, directed=False, return_labels=True)
    return np.array([(comp_labels == l) for l in range(cc)], dtype=bool).T