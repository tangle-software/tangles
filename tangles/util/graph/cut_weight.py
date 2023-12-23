import numpy as np
import scipy.sparse as sparse
from tangles.separations.finding import OrderFuncDerivative, minimize_cut
from tangles.util.graph import laplacian

from typing import Union

class CutWeightOrder(OrderFuncDerivative):
    def __init__(self, adjacency_matrix: Union[np.ndarray, sparse.spmatrix, sparse.csr_array]):
        self._A = adjacency_matrix
        self._L = laplacian(adjacency_matrix)

    def discrete_derivative(self, sep: np.ndarray) -> np.ndarray:
        return self._A.dot(sep) * sep

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        return 0.25 * (feats * (self._L @ feats)).sum(axis=0)

class RatioCutOrder(OrderFuncDerivative):
    def __init__(self, adjacency_matrix: Union[np.ndarray, sparse.spmatrix, sparse.csr_array]):
        self._A = adjacency_matrix
        self._n = adjacency_matrix.sum()
        self._total = adjacency_matrix.shape[0]

    def discrete_derivative(self, sep: np.ndarray) -> np.ndarray:
        change_vector = []
        prev_order = self(sep)
        for i in range(len(sep)):
            sep_swapped = sep.copy()
            sep_swapped[i] = -sep_swapped[i]
            change_vector.append(self(sep_swapped) - prev_order)
        return np.array(change_vector)

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        side_size = (feats>0).sum(axis=0)
        denominator = side_size * (self._total - side_size)
        if isinstance(denominator, np.ndarray):
            denominator[denominator == 0] = 1
        elif denominator == 0:
            denominator = 1
        return 0.25 * (self._n - (feats*(self._A@feats)).sum(axis=0)) * self._total / denominator

class NCutOrder(OrderFuncDerivative):
    def __init__(self, adjacency_matrix: Union[np.ndarray, sparse.spmatrix, sparse.csr_array]):
        self._A = adjacency_matrix
        self._diag = np.array(self._A.sum(axis=0).data)
        self._L = sparse.diags(self._diag) - self._A
        self._total = self._diag.sum()

    def discrete_derivative(self, sep: np.ndarray) -> np.ndarray:
        change_vector = []
        prev_order = self(sep)
        for i in range(len(sep)):
            sep_swapped = sep.copy()
            sep_swapped[i] = -sep_swapped[i]
            change_vector.append(self(sep_swapped) - prev_order)
        return np.array(change_vector)

    def __call__(self, feats: np.ndarray) -> np.ndarray:
        vol = ((feats>0)*self._diag[:,np.newaxis]).sum(axis=0) if len(feats.shape) > 1 else self._diag[feats>0].sum()
        denominator = (vol*(self._total-vol))
        if isinstance(denominator, np.ndarray):
            denominator[denominator == 0] = 1
        elif denominator == 0:
            denominator = 1
        return 0.25 * (feats*((self._L)@feats)).sum(axis=0) * self._total / denominator

def cut_weight_order(adjacency_matrix: sparse.csr_array):
    """Return the cut weight order function for a graph described by the given `adjacency_matrix`.

    The cut weight order calculates the sum of all edge weights across a separation.

    This order function corresponds to (O1) from the tangles book if entries in the adjacency_matrix
    are :math:`sigma(v, w)`.

    Parameters
    ----------
    adjacency_matrix : sparse.csr_array
        The adjacency matrix of the graph.

    Returns
    -------
    OrderFunction
        The cut weight order function.
    """

    return CutWeightOrder(adjacency_matrix)

def ratiocut_order(adjacency_matrix: sparse.csr_array):
    """Return the ratiocut order function for a graph described by the given `adjacency_matrix`.

    The ratiocut order calculates the sum of all edge weights across a separation multiplied
    by the total number of vertices in the graph and divided by the number of pairs of vertices from each side.

    Parameters
    ----------
    adjacency_matrix : sparse.csr_array
        The adjacency matrix of the graph.

    Returns
    -------
    OrderFunction
        The ratiocut order function.
    """

    return RatioCutOrder(adjacency_matrix)

def ncut_order(adjacency_matrix: sparse.csr_array):
    """Return the ncut order function for a graph described by the given `adjacency_matrix`.

    The ncut order calculates the sum of all edge weights across a separation multiplies it
    by the total volume and divides it by the volume of each side.

    The volume is the sum of the degrees of the vertices.

    Parameters
    ----------
    adjacency_matrix : sparse.csr_array
        The adjacency matrix of the graph.

    Returns
    -------
    OrderFunction
        The ncut order function.
    """

    return NCutOrder(adjacency_matrix)

def minimize_cut_weight(starting_feature: np.ndarray, adjacency_matrix: Union[np.ndarray, sparse.spmatrix, sparse.csr_array], max_steps: int = int(1e8)) -> np.ndarray:
    return minimize_cut(starting_feature, cut_weight_order(adjacency_matrix), max_steps=max_steps)

def minimize_ncut_weight(starting_feature: np.ndarray, adjacency_matrix: Union[np.ndarray, sparse.spmatrix, sparse.csr_array], max_steps: int = int(1e8)) -> np.ndarray:
    return minimize_cut(starting_feature, ncut_order(adjacency_matrix), max_steps=max_steps)

def minimize_ratiocut_weight(starting_feature: np.ndarray, adjacency_matrix: Union[np.ndarray, sparse.spmatrix, sparse.csr_array], max_steps: int = int(1e8)) -> np.ndarray:
    return minimize_cut(starting_feature, ratiocut_order(adjacency_matrix), max_steps=max_steps)