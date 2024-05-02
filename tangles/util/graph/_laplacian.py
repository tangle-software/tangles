from typing import Union, Optional
import numpy as np
from scipy import sparse


def laplacian(
    adj: Union[sparse.spmatrix, sparse.sparray, np.ndarray],
) -> Union[sparse.csr_matrix, np.ndarray, None]:
    """
    Compute the combinatorial laplacian :math:`L = D-A`, where :math:`A` is the adjacency matrix of a graph :math:`G`
    and :math:`D` is the diagonal matrix containing the degrees of :math:`G`.

    Parameters
    ----------
    A : sparse array or np.ndarray
        Adjacency matrix.

    Returns
    -------
    scipy.sparse.csr_matrix or np.ndarray
        The laplacian matrix.
    """

    if isinstance(adj, sparse.sparray):
        diag = sparse.dia_array((adj.sum(axis=0), [0]), shape=adj.shape, dtype=float)
        return (diag - adj).tocsr()
    if isinstance(adj, sparse.spmatrix):
        diag = sparse.dia_matrix(
            (adj.sum(axis=0).A1, [0]), shape=adj.shape, dtype=float
        )
        return (diag - adj).tocsr()
    if isinstance(adj, np.ndarray):
        return np.diag(adj.sum(axis=0)) - adj
    print("laplacian(adj):  unkown matrix type")
    return None


def normalized_laplacian(
    adj: Union[sparse.spmatrix, np.ndarray],
) -> Union[sparse.csr_matrix, np.ndarray, None]:
    """
    Compute the normalized laplacian :math:`L' = I - D^{-1/2} A D^{-1/2}`, where :math:`A` is the adjacency matrix of
    a graph :math:`G` and :math:`D` is the diagonal matrix containing the degrees of :math:`G`.

    Parameters
    ----------
    adj : sparse.spmatrix or np.ndarray
        Adjacency matrix.

    Returns
    -------
    scipy.sparse.csr_matrix or np.ndarray
        The normalized laplacian matrix.
    """

    lap = laplacian(adj)
    dia = lap.diagonal()
    diag_1 = np.zeros(lap.shape[0])
    diag_1[dia != 0] = 1.0 / np.sqrt(dia[dia != 0])
    if isinstance(adj, sparse.sparray):
        lap_norm = lap.multiply(diag_1[np.newaxis, :]).multiply(diag_1[:, np.newaxis])
        return lap_norm.tocsr()
    if isinstance(adj, sparse.spmatrix):
        lap_norm = lap.multiply(diag_1[np.newaxis, :]).multiply(diag_1[:, np.newaxis])
        return lap_norm.tocsr()
    if isinstance(adj, np.ndarray):
        lap_norm = lap * diag_1[np.newaxis, :] * diag_1[:, np.newaxis]
        return lap_norm
    print("normalized_laplacian(A):  unkown matrix type")
    return None


def modularity_matrix(
    adj: sparse.csr_matrix, diagonal: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the modularity matrix of the graph with adjacency matrix `A`.

    Parameters
    ----------
    adj : sparse.csr_matrix
        Adjacency matrix.
    diagonal: np.ndarray, optional
        Diagonal.

    Returns
    -------
    np.ndarray
        The modularity matrix. Note: the matrix is not sparse.
    """

    if diagonal is None:
        diagonal = adj.sum(axis=0).A1
    return (
        adj.toarray()
        - diagonal[:, np.newaxis] * diagonal[np.newaxis, :] / diagonal.sum()
    )
