import numpy as np
import scipy.sparse as sparse
from typing import Union

def laplacian(A: Union[sparse.spmatrix, sparse._arrays._sparray, np.ndarray]) -> Union[sparse.csr_matrix,np.ndarray,None]:
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

    if isinstance(A, sparse._arrays._sparray):
        D = sparse.dia_array((A.sum(axis=0), [0]), shape=A.shape, dtype=float)
        return (D - A).tocsr()
    elif isinstance(A, sparse.spmatrix):
        D = sparse.dia_matrix((A.sum(axis=0).A1,[0]), shape=A.shape, dtype=float)
        return (D-A).tocsr()
    elif isinstance(A, np.ndarray):
        return np.diag(A.sum(axis=0)) - A
    else:
        print("laplacian(A):  unkown matrix type")
        return None

def normalized_laplacian(A: Union[sparse.spmatrix, np.ndarray]) -> Union[sparse.csr_matrix,np.ndarray,None]:
    """
    Compute the normalized laplacian :math:`L' = I - D^{-1/2} A D^{-1/2}`, where :math:`A` is the adjacency matrix of
    a graph :math:`G` and :math:`D` is the diagonal matrix containing the degrees of :math:`G`.

    Parameters
    ----------
    A : sparse.spmatrix or np.ndarray
        Adjacency matrix.

    Returns
    -------
    scipy.sparse.csr_matrix or np.ndarray
        The normalized laplacian matrix.
    """

    L = laplacian(A)
    dia = L.diagonal()
    diag_1 = np.zeros(L.shape[0])
    diag_1[dia != 0] = 1.0 / np.sqrt(dia[dia != 0])
    if isinstance(A, sparse._arrays._sparray):
        Lnorm = L.multiply(diag_1[np.newaxis, :]).multiply(diag_1[:, np.newaxis])
        return Lnorm.tocsr()
    elif isinstance(A, sparse.spmatrix):
        Lnorm = L.multiply(diag_1[np.newaxis, :]).multiply(diag_1[:, np.newaxis])
        return Lnorm.tocsr()
    elif isinstance(A, np.ndarray):
        Lnorm = L*diag_1[np.newaxis, :]*diag_1[:, np.newaxis]
        return Lnorm
    else:
        print("normalized_laplacian(A):  unkown matrix type")
        return None

def modularity_matrix(A: sparse.csr_matrix, D: np.ndarray = None) -> np.ndarray:
    """
    Compute the modularity matrix of the graph with adjacency matrix `A`.

    Parameters
    ----------
    A : sparse.csr_matrix
        Adjacency matrix.

    Returns
    -------
    np.ndarray
        The modularity matrix. Note: the matrix is not sparse.
    """

    if D is None:
        D = A.sum(axis=0).A1
    return A.toarray() - D[:,np.newaxis]*D[np.newaxis,:]/D.sum()
