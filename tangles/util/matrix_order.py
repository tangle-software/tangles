import numpy as np

import scipy as sp
import scipy.sparse as sparse
from scipy.spatial import distance_matrix
import os

from tangles._typing import SetSeparationOrderFunction
from typing import Union

def matrix_order(M: Union[np.ndarray, sparse.spmatrix], feats: Union[np.ndarray, sparse.spmatrix], shift:float=0):
    """
    A general order function defined by a quadratic matrix.

    For a feature indicator vector :math:`f` this function computes :math:`|f| = f^T M f`.

    If all entries outside the main diagonal in `M` are smaller or equal to 0 and shift is 0, then the
    submodularity of the order function is guaranteed.

    Parameters
    ----------
    M : np.ndarray or sparse.spmatrix
        A quadratic matrix of shape (``seps.shape[0]``, ``seps.shape[0]``).
    feats : np.ndarray or sparse.spmatrix
        Matrix containing indicator vectors of features as columns.
    shift : float
        This parameter changes the order function's preference for balanced features
        (note: it might also have an effect on its sub- or supermodularity).
        Let :math:`c` denote the value of `shift`. The order function computes the order by :math:`|f| = f^T (M + cJ) f`.

    Returns
    -------
    orders : np.ndarray
        1-dimensional np.ndarray of length ``seps.shape[1]`` containing the orders.

    """

    if isinstance(feats, np.ndarray):
        return (feats * (M @ feats)).sum(axis=0) + shift * np.square(feats.sum(axis=0).astype(float))
    else:
        return feats.multiply(M @ feats).sum(axis=0) + shift * np.square(feats.sum(axis=0).astype(float))


def o_five(laplacian: Union[np.ndarray, sparse.spmatrix]) -> SetSeparationOrderFunction:
    """
    (O5) from the tangles book.
    """
    return lambda feats: matrix_order(laplacian, feats)


def o_seven(J: Union[np.ndarray, sparse.spmatrix]) -> SetSeparationOrderFunction:
    """
    (O7) from the tangles book.
    """
    return lambda feats: matrix_order(J, feats)


def covariance_order(A: Union[np.ndarray, sparse.spmatrix], feats: np.ndarray, shift:float=0):
    """Order function defined by the matrix :math:`A^T A`.

    For a feature indicator vector :math:`f` this function computes :math:`|f| = f^T A^T Af = (Af)^T Af`.

    Parameters
    ----------
    A : np.ndarray or sparse.spmatrix
        A matrix with ``seps.shape[0]`` columns.
    feats : np.ndarray
        Matrix containing indicator vectors of features as columns.
    shift : float
        This parameter changes the order function's preference for balanced features
        (note: it might also have an effect on its sub- or supermodularity).
        Let :math:`c` denote the value of `shift`. The order function computes the order by :math:`|f| = f^T (M + cJ) f`.

    Returns
    -------
    orders : np.ndarray
        1-dimensional np.ndarray of length ``seps.shape[1]`` containing the orders.
    """

    return np.square(A@feats).sum(axis=0) + shift * np.square(feats.sum(axis=0).astype(float))


def logdet_order(M:np.ndarray, feats:np.ndarray):
    """Order function defined by :math:`|f| = log( det( M_A )) + log( det( M_B ))` where :math:`f` is the bipartition indicator
    vector of the partition :math:`(A,B)`.

    Parameters
    ----------
    M : np.ndarray
        Square matrix of shape (``feats.shape[0]``, ``feats.shape[1]``).
    feats : np.ndarray
        Matrix containing partition indicator vectors as columns.

    Returns
    -------
    np.ndarray
        1-dimensional np.ndarray of length ``feats.shape[1]`` containing the orders.
    """

    orders = np.empty(feats.shape[1])
    for i in range(feats.shape[1]):
        sel_A = feats[:,i]>=0
        sel_B = feats[:,i]<=0
        L_A = M[np.ix_(sel_A,sel_A)]
        L_B = M[np.ix_(sel_B, sel_B)]
        _,l1 = np.linalg.slogdet(L_A)
        _,l2 = np.linalg.slogdet(L_B)
        orders[i] = l1+l2
    return orders



class MatrixOrderSVD:
    """Class representing function objects computing approximations of covariance order functions by using singular value decomposition.

    Members
    -------
    A
        A matrix of shape (m, n).
    mode
        Either 'rows' or 'cols'.
        The value 'rows' means, that the separations we want to evaluate are separations of the set of rows of `A`,
        i.e. for a separation indicator vector :math:`f` in :math:`\{-1,1\}^m` we compute :math:`|f| = f^T A A^T f`.
        The value 'cols' means, that the separations we want to evaluate are separations of the set of columns of `A`,
        i.e. for a separation indicator vector :math:`f` in :math:`{-1,1}^n` we compute :math:`|f| = f^T A^T Af`.
    shift
        This parameter changes the order function's preference for balanced separations
        (note: it might also have an effect on its sub- or supermodularity).
        Let :math:`c` denote the value of `shift`. The order function computes the order by :math:`|f| = f^T (M + cJ) f`.
    U, s, Vt
        Left singular vectors, singular values and right singular vectors of `A` (only the most significant, see `variance_explained`).
    variance_explained
        Fraction of the total variance of the data in `A` (or :math:`A^T` if `mode` is set to 'rows') considered.
        We approximate the value :math:`|f| = f^T A^T Af` (or :math:`|f| = f^T A A^T f`) by considering only the most
        significant principal components of `A` (or :math:`A^T`).
    """

    @staticmethod
    def _check_propack():
        return int(os.getenv('SCIPY_USE_PROPACK', 0)) > 0

    def __init__(self, A, mode='rows', variance_explained=0.8, shift = None, svd_args=None):
        if svd_args is None:
            svd_args = {}
        if isinstance(A, sparse.spmatrix):
            if MatrixOrderSVD._check_propack():
                svd_args['k'] = min(A.shape)
                svd_args['solver'] = 'propack'
            else:
                svd_args['k'] = min(A.shape)-1
                svd_args['solver'] = 'arpack'
            self.U, self.s, self.Vt = sparse.linalg.svds(A, **svd_args)
            sort = np.argsort(self.s)[::-1]
            self.U, self.s, self.Vt = self.U[:,sort], self.s[sort], self.Vt[sort,:]
        elif isinstance(A, np.ndarray):
            svd_args['full_matrices'] = False
            self.U, self.s, self.Vt = np.linalg.svd(A, **svd_args)
        else:
            raise ValueError('unkwon matrix type')

        self.variance_explained = variance_explained
        if variance_explained < 1:
            v = np.square(self.s)
            v = np.cumsum(v) / v.sum()
            n = np.argmax(v >= variance_explained)
            if n > 0:
                self.U, self.s, self.Vt = self.U[:,:n], self.s[:n], self.Vt[:n,:],

        self.mode = mode
        self.shift = shift


    def __call__(self, feats):
        """Compute an approximation of :math:`f^T A^T Af` or  :math:`f^T AA^T f` (depending on `mode`=='rows' or
        `mode`=='cols') for each :math:`f` in `seps`.

        Parameters
        ----------
        feats
            Matrix containing indicator vectors of features as columns.

        Returns
        -------
        np.ndarray
            1-dimensional np.ndarray of length ``seps.shape[1]`` containing the orders.
        """
        if len(feats.shape) == 1:
            feats = feats[:, np.newaxis]
        if self.mode == 'rows':
            v = self.U.T @ feats
            o = np.square(v * self.s[:, np.newaxis]).sum(axis=0)
        else:
            v = self.Vt @ feats
            o = np.square(v * self.s[:, np.newaxis]).sum(axis=0)
        if self.shift:
            o += self.shift * np.square(feats.sum(axis=0))
        return o


def margin_order_matrix(M: np.ndarray, margin:float, eps:float=1e-8, sparse_mat: bool = True, distance_p: float = 2):
    """Turns a matrix of positions into a matrix of similarities (simply based on distances).

    A matrix order function (quadratic form) based on the returned matrix behaves like a "margin order function",
    i.e. only pairs of separated points close to a (fictitious optimal) boundary between the separation's sides are taken into account.

    Similarities range from 1, if they are equal, to 0, if they have a distance greater than margin.

    Parameters
    ----------
    M : np.ndarray
        Matrix of positions.
    margin : float
        Size of the margin. In the distance matrix, all distances greater than this margin will be set to the margin.
    eps : float
        A threshold. Similarities below this value are set to 0.
    sparse_mat : bool
        Whether to return a sparse.csr_matrix.
    distance_p : float
        Which Minkowski p-norm to use. Must take a value in :math:`[1, \infty)`.

    Returns
    -------
    sparse.csr_matrix or np.ndarray
        A matrix of similarities.
    """

    dist = distance_matrix(M, M, p=distance_p)
    dist[dist > margin] = margin
    mat = 1 - dist / margin
    mat[mat < eps] = 0
    return sparse.csr_matrix(mat) if sparse_mat else mat
