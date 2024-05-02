import os
from typing import Union
import numpy as np

from scipy import sparse
from scipy.spatial import distance_matrix

from tangles._typing import SetSeparationOrderFunction


def matrix_order(
    matrix: Union[np.ndarray, sparse.spmatrix],
    feats: Union[np.ndarray, sparse.spmatrix],
    shift: float = 0,
):
    """
    A general order function defined by a quadratic matrix.

    For a feature indicator vector :math:`f` and a matrix :math:`M` this function computes :math:`|f| = f^T M f`.

    If all entries outside the main diagonal in :math:`M` are smaller or equal to 0 and shift is 0, then the
    submodularity of the order function is guaranteed.

    Parameters
    ----------
    matrix : np.ndarray or sparse.spmatrix
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
        return (feats * (matrix @ feats)).sum(axis=0) + shift * np.square(
            feats.sum(axis=0).astype(float)
        )
    return feats.multiply(matrix @ feats).sum(axis=0) + shift * np.square(
        feats.sum(axis=0).astype(float)
    )


def o_five(laplacian: Union[np.ndarray, sparse.spmatrix]) -> SetSeparationOrderFunction:
    """
    (O5) from the tangles book.
    """
    return lambda feats: matrix_order(laplacian, feats)


def o_seven(j_matrix: Union[np.ndarray, sparse.spmatrix]) -> SetSeparationOrderFunction:
    """
    (O7) from the tangles book.
    """
    return lambda feats: matrix_order(j_matrix, feats)


def covariance_order(
    adjacency_matrix: Union[np.ndarray, sparse.spmatrix],
    feats: np.ndarray,
    shift: float = 0,
):
    """
    Order function defined by the matrix :math:`A^T A`, where :math:`A` is the adjacency matrix of a graph.

    For a feature indicator vector :math:`f` this function computes :math:`|f| = f^T A^T Af = (Af)^T Af`.

    Parameters
    ----------
    adjacency_matrix : np.ndarray or sparse.spmatrix
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

    return np.square(adjacency_matrix @ feats).sum(axis=0) + shift * np.square(
        feats.sum(axis=0).astype(float)
    )


def logdet_order(matrix: np.ndarray, feats: np.ndarray):
    """Order function defined by :math:`|f| = log( det( M_A )) + log( det( M_B ))` where :math:`f` is the bipartition indicator
    vector of the partition :math:`(A,B)`.

    Parameters
    ----------
    matrix : np.ndarray
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
        sel_a = feats[:, i] >= 0
        sel_b = feats[:, i] <= 0
        lap_a = matrix[np.ix_(sel_a, sel_a)]
        lap_b = matrix[np.ix_(sel_b, sel_b)]
        _, l1 = np.linalg.slogdet(lap_a)
        _, l2 = np.linalg.slogdet(lap_b)
        orders[i] = l1 + l2
    return orders


class MatrixOrderSVD:
    r"""Class representing function objects computing approximations of covariance order functions by using singular value decomposition.

    Members
    -------
    a
        A matrix `A` of shape (m, n).
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
    u, s, vt
        Left singular vectors, singular values and right singular vectors of `A` (only the most significant, see `variance_explained`).
    variance_explained
        Fraction of the total variance of the data in `A` (or :math:`A^T` if `mode` is set to 'rows') considered.
        We approximate the value :math:`|f| = f^T A^T Af` (or :math:`|f| = f^T A A^T f`) by considering only the most
        significant principal components of `A` (or :math:`A^T`).
    """

    @staticmethod
    def _check_propack():
        return int(os.getenv("SCIPY_USE_PROPACK", 0)) > 0

    def __init__(
        self, a, mode="rows", variance_explained=0.8, shift=None, svd_args=None
    ):
        if svd_args is None:
            svd_args = {}
        if isinstance(a, sparse.spmatrix):
            if MatrixOrderSVD._check_propack():
                svd_args["k"] = min(a.shape)
                svd_args["solver"] = "propack"
            else:
                svd_args["k"] = min(a.shape) - 1
                svd_args["solver"] = "arpack"
            self.u, self.s, self.vt = sparse.linalg.svds(a, **svd_args)
            sort = np.argsort(self.s)[::-1]
            self.u, self.s, self.vt = self.u[:, sort], self.s[sort], self.vt[sort, :]
        elif isinstance(a, np.ndarray):
            svd_args["full_matrices"] = False
            self.u, self.s, self.vt = np.linalg.svd(a, **svd_args)
        else:
            raise ValueError("unkwon matrix type")

        self.variance_explained = variance_explained
        if variance_explained < 1:
            v = np.square(self.s)
            v = np.cumsum(v) / v.sum()
            n = np.argmax(v >= variance_explained)
            if n > 0:
                self.u, self.s, self.vt = (
                    self.u[:, :n],
                    self.s[:n],
                    self.vt[:n, :],
                )

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
        if self.mode == "rows":
            v = self.u.T @ feats
            o = np.square(v * self.s[:, np.newaxis]).sum(axis=0)
        else:
            v = self.vt @ feats
            o = np.square(v * self.s[:, np.newaxis]).sum(axis=0)
        if self.shift:
            o += self.shift * np.square(feats.sum(axis=0))
        return o


def linear_similarity_from_dist_matrix(
    dist: np.ndarray, margin: float, eps: float = 1e-8, sparse_mat: bool = True
):
    r"""Turns a distance matrix into a similarity matrix by linearly inverting the distances up to some maximum distance (margin).

    A matrix order function (quadratic form) based on the returned matrix behaves like a "margin order function",
    i.e. only pairs of separated points close to a (fictitious optimal) boundary between the separation's sides are taken into account.

    Similarities range from 1, if they are equal, to 0, if they have a distance greater than margin.

    Parameters
    ----------
    m : np.ndarray
        distance matrix of data
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

    dist = dist.copy()
    dist[dist > margin] = margin
    mat = 1 - dist / margin
    mat[mat < eps] = 0
    return sparse.csr_matrix(mat) if sparse_mat else mat


def linear_similarity_from_distances(
    m: np.ndarray,
    margin: float,
    eps: float = 1e-8,
    sparse_mat: bool = True,
    distance_p: float = 2,
):
    r"""Turns a matrix of positions into a matrix of similarities (simply based on distances).

    A matrix order function (quadratic form) based on the returned matrix behaves like a "margin order function",
    i.e. only pairs of separated points close to a (fictitious optimal) boundary between the separation's sides are taken into account.

    Similarities range from 1, if they are equal, to 0, if they have a distance greater than margin.

    Parameters
    ----------
    m : np.ndarray
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

    dist = distance_matrix(m, m, p=distance_p)
    return linear_similarity_from_dist_matrix(dist, margin, eps, sparse_mat)
