import numpy as np


def simplify(matrix: np.ndarray) -> np.ndarray:
    """
    Simplifies a matrix in either CNF or DNF.

    Which of these two normal forms the matrix is in does not matter. The result
    is in the same normal form as the input matrix.
    """

    do_while = True
    while do_while:
        matrix, do_while = _filter_supersets(np.unique(matrix, axis=1))
        do_while += _single_difference_in_place(matrix) + _cancel_unique_in_place(
            matrix
        )
    return matrix


def _filter_supersets(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    We simplify either a CNF or DNF by removing redundant clauses.
    Let X and Y be terms in either CNF or DNF. The simplification works as follows.
    If X and Y are in CNF
        (X or Y) and X => X.
    If X and Y are in DNF
        (X and Y) or X => X.
    """

    difference = np.einsum("ik,ij->ijk", matrix, matrix) - np.abs(
        matrix[:, np.newaxis, :]
    )
    difference[0, tuple(range(matrix.shape[1])), tuple(range(matrix.shape[1]))] = 1
    super_indices = list(set(np.nonzero(np.all(difference == 0, axis=0))[0]))
    my_slice = np.ones(matrix.shape[1], dtype=bool)
    my_slice[super_indices] = np.all(matrix[:, super_indices] == 0, axis=0)
    return matrix[:, my_slice], np.any(my_slice == False)


def _cancel_unique_in_place(matrix: np.ndarray) -> bool:
    """
    Cancels literals whose truth value is already determined by appearing in the normal form on their own.
    Let A be a literal and X a term in CNF or DNF. Then the simplification works as follows.
    If X is in CNF
        not A and (A or X) => not A and X.
    If X is in DNF
        not A or (A and X) => not A or X.
    """

    p = (
        matrix * matrix[:, ((matrix != 0).sum(axis=0) == 1)].sum(axis=1)[:, np.newaxis]
        == -1
    )
    matrix[p] = 0
    return p.sum() > 0


def _single_difference_in_place(matrix: np.ndarray) -> bool:
    """
    Cancels opposite literals.
    Let A be a literal and X a term in CNF or DNF. Then the simplification works as follows.
    If X is in CNF
        (A or X) and (not A or X) => X.
    If X is in DNF
        (A and X) or (not A and X) => X.

    The algorithm does these simplifications in arbitrary order.
    Furthermore it only uses each column once.
    """

    differences = np.nonzero(
        (matrix[:, np.newaxis, :] - matrix[:, :, np.newaxis] != 0).sum(axis=0) == 1
    )
    used = set()
    for i, j in zip(differences[0], differences[1]):
        if i not in used and j not in used:
            matrix[:, [i, j]] = np.sign(matrix[:, i] + matrix[:, j])[:, np.newaxis]
            used.update((i, j))
    return len(differences[0]) > 0


def append(termA: np.ndarray, termB: np.ndarray) -> np.ndarray:
    """
    If the input is in CNF, calculates

        termA and termB.

    If the input is in DNF, calculates

        termA or termB.

    Parameters
    ----------
    value
        a term, either in CNF or DNF.
    term
        a term, in the same normal form.

    Returns
    -------
    np.ndarray
        A term in the same normal form as the input terms.
    """

    return np.c_[termA, termB]


def distribute(value: np.ndarray, term: np.ndarray) -> np.ndarray:
    """
    If the input is in CNF, calculates

        termA or termB.

    If the input is in DNF, calculates

        termA and termB.

    This is done by "multiplying" both "sums" using the distributive laws.

    Parameters
    ----------
    value
        a term, either in CNF or DNF.
    term
        a term, in the same normal form.

    Returns
    -------
    np.ndarray
        A term in the same normal form as the input terms.
    """

    return (
        np.sign(value[:, np.newaxis, :] + term[:, :, np.newaxis])
        .T[np.nonzero(np.all(np.einsum("ik,ij->ijk", value, term) != -1, axis=0))[::-1]]
        .T
    )
