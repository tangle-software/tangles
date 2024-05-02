import numpy as np
from scipy import sparse


def atomic_sets(seps: np.ndarray) -> list[np.ndarray]:
    """Find the atomic sets for a separation array.

    Parameters
    ----------
    seps : np.ndarray
        Separation array.

    Returns
    -------
    list of np.ndarray
        A list of atomic sets, encoded by the row indices of the elements contained in the atomic set.
    """

    row_sort_idx = np.lexsort(np.rot90(seps))
    S_sorted_rows = seps[row_sort_idx]
    rows_with_change = np.nonzero(S_sorted_rows[1:, :] != S_sorted_rows[:-1, :])[0] + 1
    atomic_set_boundaries = np.concatenate(
        (
            [0],
            rows_with_change[np.nonzero(rows_with_change[1:] != rows_with_change[:-1])],
            [rows_with_change[-1], seps.shape[0]],
        )
    )
    return [
        list(row_sort_idx[range(start, next_start)])
        for (start, next_start) in zip(
            atomic_set_boundaries[:-1], atomic_set_boundaries[1:]
        )
    ]


def atomic_set_indicators(atoms: list[np.ndarray]) -> sparse.csr_matrix:
    """Turn a list of atomic sets into a sparse indicator matrix.

    Parameters
    ----------
    atoms : list of np.ndarray
        List of atoms.

    Returns
    -------
    indicator_matrix: sparse.csc_matrix
        A sparse matrix of shape (number of rows of sep matrix, number of atoms) indicating what rows are
        contained in which atom.
    """

    atom_indices = sum([[i] * len(a) for i, a in enumerate(atoms)], start=[])
    rows_in_sep_matrix = sum(atoms, start=[])
    return sparse.csc_matrix(
        (np.ones(len(atom_indices)), (rows_in_sep_matrix, atom_indices)), dtype=np.int8
    )


def seps_to_atomic(seps: np.ndarray, atoms: list[np.ndarray]) -> np.ndarray:
    """Contract the rows of a matrix to their atoms.

    Parameters
    ----------
    seps : np.ndarray
        The separation matrix.
    atoms : list of np.ndarray
        The list of atoms.

    Returns
    -------
    np.ndarray
        Contracted separation matrix.
    """

    S_atomic = -np.ones((len(atoms), seps.shape[1]), dtype=np.int8)
    for sep_index in range(seps.shape[1]):
        for atom_index in range(len(atoms)):
            if np.all(seps[atoms[atom_index], sep_index] > 0):
                S_atomic[atom_index, sep_index] = 1
    return S_atomic


def atomic_to_seps(seps_atomic: np.ndarray, atoms: list[np.ndarray]) -> np.ndarray:
    """Turns a contracted separation matrix back to a normal one.

    Parameters
    ----------
    seps_atomic : np.ndarray
        Contracted separation matrix.
    atoms : list of np.ndarray
        List of atoms.

    Returns
    -------
    np.ndarray
        Separation matrix.
    """

    num_data = len(set().union(*atoms))
    seps = -np.ones((num_data, seps_atomic.shape[1]), dtype=np.int8)
    for s in range(seps.shape[1]):
        nonz = np.nonzero(seps_atomic[:, s] > 0)[0]
        seps[list(set().union(*[atoms[nz] for nz in nonz])), s] = 1
    return seps
