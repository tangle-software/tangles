from typing import Union
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as sparse_linalg
import scipy.sparse.csgraph as csg
from ._util import _threshold_partitions


def spectral_features(
    laplacian: Union[sparse.spmatrix, np.ndarray], k: int, return_eigenvectors=False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Compute spectral bipartitions directly by computing eigenvectors of the complete graph.

    Note: using this function on an disconnected graph might result in errors in eigenvalue computation.

    Parameters
    ----------
    laplacian : sparse.spmatrix or np.ndarray
        Laplacian matrix of a graph.
    k : int
        Number of spectral bipartitions to compute.

    Returns
    -------
    np.array
        -1/1-matrix containing the spectral bipartitions as columns.
    """

    if sparse.issparse(laplacian):
        _, eigenvectors = sparse_linalg.eigsh(
            laplacian, k=k, which="LM", sigma=1e-12, v0=np.ones(laplacian.shape[0])
        )
    else:
        _, eigenvectors = np.linalg.eigh(laplacian)
    k = min(eigenvectors.shape[0], k)
    bips = _threshold_partitions(eigenvectors[:, :k])
    bips *= bips[0:1, :]
    return (bips, eigenvectors) if return_eigenvectors else bips


def spectral_features_splitted(
    laplacian: Union[sparse.spmatrix, np.ndarray],
    k: int,
    check_connected: bool = True,
    min_component_size_fraction: float = 0.01,
    ignore_small_components: bool = True,
) -> np.ndarray:
    """Compute spectral bipartitions of a graph after splitting into connected components.

    Usually this is faster than the direct method :meth:`spectral_bipartitions`.

    Parameters
    ----------
    laplacian : sparse.spmatrix or np.ndarray
        Laplacian matrix of a graph.
    k : int
        Number of spectral bipartitions to compute.
    check_connected : bool
        Check if the graph is connected.
        If it is not connected, create bipartitions separating out the connected components first.
        If `k` is bigger than the number of connected components, the remaining bipartitions will split the components.
    min_component_size_fraction : float
        Components of G smaller than `min_component_size_fraction` times the number of vertices are not analysed further.
    ignore_small_components : bool
        If True, the small components are on the small side of all bipartitions.
        Otherwise there is a bipartition for every connected component.

    Returns
    -------
    np.array
        -1/1-matrix containing the `k` spectral bipartitions of the graph as columns.
        The array might contain more than `k` separations, if the graph has more than `k` (big) connected components.
        The first few columns contain bipartitions separating out the connected components.
        If the graph is connected, the first column is constant 1.
    """

    if k < 1:
        return np.empty((laplacian.shape[0], 0))
    if check_connected:
        cc, comp_labels = csg.connected_components(
            laplacian < 0, directed=False, return_labels=True
        )
        if cc == 1:
            seps = spectral_features(laplacian, k)
        else:
            total_size = laplacian.shape[0]
            min_comp_size = total_size * min_component_size_fraction
            component_indicators = np.array(
                [(comp_labels == l) for l in range(cc)], dtype=bool
            ).T
            sizes = component_indicators.sum(axis=0)
            o = np.argsort(sizes)[::-1]
            sizes = sizes[o]
            component_indicators = component_indicators[:, o]
            n_big_comps = (
                np.flatnonzero(sizes < min_comp_size)[0]
                if sizes[-1] < min_comp_size
                else sizes.shape[0]
            )

            if ignore_small_components:
                seps = np.empty((total_size, max(k, n_big_comps)), dtype=np.int8)
                seps[:, :n_big_comps] = 2 * (component_indicators[:, :n_big_comps]) - 1
                n_seps = n_big_comps
            else:
                seps = np.empty((total_size, max(k, cc)), dtype=np.int8)
                seps[:, :cc] = 2 * component_indicators.astype(np.int8) - 1
                n_seps = cc

            if n_seps < k:
                k -= n_seps
                sizes = sizes[: min(n_big_comps, k)]
                seps_per_component = np.ones(sizes.shape[0], dtype=int)
                seps_per_component += np.ceil(
                    (k - sizes.shape[0]) * sizes / sizes.sum()
                ).astype(int)
                remove = seps_per_component.sum() - k
                if remove > 0:
                    seps_per_component[sizes.shape[0] - remove :] -= 1

                seps[:, n_seps:] = -1
                for i, s in enumerate(seps_per_component):
                    seps_c = spectral_features(
                        laplacian[component_indicators[:, i], :][
                            :, component_indicators[:, i]
                        ],
                        s + 1,
                    )
                    seps[component_indicators[:, i], n_seps : n_seps + s] = seps_c[
                        :, 1:
                    ]
                    n_seps += s
            elif n_seps > k:
                print(
                    "Warning, no spectral analysis: Graph has more connected components than number of requested separations!"
                )
            elif n_seps == k:
                print(
                    "Warning, no spectral analysis: Number of requested separations is equal to connected components!"
                )
        return seps
    return spectral_features(laplacian, k)
