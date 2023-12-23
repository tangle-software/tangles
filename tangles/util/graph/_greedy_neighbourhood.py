from typing import Union
import numpy as np
import scipy.sparse as sparse
import heapq

class _upg:
    def __init__(self, forbidden=None):
        self.heap = []
        self.set = set()
        if forbidden is not None:
            self.set.update(forbidden)

    def push(self, o, v):
        if o not in self.set:
            heapq.heappush(self.heap,(v,o))
            self.set.add(o)

    def push_all(self, os, vs):
        for (o,v) in zip(os, vs):
            self.push(o,v)

    def pop(self):
        (v,o) = heapq.heappop(self.heap)
        return o

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str([o for (v,o) in self.heap])


def greedy_neighborhood_old(A: sparse.csr_array,
                        start_neighborhood: Union[int, list, np.ndarray],
                        size: int,
                        take: int = 5,
                        forbidden_vertices: Union[list, set] = None,
                        minimize_weight=False):
    """Search a neighborhood in the graph with adjacency matrix `A`.

    Parameters
    ----------
    A : sparse.csr_array
        The adjacency matrix of a graph.
    start_neighborhood : int, list or np.ndarray
        Indices of vertices to start with.
    size : int
        Size of the neighborhood.
    take : int
        Number of neighbors added in each step. Only the best ones are taken.
    forbidden_vertices : list or set, optional
        Set of vertices that should not be added to the neighborhood.
    minimize_weight : bool
        If True, the greedy procedure selects edges with the smallest weight.

    Returns
    -------
    np.ndarray
        The indices of the found neighborhood.
    """

    if isinstance(start_neighborhood, int):
        neighborhood = [start_neighborhood]
    else:
        assert isinstance(start_neighborhood, list) or isinstance(start_neighborhood, np.ndarray)
        neighborhood = start_neighborhood if isinstance(start_neighborhood, list) else start_neighborhood.tolist()

    sign = 1 if minimize_weight else -1

    if forbidden_vertices is None:
        heap = _upg(forbidden=neighborhood)
    else:
        if isinstance(forbidden_vertices, np.ndarray):
            forbidden_vertices = forbidden_vertices.tolist()
        heap = _upg(forbidden=neighborhood+forbidden_vertices)
    start_graph = A[neighborhood,:]
    heap.push_all(start_graph.indices, sign*start_graph.data)
    while len(neighborhood)<size and len(heap)>0:
        for o in [heap.pop() for _ in range(min(take, size-len(neighborhood),len(heap)))]:
            neighborhood.append(o)
            heap.push_all(A.indices[A.indptr[o]:A.indptr[o+1]], sign*A.data[A.indptr[o]:A.indptr[o+1]])
    return np.array(neighborhood)


def greedy_neighborhood(A: sparse.csr_array,
                        start_neighborhood: Union[int, list, np.ndarray],
                        size: int,
                        take: int = 5,
                        forbidden_vertices: Union[list, set] = None,
                        minimize_weight=False):
    """Search a neighborhood in the graph with adjacency matrix `A`.

    Parameters
    ----------
    A : sparse.csr_array
        The adjacency matrix of a graph.
    start_neighborhood : int, list or np.ndarray
        Indices of vertices to start with.
    size : int
        Size of the neighborhood.
    take : int
        Number of neighbors added in each step. Only the best ones are taken.
    forbidden_vertices : list or set, optional
        Set of vertices that should not be added to the neighborhood.
    minimize_weight : bool
        If True, the greedy procedure selects edges with the smallest weight.

    Returns
    -------
    np.ndarray
        The indices of the found neighborhood.
    """

    neighborhood = np.zeros(A.shape[0], dtype=bool)
    neighborhood[start_neighborhood] = True

    while (n_size := neighborhood.sum()) < size:
        boundary = A[neighborhood,:].sum(axis=0).A1
        boundary[neighborhood] = 0
        if forbidden_vertices is not None:
            boundary[forbidden_vertices] = 0
        if minimize_weight:
            boundary[boundary!=0] = 1 / boundary[boundary!=0]
        num_add = min(take, size-n_size, (boundary>0).sum())
        if num_add == 0:
            break
        partition_idcs = np.argpartition(boundary, boundary.shape[0]-num_add)
        neighborhood[partition_idcs[boundary.shape[0]-num_add:]] = True
    return neighborhood
