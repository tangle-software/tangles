from typing import Union
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.csgraph as csg

def _augmented_graph_S_T(A: sparse.csr_matrix, S, T):
    s_idx = A.shape[0]
    t_idx = s_idx+1

    ST = np.r_[S, T]
    st = np.r_[s_idx*np.ones(len(S), dtype=int), t_idx*np.ones(len(T),dtype=int)]
    sorting = ST.argsort()
    insert_idcs = A.indptr[ST[sorting]+1]
    st = st[sorting]

    indices = np.r_[np.insert(A.indices, insert_idcs, st), ST]
    data = np.r_[np.insert(A.data, insert_idcs, int(1e10)), np.ones(len(ST),dtype=int)*int(1e10)]
    indptr = np.r_[A.indptr, A.indices.shape[0]+len(S), A.indices.shape[0]+len(S)+len(T)]
    for i in ST+1:
        indptr[i:] += 1

    return sparse.csr_array((data,indices, indptr))

def min_S_T_cut(A : sparse.csr_array, S: Union[set, list], T: Union[set, list]):
    """Search a minimal weight `S`-`T`-cut in the graph with adjacency matrix `A`.

    `S` and `T` are subsets of the graph's vertex set. The cut separates the vertices in `S` from the vertices in `T`.
    Note: if `S` and `T` are not disjoint, the function will produce an error.

    Parameters
    ----------
    A : sparse.csr_array
        The adjacency matrix of a graph.
    S : set or list
        Subset of the vertices of the graph with adjacency matrix `A`. The vertices are identified by the the index of
        the row (or equivalently column) of that vertex in `A`. 
    T : set or list
        Subset of the vertices of the graph with adjacency matrix `A`. The vertices are identified by the the index of
        the row (or equivalently column) of that vertex in `A`. 

    Returns
    -------
    tuple (float, np.ndarray)
        The flow between `S` and `T` in the first component,
        a -1/1-array indicating the sides of the minimal cut between `S` and `T` in the second component. 
    """

    assert len(set(S) & set(T)) == 0

    scale = 1
    if not np.issubdtype(A.dtype, np.integer):
        scale = 1e6
        A = sparse.csr_array(((A.data*scale), A.indices, A.indptr), dtype=int)

    B = _augmented_graph_S_T(A, S, T)
    result = csg.maximum_flow(B, A.shape[0], A.shape[0]+1)

    residual_network = B - result.flow

    sep = np.ones(A.shape[0], dtype=np.int8)
    small_side = csg.breadth_first_order(residual_network, A.shape[0], return_predecessors=False)
    if (small_side>A.shape[0]).any():
        print("could not separate the sets... for some reason...")
    else:
        sep[small_side[1:]] = -1
    return result.flow_value/scale, sep