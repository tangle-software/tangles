import numpy as np
import scipy.sparse as sparse

def _nodal_domains(A: np.ndarray, f: np.ndarray, type:str = 'weak', eps: float = 1e-10) -> np.ndarray:
    if type == 'weak':
        sel_n = np.nonzero(f < eps)[0]
        sel_p = np.nonzero(f > -eps)[0]
    elif type == 'strong':
        sel_n = np.nonzero(f < -eps)[0]
        sel_p = np.nonzero(f > eps)[0]
    else:
        sel_n = np.nonzero(f <= eps)[0]
        sel_p = np.nonzero(f > eps)[0]
    c_n = sparse.csgraph.connected_components(A[sel_n[:,None], sel_n])
    c_p = sparse.csgraph.connected_components(A[sel_p[:,None], sel_p])
    nd = np.zeros((A.shape[0],c_n[0]+c_p[0]))
    for i in range(c_n[0]):
        nd[sel_n[c_n[1] == i],i] = 1
    for i in range(c_p[0]):
        nd[sel_p[c_p[1] == i],i+c_n[0]] = 1
    return nd

def nodal_domains(A: np.ndarray, U: np.ndarray, type:str = 'weak', eps:float = 1e-10) -> np.ndarray:
    """Calculate the nodal domains of a function from the vertices of the graph with adjacency matrix `A` to the real numbers.

    The nodal domains are calculated for each column in `U`, interpreting each column in `U` as a separate function.

    The nodal domains are the connected components of the sides of a function.
    Here, one side is the set of vertices the function assigns a negative value to and the other side is the set of
    vertices the function assigns a positive value to (see the `type` parameter description for more details).

    Parameters
    ----------
    A : np.ndarray
        The adjacency matrix of a graph.
    U : np.ndarray
        A matrix or a vector. 
        If it is s a vector, it represents a function from the vertices of the graph with adjacency matrix `A` to the real numbers.
        If it is a matrix, each column represents such a function.
    type : {"weak", "strong", "decomposition"}
        Specifies how to handle the epsilon value to identify connected components:

        - 'weak' (default): The negative side are the values mapped to values smaller than epsilon.
          The positive side are the values mapped to values larger than -epsilon.
        - 'strong': Like 'weak' but those that close to zero, closer than epsilon are removed from both sides.
        - 'decomposition'. Those close to zero are added to the negative side.

    eps : float
        The epsilon value.

    Returns
    -------
    np.ndarray
        A matrix containing the indicator vectors of the nodal domains of each
        input vector as columns.
    """
    
    all_nd = []
    U_matrix = U[:, np.newaxis] if len(U.shape) == 1 else U
    for i in range(U_matrix.shape[1]):
        nds = _nodal_domains(A, U_matrix[:,i], type=type, eps=eps)  # what was this 'flat' in here? This returns an iterator! this can be fatal!
        if nds.shape[1] == 2:
            all_nd.append(nds[:,0][:,np.newaxis])
        else:
            all_nd.append(nds)
    return np.hstack(all_nd)