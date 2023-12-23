import math
from typing import Union, Optional
import numpy as np
import scipy.sparse as sparse
from scipy.spatial.distance import pdist


def grid_distance_graph(grid_shape, max_dist=3.0, normalize=False, dist="euklid") -> sparse.csr_array:
    """A function to create a grid-like graph where every vertex is connected to all neighbors that are close (in grid coordinates).

    Returns a 2-dimensional matrix, where the index of each axis represents the index of a square.
    The value is the distance between the squares in the grid.

    Parameters
    ----------
    grid_shape : shape
        A shape.
    max_dist : float
        Maximal distance of two connected vertices.
    normalize : bool
        If True, distances are normalized.
    dist : {'euklid', 'manhattan'}
        The metric that will be used to calculate the distances.

    Returns
    -------
    sparse.csr_array
        A similarity graph between the pixels.
    """

    num_pixels = grid_shape[0]*grid_shape[1]
    max_dist_int = int(np.ceil(max_dist))
    neighborhood_size = 2 * max_dist_int + 1
    coord_neighborhood = np.stack(np.meshgrid(range(-max_dist_int, max_dist_int+1), range(-max_dist_int, max_dist_int+1)), 2).reshape(neighborhood_size * neighborhood_size, 2)

    if dist == "euklid":
        dist_neighborhood = np.sqrt(np.square(coord_neighborhood).sum(axis=1))
    else:
        dist_neighborhood = np.abs(coord_neighborhood).sum(axis=1)
    dist_neighborhood[dist_neighborhood > max_dist] = 0
    coord_neighborhood = coord_neighborhood[dist_neighborhood > 1e-8, :]
    dist_neighborhood = dist_neighborhood[dist_neighborhood > 1e-8]
    if normalize:
        dist_neighborhood /= max_dist

    pixel_coords = np.stack(np.meshgrid(range(grid_shape[0]), range(grid_shape[1]), indexing='ij'), 2)
    pixel_coords = pixel_coords[:,:,:,np.newaxis] + coord_neighborhood[:,:,np.newaxis,np.newaxis].T

    valid_coords = (pixel_coords>=0).all(axis=2) & (pixel_coords < np.array(grid_shape)[np.newaxis,np.newaxis,:,np.newaxis]).all(axis=2)
    valid_neighbors = valid_coords.reshape(num_pixels, valid_coords.shape[2])

    index_neighborhood = (coord_neighborhood * [grid_shape[1], 1]).sum(axis=1)

    all_neighbor_indices = np.arange(num_pixels)[:,np.newaxis] + index_neighborhood
    all_data = np.broadcast_to(dist_neighborhood, (all_neighbor_indices.shape[0], dist_neighborhood.shape[0]))

    num_neighbors = valid_neighbors.sum(axis=1)
    num_neighbors_total = valid_neighbors.shape[0]*valid_neighbors.shape[1]

    sel_neighbors = valid_neighbors.reshape(num_neighbors_total)
    indices = all_neighbor_indices.reshape(num_neighbors_total)[sel_neighbors]
    indptr = np.cumsum(np.r_[np.zeros(1, dtype=int),num_neighbors])
    data = all_data.reshape(num_neighbors_total)[sel_neighbors]

    mat = sparse.csr_array((data, indices, indptr))
    mat.sort_indices()
    return mat

def image_to_similarity_graph(image:np.ndarray, pixel_coord_distance_graph:sparse.csr_array, sigma_c:float=0.2, sigma_d:float=3, threshold:float=0.01, color_sim_p=1):
    """Computes a similarity graph for pixels of an image.

    Uses a graph to describe the differences between the pixel coordinates and then uses
    a gaussian function to calculate the similarity from this distance.
    A gaussian function is also used for color similarity by applying it to the sum of the absolute values of the
    differences in color channels.

    Parameters
    ----------
    image : np.ndarray
        An image, either a (width, height, 3) or (width, height, 1) grayscale image.
    pixel_coord_distance_graph : sparse.csr_array
        A sparse matrix containing the distances between pixel coordinates.
    sigma_c : float
        The spread of the gaussian function used for the similarity calculation of the color value.
    sigma_d : float
        The spread of the gaussian function used for the similarity calculation of the distance value.
    threshold : float
        Edges below this value get removed in the resultant similarity graph.

    Returns
    -------
    sprase.csr_array
        A sparse similarity graph between pixels of the image.
    """

    flat_image = image.reshape(image.shape[0] * image.shape[1], image.shape[2] if len(image.shape) == 3 else 1)
    neighborhood_center_colors = flat_image.repeat(pixel_coord_distance_graph.indptr[1:] - pixel_coord_distance_graph.indptr[:-1], axis=0)
    color_similarities = np.power(np.abs(neighborhood_center_colors - flat_image[pixel_coord_distance_graph.indices, :]), color_sim_p).sum(axis=1)
    color_similarities -= color_similarities.min()
    color_similarities = color_similarities / color_similarities.max()
    color_similarities = np.exp(-color_similarities / (sigma_c ** 2))

    similarity_graph = pixel_coord_distance_graph.copy()
    if similarity_graph.data.max() > similarity_graph.data.min():
        similarity_graph.data -= similarity_graph.data.min()
        similarity_graph.data /= similarity_graph.max()
    similarity_graph.data = np.exp(-similarity_graph.data / (sigma_d ** 2))

    similarity_graph.data *= color_similarities

    similarity_graph.data[similarity_graph.data < threshold] = 0
    similarity_graph.eliminate_zeros()
    return similarity_graph

def cosine_similarity(data: np.ndarray, sim_thresh: float = 1e-10, max_neighbours: int = None, return_sparse: bool=True, sequential: bool = False) -> Union[np.ndarray, sparse.csr_array]:
    """Return the cosine similarity matrix of the rows of the matrix data.

    Parameters
    ----------
    data : np.ndarray
        The data.
    sim_thresh : float
        Similarities smaller than sim_thresh are set to 0.
    return_sparse : bool
        Whether to return a sparse matrix.
    sequential : bool
        Use less memory, slower if memory was not an issue.

    Returns
    -------
    np.ndarray
        A matrix of shape (``data.shape[0]``, ``data.shape[0]``) containing the cosine similarities of the rows.
    """

    neighbors_to_remove = data.shape[0] - max_neighbours if max_neighbours is not None and max_neighbours>0 else None
    neighbors_to_remove = max(0, neighbors_to_remove)

    norm = np.sqrt(np.square(data).sum(axis=1))
    norm[norm == 0] = 1
    data_normed = data / norm[:, np.newaxis]
    if not sequential:
        adj_matrix = data_normed @ data_normed.T
        np.fill_diagonal(adj_matrix,0)
        if neighbors_to_remove:
            part = np.argpartition(adj_matrix, neighbors_to_remove, axis=1)
            for i in range(adj_matrix.shape[0]):
                adj_matrix[i,part[i,:neighbors_to_remove]] = 0
        adj_matrix[adj_matrix < sim_thresh] = 0
        adj_matrix += adj_matrix.T
        adj_matrix /= 2
        return sparse.csr_matrix(adj_matrix) if return_sparse else adj_matrix

    adj_matrix = sparse.lil_matrix((data_normed.shape[0], data_normed.shape[0]))
    for i in range(data_normed.shape[0]):
        sim = data_normed @ data_normed[i,:]
        sim[i] = 0
        if neighbors_to_remove:
            part = np.argpartition(sim, neighbors_to_remove)
            sim[part[:neighbors_to_remove]] = 0
        adj_matrix[sim >= sim_thresh,i] = sim[sim >= sim_thresh]
    adj_matrix = adj_matrix.tocsr() if return_sparse else adj_matrix.todense()
    adj_matrix += adj_matrix.T
    adj_matrix /= 2
    return adj_matrix

def hamming_similarity(data: np.ndarray, sim_thresh: int = 0, return_sparse: bool=True, sequential: bool = True) -> Union[np.ndarray, sparse.csr_array]:
    if not sequential:
        adj_matrix = (data[:,np.newaxis,:] == data[np.newaxis,:,:]).sum(axis=2)
        adj_matrix[adj_matrix < sim_thresh] = 0
        np.fill_diagonal(adj_matrix,0)
        return sparse.csr_matrix(adj_matrix) if return_sparse else adj_matrix
    sim_datatype = np.uint8 if data.shape[1]<1<<8 else np.uint16 if data.shape[1]<1<<16 else int
    sim_data, indices, indptr = [],  [], np.empty(data.shape[0] + 1, dtype=int)
    indptr[0] = 0
    for i in range(data.shape[0]):
        sim = (data[i,:] == data[i+1:, :]).sum(axis=1)
        idcs = np.nonzero(sim > sim_thresh)[0]
        indptr[i+1] = indptr[i] + len(idcs)
        indices.append(idcs + i+1)
        sim_data.append(sim[idcs])
    sim_matrix = sparse.csr_matrix((np.concatenate(sim_data), np.concatenate(indices), indptr), shape=(data.shape[0],data.shape[0]), dtype=sim_datatype)
    sim_matrix += sim_matrix.T
    return sim_matrix if return_sparse else sim_matrix.todense()




def k_nearest_neighbors(X:np.ndarray, metric:str='precomputed', k:int=1, ties_all:bool=False) -> sparse.csr_matrix:
    """Creates a k-nearest neighbor graph (or something like a k-nearest neighbor graph) from distances.

    Parameters
    ----------
    X : np.ndarray
        If metric is 'precomputed' `X` is a condensed distance matrix (see :meth:`scipy.spatial.distance.squareform`).
        Otherwise `X` is the data and :meth:`scipy.spatial.distance.pdist` is called taking `X` as its argument.
    metric : str
        Either 'precomputed' or a metric from :meth:`scipy.spatial.distance.pdist`.
        If not 'precomputed', `metric` is forwarded to the :meth:`scipy.spatial.distance.pdist` call.
    k : int
        Number of neighbors.
    ties_all : bool
        If True, every node gets a connection to all other nodes that have a distance smaller than the k-nearest neighbor,
        i.e. if there are multiple k-nearest neighbors, we connect them all, resulting in a graph that might have degrees greater than `k`
        (so it is not really a k-nn graph, but something similar where ties are broken in a special way).
        If False, only one of possibly multiple k-nearest neighbors is connected.

    Returns
    -------
    sparse.csr_matrix
        Sparse adjacency matrix of a k-nn graph (note that this graph is directed).
    """

    pw_distances = pdist(X, metric=metric) if metric != 'precomputed' else X

    data_size = math.ceil(math.sqrt(2 * pw_distances.shape[0]))
    adj_mat = sparse.lil_matrix((data_size, data_size))

    dist_idcs = np.empty(data_size - 1, dtype=int)
    neigbhbor_idcs_func = (lambda dist_idcs_part : np.flatnonzero(pw_distances[dist_idcs] <= pw_distances[dist_idcs][dist_idcs_part].max())) if ties_all else (lambda dist_idcs_part : dist_idcs_part)
    start_idx_tail = 0

    for i in range(data_size-1):
        end_idx_tail = start_idx_tail + data_size - (i+1)
        dist_idcs[i:] = np.arange(start_idx_tail, end_idx_tail)

        dist_idcs_part = np.argpartition(pw_distances[dist_idcs], kth=k)[:k]
        neigh_idcs = neigbhbor_idcs_func(dist_idcs_part)
        neigh_idcs[neigh_idcs>=i] += 1
        adj_mat[i,neigh_idcs] = 1

        dist_idcs[:i] += 1
        dist_idcs[i] = start_idx_tail
        start_idx_tail = end_idx_tail

    dist_idcs_part = np.argpartition(pw_distances[dist_idcs], kth=k)[:k]
    adj_mat[data_size-1,neigbhbor_idcs_func(dist_idcs_part)] = 1

    return adj_mat.tocsr()

def epsilon_neighborhood_graph(X:np.ndarray, max_dist: float, dist2sim=None, metric:str='precomputed') -> sparse.csr_matrix:
    """Creates a neighborhood graph from precomputed distances.

    Every node is connected to every other node that has distance at most `max_dist`.

    Parameters
    ----------
    X : np.ndarray
        If metric is 'precomputed' `X` is a condensed distance matrix (see :meth:`scipy.spatial.distance.squareform`).
        Otherwise `X` is the data and :meth:`scipy.spatial.distance.pdist` is called taking `X` as its argument.
    max_dist : float
        Maximal distance.
    dist2sim : Callable[[float], float], optional
        Function that transforms a similarity to a distance.
        Defaults to the constant 1 function (i.e. unweighted).
    metric : str
        Either 'precomputed' or a metric from :meth:`scipy.spatial.distance.pdist`.


    Returns
    -------
    sparse.csr_matrix
        Sparse adjacency matrix of the epsilon-neighborhood graph.
    """

    if dist2sim is None:
        dist2sim = lambda x: 1  # default weight is one for all near nodes

    pw_distances = pdist(X, metric=metric) if metric != 'precomputed' else X

    data_size = math.ceil(math.sqrt(2 * pw_distances.shape[0]))
    adj_mat = sparse.lil_matrix((data_size, data_size))

    start_idx_tail = 0
    for i in range(data_size):
        end_idx_tail = start_idx_tail + data_size - (i+1)

        i_dist = pw_distances[start_idx_tail:end_idx_tail]
        near_idcs = np.flatnonzero(i_dist<max_dist)
        adj_mat[i,1+i+near_idcs] = dist2sim(i_dist[near_idcs])
        start_idx_tail = end_idx_tail

    adj_mat = adj_mat.tocsr()
    return adj_mat.maximum(adj_mat.T)


def cosine_similarity_graph_csr_data(mat: sparse.csr_matrix, sim_thresh=0.25, weight_range=None, chunk_size=100, verbose=False):
    """
    Creates a similarity graph on the data based on cosine similarity between the data points.
    Works with sparse matrices and takes less memory than :meth:`cosine_similarity_graph`.

    Parameters
    ----------
    data : scipy.sparse.csr_matrix
        The data.
    sim_thresh : float
        Minimum similarity for an edge.
    weight_range : list of floats, optional
        If note None, scale weights to this range.
    chunk_size : int
        The size of the chunks.

    Returns
    -------
    adj_matrix : scipy.sparse.csr_matrix
        A sparse adjacency matrix.
    """

    if not isinstance(mat, sparse.csr_matrix):
        mat = mat.tocsr()

    norm = np.empty((mat.shape[0],1))
    for i in range(mat.shape[0]):
        norm[i,0] = np.sqrt(np.square(mat.data[mat.indptr[i]:mat.indptr[i+1]]).sum())
    norm[norm==0] = 1 # avoid division by 0

    data = []
    indices = []
    indptr = [np.array([0], dtype=int)]
    for chunk_start in range(0,mat.shape[0],chunk_size):
        chunk_end = min(chunk_start+chunk_size, mat.shape[0])
        sim = np.asarray(((mat[chunk_start:chunk_end,:]@mat.T)/norm[chunk_start:chunk_end])/norm.T)
        np.fill_diagonal(sim[:,chunk_start:chunk_end],0)
        valid = sim>=sim_thresh
        num_nonzero = valid.sum(axis=1)
        indptr.append(indptr[-1][-1]+np.cumsum(num_nonzero))
        indices.append(np.where(valid)[1])
        data.append(sim.ravel()[valid.ravel()])
        if verbose:
            print(f"{chunk_end}: adj mat nonzero: {sum(len(chunk) for chunk in data)}")
    adj_matrix = sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.concatenate(indptr)), shape=(mat.shape[0],mat.shape[0]))
    if weight_range is not None:
        adj_matrix.data = weight_range[0] + (adj_matrix.data - adj_matrix.data.min()) / (adj_matrix.data.max() - adj_matrix.data.min()) * (weight_range[1] - weight_range[0])
    return adj_matrix
