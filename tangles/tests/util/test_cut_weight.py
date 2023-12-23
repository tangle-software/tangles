import pytest
import scipy.sparse as sparse
from skimage.color import rgb2gray
from skimage import io
from sklearn.feature_extraction import image as img
import networkx as nx
import numpy as np
from tangles.util.graph.cut_weight import cut_weight_order, ratiocut_order, ncut_order
from tangles.separations.finding import minimize_cut
import tangles.tests.path as path

def order_func(adjacency_matrix: sparse.csr_array, name: str):
    if name == 'cut_weight':
        return cut_weight_order(adjacency_matrix)
    elif name == 'ratiocut':
        return ratiocut_order(adjacency_matrix)
    elif name == 'ncut':
        return ncut_order(adjacency_matrix)

##### PATH LENGTH TWO EXAMPLE #####

@pytest.fixture
def path_length_two_data():
    return {
        'adjacency_matrix': sparse.csr_array(
            np.array(
                [
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 0]
                ]
            )
        ),
        'feats': np.array(
            [
                [1, -1, -1, 1],
                [-1, 1, 1, 1],
                [-1, 1, -1, 1]
            ]
        ),
        'cut_weight': np.array([2.0, 2.0, 1.0, 0.0]),
        'ratiocut': np.array([3.0, 3.0, 1.5, 0]),
        'ncut': np.array([2.0, 2.0, 4/3, 0.0])
    }

@pytest.mark.parametrize("order_func_type", ['cut_weight', 'ratiocut', 'ncut'])
def test_path_length_two(path_length_two_data: dict, order_func_type: str):
    adjacency_matrix = path_length_two_data['adjacency_matrix']
    feats = path_length_two_data['feats']
    intended_order = path_length_two_data[order_func_type]

    order = order_func(adjacency_matrix, order_func_type)

    for sep_index in range(feats.shape[1]):
        assert order(feats[:, sep_index]) == intended_order[sep_index]
    assert np.all(order(feats) == intended_order)

##### TEST CHANGE RANDOMIZED #####

@pytest.fixture
def random_graph() -> sparse.csr_array:
    adj = np.random.random_sample((100, 100)) - 0.5
    adj = adj + adj.T
    np.fill_diagonal(adj, 0)
    return sparse.csr_array(adj)

@pytest.mark.parametrize("order_func_type", ['cut_weight', 'ratiocut', 'ncut'])
def test_change(random_graph: sparse.csr_array, order_func_type: str):
    n_feats = 5
    order = order_func(random_graph, order_func_type)
    np.random.seed(1)
    for _ in range(n_feats):
        sep = 2 * (np.random.random(random_graph.shape[0])>=0.5) - 1
        assert _change_correct(sep, order)

def _change_correct(sep: np.ndarray, order_func, tol = 1e-8) -> bool:
    change = order_func.discrete_derivative(sep)
    prev_order = order_func(sep)
    for vertex in range(len(sep)):
        sep_swapped = sep.copy()
        sep_swapped[vertex] = -sep_swapped[vertex]
        new_order = order_func(sep_swapped)
        if not abs(new_order - prev_order - change[vertex]) < tol:
            return False
    return True

##### MIMIZE CUT EXAMPLE #####

def mona_lisa() -> tuple[sparse.csr_array, int]:
    image = rgb2gray(io.imread(path.get_test_data_path("Images/MonaLisaVerySmall.png")))
    A = img.img_to_graph(image)
    A.data = np.exp(-A.data / (0.5 * A.data.std()))
    A.data[A.data < 1e-10] = 0
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr(), A.shape[0]

def barabasi(n_vertices: int, seed: int) -> tuple[sparse.csr_array, int]:
    G = nx.barabasi_albert_graph(n_vertices, 2, seed=seed)
    A = nx.adjacency_matrix(G)
    return A.tocsr()

@pytest.mark.parametrize("order_func_type", ['cut_weight', 'ratiocut', 'ncut'])
@pytest.mark.long
@pytest.mark.skip(reason="Skipping long tests by default.")
def test_localmin_mona(order_func_type: str):
    mona_adj, mona_n_vertices = mona_lisa()
    order = order_func(mona_adj, order_func_type)
    assert _localmin_test(10, mona_n_vertices, order)

@pytest.mark.parametrize("order_func_type", ['cut_weight', 'ratiocut', 'ncut'])
@pytest.mark.long
@pytest.mark.skip(reason="Skipping long tests by default.")
def test_localmin_barabasi(order_func_type: str):
    barabasi_n_vertices = 100
    for i in range(1):
        assert _localmin_test(10, barabasi_n_vertices, order_func(barabasi(barabasi_n_vertices, i), order_func_type))

def _localmin_test(n_initial_cuts: int, n_vertices: int, cut_weight_func) -> bool:
    np.random.seed(1)
    for _ in range(n_initial_cuts):
      s = 2 * (np.random.random(n_vertices)>=0.5) - 1
      t = minimize_cut(s, cut_weight_func)
      if not _check_local_min(t, cut_weight_func):
        return False
    return True

def _check_local_min(sep: np.ndarray, order_func, eps: float = 0.0) -> bool:
  weight_original = order_func(sep)

  for i in range(sep.shape[0]):
    sep_swapped = sep.copy()
    sep_swapped[i] = -sep_swapped[i]
    weight_swapped = order_func(sep_swapped)
    if weight_swapped < weight_original-eps:
      print(f"swapped {weight_swapped} <vs> original {weight_original}")
      return False

  return True