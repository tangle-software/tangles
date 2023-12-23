from tangles.analysis import coherence_levels, complexity_levels, visibility
from ._test_data.generate_tangle_matrix import _generate_basic_example_tangle_matrix
import numpy as np

def test_visibility():
    _, tangle_matrix = _generate_basic_example_tangle_matrix()
    assert np.all(visibility(tangle_matrix, column_orders=np.arange(tangle_matrix.shape[1])) == np.array([4, 1, 3, 3, 3, 1, 2, 3, 3]))

def test_coherence_level_function():
    _coherence_as_intended()
    _coherence_if_empty()

def test_complexity_level_function():
    _complexity_just_one_tangle()
    _complexity_only_leafs()
    _complexity_not_just_leafs()

def _coherence_as_intended():
    intended_coherence = np.array([1, 5, 3, 7, 7, 9, 10, 10, 0], dtype=int)
    tangle_matrix = _generate_from_intended_coherence(intended_coherence)
    intended_coherence[intended_coherence == 10] = -1
    assert np.all(intended_coherence == coherence_levels(tangle_matrix))

def _coherence_if_empty():
    intended_coherence = np.array([], dtype=int)
    tangle_matrix = _generate_from_intended_coherence(intended_coherence)
    assert coherence_levels(tangle_matrix).shape == (0,)

def _complexity_only_leafs():
    tangle_matrix, _ = _generate_basic_example_tangle_matrix()
    assert np.all(complexity_levels(tangle_matrix) == np.array([1, 6, 6, 6, 6]))

def _complexity_not_just_leafs():
    _, tangle_matrix = _generate_basic_example_tangle_matrix()
    assert np.all(complexity_levels(tangle_matrix) == np.array([1, 6, 6, 6, 6, 0, 1, 3, 3]))

def _complexity_just_one_tangle():
    assert np.all(complexity_levels(np.array([1, -1, 1, -1, 1, -1])[np.newaxis, :]) == np.array([0]))

def _generate_from_intended_coherence(intended_coherence:np.ndarray, num_feats:int=10):
    tangle_matrix = np.zeros((len(intended_coherence), num_feats), dtype=np.int8)
    for row, coherence in enumerate(intended_coherence):
        tangle_matrix[row, :coherence] = _random_plus_one_minus_one(coherence)
    return tangle_matrix

def _random_plus_one_minus_one(length:int):
    rand_array = np.random.rand(length)
    rand_array[rand_array>=0.5] = 1
    rand_array[rand_array<0.5] = -1
    return rand_array