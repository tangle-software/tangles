from tangles.analysis import hard_clustering_tuebingen, soft_clustering_tuebingen
from tangles.separations import FeatureSystem
from ._test_data.load_numpy_from_csv import load
from ._test_data.generate_tangle_matrix import generate_small_example_with_splitting, generate_small_example
import numpy as np

def test_hard_clustering_tuebingen():
    _test_small_example_hard_clustering()
    _test_not_only_leafs_hard_clustering()

def test_soft_clustering_tuebingen():
    _test_small_example_soft_clustering()
    _test_soft_clustering_example()

def _test_small_example_hard_clustering():
    tangle_matrix, feat_sys = generate_small_example()
    clustering = hard_clustering_tuebingen(tangle_matrix, np.arange(len(feat_sys)), feat_sys)
    intended_clustering = load('big_5_hard_clustering.csv', dtype=float).T
    assert np.all(clustering == intended_clustering)

def _test_not_only_leafs_hard_clustering():
    tangle_matrix, feat_sys = generate_small_example_with_splitting()
    clustering = hard_clustering_tuebingen(tangle_matrix, np.arange(len(feat_sys)), feat_sys)
    intended_clustering = load('big_5_hard_clustering_with_splitting.csv', dtype=float).T
    assert np.all(clustering == intended_clustering)

def _test_small_example_soft_clustering():
    tangle_matrix, feat_sys = generate_small_example()
    clustering = soft_clustering_tuebingen(tangle_matrix, np.arange(len(feat_sys)), feat_sys)
    intended_clustering = load('big_5_soft_clustering.csv', dtype=float).T
    assert np.sum(clustering - intended_clustering) < 0.000001

def _test_soft_clustering_example():
    tangle_matrix, feat_sys = _generate_soft_clustering_example()
    clustering = soft_clustering_tuebingen(tangle_matrix, np.array([0]*6), feat_sys)
    intended_clustering = load('soft_soft_clustering.csv', dtype=float).T
    assert np.all(clustering == intended_clustering)

def _generate_soft_clustering_example():
    tangle_matrix = load('soft_tangles.csv')
    feat_sys = FeatureSystem(1)
    feat_sys.add_seps(load('soft_features.csv').T)
    return tangle_matrix, feat_sys