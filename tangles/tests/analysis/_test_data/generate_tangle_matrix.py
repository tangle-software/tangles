import numpy as np
from tangles.separations import FeatureSystem
from .load_numpy_from_csv import load

def _generate_basic_example_tangle_matrix():
    tangle_matrix = np.zeros((9, 10), dtype=np.int8)
    tangle_matrix[0, :] = _random_plus_one_minus_one(10)
    tangle_matrix[1, :] = tangle_matrix[0, :]
    tangle_matrix[1, 1] = -tangle_matrix[0, 1]
    tangle_matrix[2, :] = tangle_matrix[1, :]
    tangle_matrix[2, 6] = -tangle_matrix[1, 6]
    tangle_matrix[3, :] = tangle_matrix[1, :]
    tangle_matrix[3, 3] = -tangle_matrix[1, 3]
    tangle_matrix[4, :] = tangle_matrix[3, :]
    tangle_matrix[4, 6] = -tangle_matrix[3, 6]
    tangle_matrix[0, 5:] = 0
    tangle_matrix[1, 7:] = 0
    tangle_matrix[5, 0] = tangle_matrix[0, 0]
    tangle_matrix[6, :3] = tangle_matrix[1, :3]
    tangle_matrix[7, :6] = tangle_matrix[1, :6]
    tangle_matrix[8, :6] = tangle_matrix[3, :6]
    return tangle_matrix[:5, :], tangle_matrix

def _random_plus_one_minus_one(length:int):
    rand_array = np.random.rand(length)
    rand_array[rand_array>=0.5] = 1
    rand_array[rand_array<0.5] = -1
    return rand_array

def generate_small_example_with_splitting():
    tangle_matrix = load('big_5_tangles_with_splitting.csv')
    feat_sys = FeatureSystem(10)
    feat_sys.add_seps(load('big_5_feats.csv').T)
    return tangle_matrix, feat_sys

def generate_small_example():
    tangle_matrix = load('big_5_tangles.csv')
    feat_sys = FeatureSystem(10)
    feat_sys.add_seps(load('big_5_feats.csv').T)
    return tangle_matrix, feat_sys