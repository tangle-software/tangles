import numpy as np
from itertools import combinations, product
from tangles.util import unique_cols
from tangles._typing import SetSeparationOrderFunction


# S (as well as T) is a matrix in -1/1-indictor format!
def compute_corners(S, T=None):
    if T is None:
        T = S
        C = np.empty((S.shape[0], 4*S.shape[1]*(S.shape[1]-1)//2), dtype=np.int8)
        combis = combinations(range(S.shape[1]), 2)
    else:
        C = np.empty((S.shape[0], 4*S.shape[1]*T.shape[1]), dtype=np.int8)
        combis = product(range(S.shape[1]), range(T.shape[1]))
    for i,((id_1,id_2),(ori_1, ori_2)) in enumerate(product(combis, [(-1,-1),(-1,1),(1,-1),(1,1)])):
        C[:,i] = np.minimum(ori_1*S[:,id_1],ori_2*T[:,id_2])
    return unique_cols(C)

def bip_indicator_from_subset_indicator(T):
    S = -np.ones(T.shape, dtype=np.int8)
    S[T>0] = 1
    return S



def order_func_balance(feats):
    """
    Order function that takes lower values on less balanced partitions.
    """

    return -np.abs(feats.sum(axis=0))

def order_func_min_side_size(feats):
    """
    Order function that takes lower values on less balanced partitions.
    """

    return np.minimum((feats>0).sum(axis=0), (feats<0).sum(axis=0))

def compound_order(order_functions: list[SetSeparationOrderFunction]):
    return lambda feats: np.vstack([o(feats) for o in order_functions])



