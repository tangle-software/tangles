import numpy as np

def faster_uniquerows(mat: np.ndarray, return_counts: bool= False, return_index: bool = False, return_inv: bool = False) -> tuple[np.ndarray, ...]:
    if len(mat.shape)<2:
        mat = mat[:,np.newaxis]
    sorted_idcs = np.lexsort(mat.T)
    mat_sorted = mat[sorted_idcs,:]

    diff = (mat_sorted[1:, :] != mat_sorted[:-1, :]).any(axis=1)
    uni_indicator = np.empty(len(diff)+1, dtype=bool)
    uni_indicator[0] = True
    uni_indicator[1:] = diff

    ret = (mat_sorted[uni_indicator, :],)
    if return_counts:
        unique_idcs = np.flatnonzero(uni_indicator)
        counts = np.empty(unique_idcs.shape[0], dtype=int)
        counts[:-1] = unique_idcs[1:]-unique_idcs[:-1]
        counts[-1] = uni_indicator.shape[0]-unique_idcs[-1]
        ret += (counts,)
    if return_index:
        ret += (sorted_idcs[uni_indicator],)
    if return_inv:
        ret += (uni_indicator.cumsum()[np.argsort(sorted_idcs)] - 1,)
    return ret if len(ret)>1 else ret[0]


def unique_rows(mat: np.ndarray):
    _, idcs = faster_uniquerows(mat, return_index=True)
    idcs.sort()
    return mat[idcs,:]

def unique_cols(mat: np.ndarray):
    _, idcs = faster_uniquerows(mat.T, return_index=True)
    idcs.sort()
    return mat[:,idcs]