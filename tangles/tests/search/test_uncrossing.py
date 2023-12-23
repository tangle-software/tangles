import time

import numpy as np
import pytest
from scipy.spatial import distance_matrix

from tangles.util import faster_uniquerows
from tangles.separations import SetSeparationSystem, SetSeparationSystemOrderFunc
from tangles import agreement_func, uncross_distinguishers, TangleSweep


def create_test_data(centers, scale, counts):
    if type(counts) == int:
        counts = [counts] * centers.shape[0]
    M = np.empty((np.sum(counts), centers.shape[1]))
    idx = 0
    for c, s, n in zip(centers, scale, counts):
        M[idx:idx + n, :] = c + np.random.normal(0, s, n * c.shape[0]).reshape(n, c.shape[0])
        idx += n
    return M


def sampled_pairs_hyperplane_seps(M, num_seps):
    idcs = np.random.randint(M.shape[0], size=(num_seps, 2))
    equal = np.flatnonzero(idcs[:, 0] == idcs[:, 1])
    while equal.shape[0] > 0:  # not the nicest way to do this...
        idcs[equal, 1] = np.random.randint(M.shape[0], size=equal.shape[0])
        equal = equal[np.flatnonzero(idcs[equal, 0] == idcs[equal, 1])]

    d = M[idcs[:, 0]] - M[idcs[:, 1]]
    l = np.sqrt(np.square(d).sum(axis=1))[:, np.newaxis]
    l[l == 0] = 1
    d /= l
    p = 0.5 * (M[idcs[:, 0]] + M[idcs[:, 1]])

    B = ((M[:, :, np.newaxis] - p[:, :, np.newaxis].T) * d[:, :, np.newaxis].T).sum(axis=1) > 0
    S = -np.ones(B.shape, dtype=np.int8)
    S[B] = 1
    return S, p, d


def uncross_distinguishers_hyperplanes_test(seed=29292):
    np.set_printoptions(linewidth=10000)
    np.random.seed(seed)
    M = create_test_data(np.random.random((5, 3)) * 20, [1, 3, 1, 1, 3], 100)
    S, _, _ = sampled_pairs_hyperplane_seps(M, 10000)
    S *= S[0:1, :]
    S = faster_uniquerows(S.T).T

    dist = distance_matrix(M, M)
    d_max = 5
    dist_clamped = dist.copy()
    dist_clamped[dist_clamped > d_max] = d_max
    sim = 1 - dist_clamped / d_max

    def order_func(S_):
        return -(S_ * (sim @ S_)).sum(axis=0)

    sep_sys = SetSeparationSystem.with_array(S)
    sep_sys_ord = SetSeparationSystemOrderFunc(sep_sys, order_func)
    search = TangleSweep(agreement_func(sep_sys), le_func=sep_sys.is_le)

    n_seps = min(S.shape[1], 1000)
    sep_ids_1 = sep_sys_ord.sorted_ids[:n_seps].copy()
    t = time.time()
    for i, id in enumerate(sep_ids_1):
        if search.append_separation(id, 2) == 0:
            print("cannot append!")
            break

    t = time.time()
    disting_levels, disting_sep_ids = uncross_distinguishers(search, sep_sys_ord, 2, verbose=False)
    print(f"tot took: {time.time() - t}")

    cross1, cross2 = sep_sys.find_first_cross(disting_sep_ids)
    assert (cross1 is None)

    # do they distinguish all tangles?
    tangles = search.tree.tangle_matrix(2)
    assert (faster_uniquerows(tangles[:, disting_levels]).shape[0] == tangles.shape[0])

    print(f"Result: {disting_sep_ids}")

    # let's use a different order function with the same sep_sys
    sim2 = np.exp(-np.square(dist) / 5)

    def order_func2(S_):
        return -(S_ * (sim2 @ S_)).sum(axis=0)

    sep_sys_ord2 = SetSeparationSystemOrderFunc(sep_sys, order_func2)
    search2 = TangleSweep(agreement_func(sep_sys), le_func=sep_sys.is_le)

    sep_ids_2 = sep_sys_ord2.sorted_ids[:n_seps].copy()
    t = time.time()
    for i, id in enumerate(sep_ids_2):
        if search2.append_separation(id, 2) == 0:
            print("cannot append!")
            break

    t = time.time()
    disting_levels, disting_sep_ids = uncross_distinguishers(search2, sep_sys_ord2, 2)
    print(f"tot took: {time.time() - t}")

    cross1, cross2 = sep_sys.find_first_cross(disting_sep_ids)
    assert (cross1 is None)

    tangles2 = search2.tree.tangle_matrix(2)
    assert (faster_uniquerows(tangles2[:, disting_levels]).shape[0] == tangles2.shape[0])

    print(f"Result: {disting_sep_ids}")


@pytest.mark.long
@pytest.mark.skip(reason="Skipping long tests by default.")
def test_uncross_distinguishers_hyperplanes():
    n_tests = 1
    for i in range(n_tests):
        print(f"Test [{i}] ...", end="")
        uncross_distinguishers_hyperplanes_test(seed=71623 + i * 9)
        print("finished!")