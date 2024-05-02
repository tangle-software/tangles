import itertools
from typing import Type, Iterable

import pytest
import numpy as np
from tangles.separations.system import (
    SetSeparationSystem,
    FeatureSystem,
    SetSeparationSystemBase,
    AdvancedFeatureSystem,
)
from tangles.util import faster_uniquerows


def generate_hyperplane_seps(n, overlap=None) -> np.ndarray:
    overlap = [0] if overlap is None else list(overlap)
    for o in overlap:
        assert 0 <= o <= n - 1

    x, y = np.meshgrid(np.arange(0, n, 1), np.arange(0, n, 1))
    z = np.zeros((n, n), dtype=np.int8)

    def sep_to_vec(a, b):
        return (z + a - b).reshape((n * n, 1))

    x_sep = np.hstack(
        [sep_to_vec(x <= j, x > j - o) for o in overlap for j in range(o, n - 1)]
    )
    y_sep = np.hstack(
        [sep_to_vec(y <= i, y > i - o) for o in overlap for i in range(o, n - 1)]
    )
    seps = np.hstack((x_sep, y_sep))
    num_seps = 2 * sum(n - 1 - o for o in overlap)
    assert seps.shape[1] == num_seps
    return seps


def run_hyperplane_test_case(
    sep_sys_type: Type[SetSeparationSystemBase],
    n: int,
    overlap: Iterable[int] | None = None,
):
    overlap = [0] if overlap is None else list(overlap)
    seps = generate_hyperplane_seps(n, overlap=overlap)
    num_seps = seps.shape[1]

    sep_sys, [sep_ids, sep_orientations] = sep_sys_type.with_array(
        seps, return_sep_info=True
    )
    assert np.all(sep_ids == np.arange(0, num_seps))
    assert np.all(sep_orientations == np.ones(num_seps))
    assert len(sep_sys) == num_seps

    assert np.all(sep_sys[sep_ids] == seps)
    assert np.all(sep_sys[1:] == seps[:, 1:])
    assert np.all(sep_sys[: num_seps - 1] == seps[:, : num_seps - 1])
    assert np.all(sep_sys[::2] == seps[:, ::2])

    e_id = sep_sys.add_seps(-np.ones(n * n))[0][0]
    assert e_id == num_seps
    assert np.all(sep_sys[e_id] == -np.ones(n * n))

    ids, oris = sep_sys.get_sep_ids(np.ones((n * n, 1), dtype=int))
    assert ids == e_id
    assert oris == -1

    ids, oris = sep_sys.get_sep_ids(seps)
    assert np.all(ids == np.arange(0, num_seps))
    assert np.all(oris == np.ones(num_seps))

    ids, oris = sep_sys.get_sep_ids(-seps)
    assert np.all(ids == np.arange(0, num_seps))
    assert np.all(oris == -np.ones(num_seps))

    ids, oris = sep_sys.add_seps(-seps)
    assert np.all(ids == np.arange(0, num_seps))
    assert np.all(oris == -np.ones(num_seps))

    y_off = num_seps // 2
    mx = (n - 1 - overlap[0]) // 2
    my = y_off + mx

    matrix = seps[:, [mx, my]]
    ids, oris = sep_sys.get_sep_ids(matrix)
    assert np.all(ids == [mx, my])
    assert np.all(oris == [1, 1])

    corners, oris = sep_sys.get_corners(mx, my)
    assert np.all(corners == np.arange(num_seps + 1, num_seps + 5))
    assert np.all(oris == np.ones(4))

    corners, oris = sep_sys.get_corners(mx + 1, mx - 1)
    assert np.all(corners == [mx + 1, num_seps + 5, e_id, mx - 1])
    assert np.all(oris == [-1, 1, 1, 1])

    def check_corners(a, b):
        ab = sep_sys[a, b]
        _m = np.hstack((-ab, ab))
        some_variable_named_c = np.minimum(_m[:, [0, 2, 0, 2]], _m[:, [1, 1, 3, 3]])
        c, o = sep_sys.get_corners(a, b)
        return np.all(some_variable_named_c == o * sep_sys[c])

    idx = list(
        enumerate(
            [(xy, i, ov) for xy in (0, 1) for ov in overlap for i in range(ov, n - 1)]
        )
    )

    for a, (xy_a, i_a, ov_a) in idx:
        for b, (xy_b, i_b, ov_b) in idx[a:]:
            assert check_corners(a, b)
            a_le_b = (xy_a == xy_b) and (i_b <= i_a) and (i_b - ov_b <= i_a - ov_a)
            b_le_a = (xy_a == xy_b) and (i_a <= i_b) and (i_a - ov_a <= i_b - ov_b)
            should_be_nested = a_le_b or b_le_a
            assert (
                sep_sys.is_nested(a, b) == should_be_nested
            ), f"{a}:{(xy_a, i_a, ov_a)} || {b}:{(xy_b, i_b, ov_b)}"
            assert (
                sep_sys.is_le(b, 1, a, 1) == a_le_b
            ), f"{a}:{(xy_a, i_a, ov_a)} <= {b}:{(xy_b, i_b, ov_b)}"
            assert (
                sep_sys.is_le(a, -1, b, -1) == a_le_b
            ), f"{a}:{(xy_a, i_a, ov_a)}^-1 >= {b}:{(xy_b, i_b, ov_b)}^-1"
            assert (
                sep_sys.is_le(a, 1, b, 1) == b_le_a
            ), f"{a}:{(xy_a, i_a, ov_a)} >= {b}:{(xy_b, i_b, ov_b)}"
            assert (
                sep_sys.is_le(b, -1, a, -1) == b_le_a
            ), f"{a}:{(xy_a, i_a, ov_a)}^-1 <= {b}:{(xy_b, i_b, ov_b)}^-1"


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
def test_sep_sys_bips(sep_sys_type: Type[SetSeparationSystemBase]):
    run_hyperplane_test_case(sep_sys_type=sep_sys_type, n=10)


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
def test_sep_sys_seps(sep_sys_type: Type[SetSeparationSystemBase]):
    run_hyperplane_test_case(sep_sys_type=sep_sys_type, n=10, overlap=[0, 2])


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
@pytest.mark.parametrize("n", [1, 10, 50])
def test_add_seps(sep_sys_type: Type[SetSeparationSystemBase], n):
    seps = 2 * np.eye(n) - 1
    seps_redundant = np.hstack((seps, -seps, seps))
    sep_sys = sep_sys_type(n)
    ids, orientations = sep_sys.add_seps(seps_redundant)
    r10 = np.arange(n)
    ones = np.ones(n)
    assert np.all(ids == np.hstack((r10, r10, r10)))
    assert np.all(orientations == np.hstack((ones, -ones, ones)))


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
def test_growing_unique_sep_sys(sep_sys_type: Type[SetSeparationSystemBase]):
    np.random.seed(9)
    seps = 2 * (np.random.random((100, 50)) > 0.5) - 1
    new_seps = 2 * (np.random.random((100, 50)) > 0.5) - 1

    sep_sys_bit = sep_sys_type.with_array(seps)

    for i in range(new_seps.shape[1] // 2):
        ids_bit, ori_bit = sep_sys_bit.add_seps(new_seps[:, 2 * i : 2 * (i + 1)])

    sel = np.array([1, 4, 7, 2, 9])
    o = np.array([-1, 1, -1, -1, 1])[np.newaxis, :]
    ids_bit, ori_bit = sep_sys_bit.add_seps(seps[:, sel] * o)

    assert (ids_bit == sel).all()
    assert (ori_bit == o).all()


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
@pytest.mark.parametrize("n", [100, 300, 500])
def test_sep_ids(sep_sys_type: Type[SetSeparationSystemBase], n):
    sep_sys = sep_sys_type(n)
    np.random.seed(100)
    seps = (2 * (np.random.random((n, 1000)) > 0.5) - 1).astype(np.int8)
    seps = faster_uniquerows(seps.T).T
    sep_ids, _ = sep_sys.add_seps(seps)

    for i in range(len(sep_ids)):
        new_id, new_ori = sep_sys.add_seps(seps[:, i])
        assert new_id == i
        assert new_ori == 1
        new_id, new_ori = sep_sys.add_seps(-seps[:, i])
        assert new_id == i
        assert new_ori == -1

        new_id, new_ori = sep_sys.add_seps(sep_sys[i])
        assert new_id == i
        assert new_ori == 1
        new_id, new_ori = sep_sys.add_seps(-sep_sys[i])
        assert new_id == i
        assert new_ori == -1

    l = 11
    r = np.arange(l)
    for i in range(seps.shape[1] // l):
        new_ids, new_oris = sep_sys.add_seps(seps[:, r])
        assert (new_ids == r).all()
        assert (new_oris == 1).all()
        new_ids, new_oris = sep_sys.add_seps(-seps[:, r])
        assert (new_ids == r).all()
        assert (new_oris == -1).all()

        new_ids, new_oris = sep_sys.add_seps(sep_sys[r])
        assert (new_ids == r).all()
        assert (new_oris == 1).all()
        new_ids, new_oris = sep_sys.add_seps(-sep_sys[r])
        assert (new_ids == r).all()
        assert (new_oris == -1).all()

        r += l


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
def test_compute_infimum(sep_sys_type: Type[SetSeparationSystemBase]):
    seps = np.zeros((100, 30))
    seps[np.random.random(seps.shape) > 0.5] = -1
    seps[np.random.random(seps.shape) > 0.5] = 1

    if sep_sys_type in {FeatureSystem}:
        seps[seps == 0] = 1

    sep_sys = sep_sys_type.with_array(seps)

    for size in range(2, 5):
        for _ in range(10):
            sel = np.random.choice(range(seps.shape[1]), size)
            for _ in range(3):
                ori = 2 * np.random.randint(0, 1, size=len(sel)) - 1
                orientations = np.array(ori)
                wanted = (seps[:, sel] * orientations[np.newaxis, :]).min(axis=1)
                from_sep_sys = sep_sys.compute_infimum(sel, ori)
                assert (wanted == from_sep_sys).all()


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
def test_metadata(sep_sys_type: Type[SetSeparationSystemBase]):
    seps = 2 * np.eye(100) - 1
    sep_sys = sep_sys_type.with_array(
        seps, metadata=[f"sep{i}" for i in range(seps.shape[1])]
    )
    for i in range(len(sep_sys)):
        metadata = sep_sys.separation_metadata(i)
        assert metadata.info == f"sep{i}"
        assert metadata.next is None

    sep_sys.add_seps(np.ones(seps.shape[1]))
    assert sep_sys.separation_metadata(len(sep_sys) - 1).info is None

    sep_sys.add_seps(seps[:, 5], "sep5_1")
    metadata = sep_sys.separation_metadata(5)
    assert metadata.info == "sep5"
    assert metadata.next is not None
    metadata = metadata.next
    assert metadata.info == "sep5_1"

    sep_sys.add_seps(-seps[:, 10], "-sep10_1")
    metadata = sep_sys.separation_metadata(10)
    assert metadata.info == "sep10"
    assert metadata.type == "custom"
    assert metadata.orientation == 1
    assert metadata.next is not None
    metadata = metadata.next
    assert metadata.info == "-sep10_1"
    assert metadata.type == "custom"
    assert metadata.orientation == -1


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
def test_corner_metadata(sep_sys_type: Type[SetSeparationSystemBase]):
    np.random.seed(0)
    seps = (2 * (np.random.random((20, 6)) > 0.5) - 1).astype(np.int8)
    sep_sys = sep_sys_type.with_array(
        seps, metadata=[f"sep{i}" for i in range(seps.shape[1])]
    )

    # compute corners and corners of corners...
    for _ in range(2):
        for i, j in itertools.combinations(sep_sys.all_sep_ids(), 2):
            sep_sys.get_corners(i, j)

    # test if the information given by the metadata is correct for every sep
    for sep_id in sep_sys.all_sep_ids():
        for metadata in sep_sys.separation_metadata(sep_id).tail_as_gen():
            if metadata.type != "inf":
                continue
            inf = metadata.orientation * np.minimum(
                sep_sys[metadata.info[0][0]] * metadata.info[0][1],
                sep_sys[metadata.info[1][0]] * metadata.info[1][1],
            )
            assert (inf == sep_sys[sep_id]).all()


@pytest.mark.parametrize(
    "sep_sys_type",
    [
        pytest.param(SetSeparationSystem, id="bitarray"),
        pytest.param(FeatureSystem, id="bitarray bips"),
        pytest.param(AdvancedFeatureSystem, id="advanced"),
    ],
)
def test_assemble_metadata(sep_sys_type: Type[SetSeparationSystemBase]):
    np.random.seed(0)
    seps = (2 * (np.random.random((20, 6)) > 0.5) - 1).astype(np.int8)
    sep_sys = sep_sys_type.with_array(
        seps, metadata=[f"sep{i}" for i in range(seps.shape[1])]
    )

    # compute corners and corners of corners...
    for _ in range(2):
        for i, j in itertools.combinations(sep_sys.all_sep_ids(), 2):
            sep_sys.get_corners(i, j)

    def _sep_from_meta_tuple(meta_tuple, sep_sys):
        ori, sep_info = meta_tuple
        if isinstance(sep_info, tuple):
            (ori_1, sep_1), (ori_2, sep_2) = sep_info
            return np.minimum(
                ori_1 * _sep_from_meta_tuple(sep_1, sep_sys),
                ori_2 * _sep_from_meta_tuple(sep_2, sep_sys),
            )
        return ori * sep_sys[int(sep_info[3:])]

    for sep_id in sep_sys.all_sep_ids():
        meta = sep_sys.assemble_meta_info(sep_id)
        assert (_sep_from_meta_tuple(meta, sep_sys) == sep_sys[sep_id]).all()
