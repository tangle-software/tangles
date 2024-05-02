from itertools import combinations
from typing import Union, Tuple, Optional
import numpy as np
import math
from tangles.separations import (
    AdvancedFeatureSystem,
    SetSeparationSystemBase,
)
from tangles._typing import SetSeparationOrderFunction


def compute_corner_features(
    features: np.ndarray,
    min_side_size: float = 0,
    min_side_size_is_fraction: bool = False,
    order_func: SetSeparationOrderFunction = None,
    max_order_factor: float = 1.0,
    global_max_order: float = math.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute corners of features/separations given as numpy array of oriented indicator vectors (i.e. -1/1-vectors)


    Parameters
    ----------
    features : np.ndarray
        A bunch of features.
    min_side_size : float
        Discard corners with minimum size smaller than `min_side_size` (or `min_side_size*datasize` if `min_side_size_is_fraction` is True).
        This parameter can be used to control the *balance*.
    min_side_size_is_fraction : bool
        If True, the parameter `min_side_size` is interpreted as a fraction of the data set size.
    order_func : SetSeparationOrderFunction or None
        an order function
    max_order_factor: float
        Discard corners of two features with an order greater than `max_order_factor` times the greater order of the two features.
    global_max_order: float
        Discard corners that have an order greater than this value

    Returns
    -------
    Tuple[np.ndarray, list]:
        an array containing the corners and a list of the same length containing tuples `((index1, orientation1), (index2, orientation2))` which of the original features
        went into each of the resulting corners.
    """

    if not isinstance(features, np.ndarray):
        raise ValueError("cannot understand your features")

    features = AdvancedFeatureSystem.with_array(
        features
    )  # misuse advanced features class for hashing functionality ;-)
    new_feature_range = append_corner_features(
        features,
        min_side_size=min_side_size,
        min_side_size_is_fraction=min_side_size_is_fraction,
        order_func=order_func,
        max_order_factor=max_order_factor,
        global_max_order=global_max_order,
    )
    corners = features[new_feature_range.start :]
    meta = np.empty(len(new_feature_range), dtype=object)
    meta[:] = [m.info for m in features.feature_metadata(new_feature_range)]
    return corners, meta


def append_corner_features(
    feature_system: SetSeparationSystemBase,
    feature_ids: Union[range, list, np.ndarray, None] = None,
    min_side_size: float = 0,
    min_side_size_is_fraction: bool = False,
    order_func: SetSeparationOrderFunction = None,
    max_order_factor: float = 1.0,
    global_max_order: float = math.inf,
) -> range:
    """
    Compute corners and add them to a FeatureSystem or SetSeparationSystem

    Parameters
    ----------
    feature_system : SetSeparationSystemBase, FeatureSystem, AdvancedFeatureSystem or SetSeparationSystem
        A feature system or separation system
    feature_ids : list, np.ndarray, range or None
        a selection of a subset of `feature_system`
    min_size_fraction : float
        Discard corners with minimum size smaller than `min_size_fraction * dataset_size`. This parameter can be used to control the *balance*.
    order_func : SetSeparationOrderFunction or None
        an order function
    max_order_factor: float
        Discard corners of two features with an order greater than `max_order_factor` times the greater order of the two features.
    global_max_order: float
        Discard corners that have an order greater than this value

    Returns
    -------
    range:
        the id range of the added features
    """

    if not isinstance(feature_system, SetSeparationSystemBase):
        raise ValueError("cannot understand your feature system")

    first_corner_id = len(feature_system)
    if feature_ids is None:
        feature_ids = range(len(feature_system))
    if min_side_size_is_fraction:
        min_side_size = feature_system.datasize * min_side_size
    for i, j in combinations(feature_ids, 2):
        ori = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        # TODO: could be optimized if we don't need the array representation in case the order function is None...
        corners = np.c_[
            feature_system.compute_infimum([i, j], ori[0]),
            feature_system.compute_infimum([i, j], ori[1]),
            feature_system.compute_infimum([i, j], ori[2]),
            feature_system.compute_infimum([i, j], ori[3]),
        ]
        corners, ori = (
            corners[:, (sel := feature_system.get_sep_ids(corners)[0] == -1)],
            ori[sel, :],
        )
        if corners.shape[1] > 0 and min_side_size > 0:
            corners, ori = (
                corners[
                    :,
                    (
                        sel := np.minimum(
                            (side_sizes := (corners > 0).sum(axis=0)),
                            feature_system.datasize - side_sizes,
                        )
                        >= min_side_size
                    ),
                ],
                ori[sel, :],
            )
        if corners.shape[1] > 0 and order_func is not None:
            corners, ori = (
                corners[
                    :,
                    (
                        sel := order_func(corners)
                        <= min(
                            max_order_factor * max(order_func(feature_system[(i, j)])),
                            global_max_order,
                        )
                    ),
                ],
                ori[sel, :],
            )
        feature_system.add_seps(corners, [((i, o_i), (j, o_j)) for o_i, o_j in ori])

    return range(first_corner_id, len(feature_system))
