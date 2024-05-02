from typing import Callable, Union, Optional
import numpy as np
from tangles.separations.system import SetSeparationSystem, FeatureSystem
from tangles.search import UncrossingSweep
from tangles._typing import SetSeparationOrderFunction
import tangles.search.progress as tsp
from tangles.search._sweep import TangleSweep
from tangles.separations import SetSeparationSystemBase
from tangles import agreement_func
from tangles.search._tangle_search_interface import TangleSearchWidget


class TangleSweepFeatureSys(TangleSearchWidget):
    """A convenience object bundling a tangle sweep object and a feature system (or separation system).

    Attributes
    ----------
    sep_sys : :class:`~tangles.separations.system._set_system.SetSeparationSystem` or :class:`~tangles.separations.system._set_system.FeatureSystem`
        A feature system or separation system.
    sep_ids : np.ndarray, optional
        A list of separation ids.
    order_values : list
        Orders of the features (or separations).
    """

    def __init__(
        self,
        sep_sys: SetSeparationSystemBase,
        sep_ids: Optional[np.array] = None,
        forbidden_tuple_size: int = 3,
    ):
        self._sweep = TangleSweep(
            agreement_func(sep_sys), sep_sys.is_le, sep_ids, forbidden_tuple_size
        )
        self._sep_sys = sep_sys
        self._order_values: Optional[np.array] = None
        # hijacked design: instead of wrapping a low level engine class for the algorithm, we have to wrap a higher level class that is exposed to the user in other places...

    @property
    def sep_sys(self):
        return self._sep_sys

    @property
    def tree(self):
        return self._sweep.tree

    @property
    def search_object(self):
        return self._sweep

    @property
    def original_feature_ids(self):
        return self.tree.sep_ids

    @property
    def all_oriented_feature_ids(self):
        return self.tree.sep_ids

    def oriented_feature_ids_for_agreement(self, agreement: int):
        return self.tree.sep_ids[: self.tree.tree_height_for_agreement(agreement)]

    def tangle_matrix(
        self, min_agreement: Optional[int] = None, only_initial_seps: bool = True
    ):
        return self.tree.tangle_matrix(agreement=min_agreement)

    def lower_agreement(self, min_agreement: int, progress_callback=None):
        """
        Extend nodes in the tangle search tree until the agreement search limit has decreased below the
        specified agreement value.

        This method just forwards to :meth:`tangles.TangleSweep.sweep_below`.

        Parameter
        ---------
        min_agreement : int
            The new agreement search limit.
        """

        if min_agreement <= self.tree.limit:
            self._sweep.sweep_below(
                agreement=min_agreement, progress_callback=progress_callback
            )


def search_tangles(
    separations: Union[SetSeparationSystemBase, np.ndarray],
    min_agreement: int,
    max_number_of_seps: Union[int, None] = None,
    order: Optional[Union[list, np.ndarray, SetSeparationOrderFunction]] = None,
    progress_callback=None,
    sep_metadata: Union[list, np.ndarray] = None,
) -> TangleSweepFeatureSys:
    """
    Search tangles and return a :class:`TangleSweepFeatureSys` (a container for the result).

    Important: This function does not uncross any of the distinguishing separations and you cannot, in general, use the
    result of this function to uncross the resulting tree afterwards. Please use the function
    :func:`search_tangles_uncrossed` instead.

    Parameters
    ----------
    separations : SetSeparationSystem, FeatureSystem or np.ndarray
        A bunch of separations.

        If its type is an np.ndarray and the array contains exactly three unique values, a SetSeparationSystem is created.
        In this case, the two (possibly overlapping) sides of the separations are defined by the non-positive values and the non-negative values, respectively.

        If its type is an np.ndarray and the array contains less or more than three values, a FeatureSystem is
        created with non-positive values defining one side and positive values the other side of a separation (in each column).

    min_agreement : int
        The minimum intersection size of three separations.
    max_number_of_seps : int or None
        Maximal number of separations that should be added.
    order : list, np.ndarray or :class:`SetSeparationOrderFunction`, optional
        An object indicating the order of the separations. This order determines in which order the separations are added to the tangle search tree.

        If `order` is a list or np.ndarray, the value ``order[k]`` is the index/id of the separation appended in the k-th step.

        If `order` is a :class:`SetSeparationOrderFunction`, it is used to compute a value for each separation and the separations are appended in ascending order.
        A :class:`SetSeparationOrderFunction` is a ``Callable[[np.ndarray], np.ndarray]``.

    progress_callback : :class:`~tangles.search.progress.DefaultProgressCallback` or callable
        A callable providing a progress indication (see :class:`~tangles.search.progress.DefaultProgressCallback` for reference).
    sep_metadata: list or np.ndarray
        The metadata for separations. Only used if `separations` is of type np.ndarray.

    Returns
    -------
    :class:`TangleSweepFeatureSys`
        A tangle search object containing the result of the search and the feature system (or separation system).
    """

    sep_sys = _create_sep_sys(separations, sep_metadata)

    if max_number_of_seps is None or len(sep_sys) < max_number_of_seps:
        max_number_of_seps = len(sep_sys)

    tangle_sweep = TangleSweepFeatureSys(sep_sys)

    if order is None:
        sep_ids_to_add = range(max_number_of_seps)
    elif isinstance(order, Callable):
        if progress_callback:
            progress_callback(
                tsp.PROGRESS_TYPE_SOMETHING_STARTING, info="computing orders..."
            )
        tangle_sweep._order_values = order(sep_sys[:])
        sep_ids_to_add = tangle_sweep._order_values.argsort()
        if progress_callback:
            progress_callback(
                tsp.PROGRESS_TYPE_SOMETHING_FINISHED,
                info="computing orders... finished",
            )
    elif isinstance(order, (list, np.ndarray)):
        sep_ids_to_add = order
        tangle_sweep._order_values = order
    else:
        raise ValueError("unknown order type")

    search_obj = tangle_sweep.search_object
    if progress_callback:
        progress_callback(
            tsp.PROGRESS_TYPE_SOMETHING_STARTING, sweep=search_obj, info="appending"
        )

    for i, sep_id in enumerate(sep_ids_to_add):
        search_obj.append_separation(sep_id, min_agreement)
        if progress_callback:
            progress_callback(
                tsp.PROGRESS_TYPE_SEP_APPENDING_RUNNING,
                sweep=search_obj,
                num_total_seps=len(sep_ids_to_add),
                num_seps_added=i + 1,
            )

    if progress_callback:
        progress_callback(
            tsp.PROGRESS_TYPE_SOMETHING_FINISHED,
            sweep=search_obj,
            info="appending finished",
        )

    return tangle_sweep


def search_tangles_uncrossed(
    separations: Union[SetSeparationSystemBase, np.ndarray],
    min_agreement: int,
    order_func: SetSeparationOrderFunction,
    max_number_of_seps: Union[int, None] = None,
    progress_callback=None,
    sep_metadata: Union[list, np.ndarray] = None,
) -> UncrossingSweep:
    """
    Search tangles, uncross crossing distinguishers and return an object containing the result.

    Parameters
    ----------
    separations : SetSeparationSystem, FeatureSystem or np.ndarray
        A bunch of separations.

        If its type is an np.ndarray and the array contains exactly three unique values, a SetSeparationSystem is created.
        In this case, the two (possibly overlapping) sides of the separations are defined by the non-positive values and the non-negative values, respectively.

        If its type is an np.ndarray and the array contains less or more than three values, a FeatureSystem is
        created with non-positive values defining one side and positive values the other side of a separation (in each column).

    min_agreement : int
        The minimum intersection size of three separations.
    max_number_of_seps : int
        Maximal number of separations that should be added.
    order_func : :class:`SetSeparationOrderFunction`
        Used to compute the order of each separation. The separations are appended in ascending order.
        A :class:`SetSeparationOrderFunction` is a ``Callable[[np.ndarray], np.ndarray]``.
    progress_callback : :class:`~tangles.search.progress.DefaultProgressCallback` or callable
        A callable providing a progress indication (see :class:`~tangles.search.progress.DefaultProgressCallback` for reference).
    sep_metadata : list or np.ndarray
        The metadata for separations. Only used if `separations` is of type np.ndarray.

    Returns
    -------
    :class:`UncrossingSweep`
        A tangle search object containing the result of the search.
    """

    sep_sys = _create_sep_sys(separations, sep_metadata)
    if max_number_of_seps is None or len(sep_sys) < max_number_of_seps:
        max_number_of_seps = len(sep_sys)
    tangle_sweep = UncrossingSweep(
        sep_sys,
        order_func,
        copy_sep_sys=isinstance(separations, SetSeparationSystemBase),
    )
    tangle_sweep.append_next_features(
        agreement=min_agreement,
        number_of_seps=max_number_of_seps,
        progress_callback=progress_callback,
    )
    return tangle_sweep


def _create_sep_sys(separations, metadata=None):
    if isinstance(separations, SetSeparationSystemBase):
        sep_sys = separations
    elif isinstance(separations, np.ndarray):
        unique_values = set(np.ravel(separations))
        if (
            len(unique_values) == 3
        ):  # we assume the separations are encoded as -1/0/1 or by other three neg/0/pos values
            assert (
                any(a < 0 for a in unique_values)
                and any(a > 0 for a in unique_values)
                and any(a == 0 for a in unique_values)
            )
            sep_sys = SetSeparationSystem.with_array(separations, metadata=metadata)
        else:  # we assume the separations are encoded as 0/1, -1/1 or neg/pos
            assert any(a <= 0 for a in unique_values) and any(
                a > 0 for a in unique_values
            )
            sep_sys = FeatureSystem.with_array(separations, metadata=metadata)
    else:
        print("Sorry, cannot understand your separations. Please try again...")
        sep_sys = FeatureSystem(0)
    return sep_sys
