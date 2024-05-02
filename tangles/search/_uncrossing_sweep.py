import copy
from typing import Optional
import numpy as np

from tangles.separations.system import (
    SetSeparationSystemBase,
    SetSeparationSystemOrderFunc,
)
from tangles.agreement import agreement_func
from tangles.util.tree import BinTreeNode
from tangles._typing import SetSeparationOrderFunction
from ._sweep import TangleSweep
from ._uncrossing import EfficientDistinguisherUncrossing
from ._tree import TangleSearchTree
from ._tree_of_tangles import TreeOfTangles, create_tot
from ._tangle_search_interface import TangleSearchWidget
from .progress import (
    PROGRESS_TYPE_SOMETHING_STARTING,
    PROGRESS_TYPE_SEP_APPENDING_RUNNING,
    PROGRESS_TYPE_SOMETHING_FINISHED,
)


class UncrossingSweep(TangleSearchWidget):
    """
    A :class:`TangleSweep` wrapped with uncrossing functionality.

    Whenever tangles are searched, the :class:`UncrossingSweep` ensures that the efficient distinguishers in the
    :class:`~tangles.search._tree.TangleSearchTree` are uncrossed. This is a necessary condition to create a tangle
    search tree.

    Additionally, objects of this class hold an ordered separation system, so also some convenient separation system
    functionality is provided.

    Parameters
    ----------
    sep_sys : :class:`SetSeparationSystemBase`
        The separation system.
    order_func : :class:`SetSeparationOrderFunction`
        Used to compute the order of each separation. The separations are appended in ascending order.
        A :class:`SetSeparationOrderFunction` is a ``Callable[[np.ndarray], np.ndarray]``.
    forbidden_tuple_size : int
        The maximum size of forbidden tuples. The standard tangles use a maximum size of 3 (i.e. forbidden triples).
    copy_sep_sys : bool
        Whether the separation system `sep_sys` should be copied. If True, this ensures that the given separation
        system is left unchanged by the :class:`UncrossingSweep`.
    """

    def __init__(
        self,
        sep_sys: SetSeparationSystemBase,
        order_func: SetSeparationOrderFunction,
        forbidden_tuple_size: int = 3,
        copy_sep_sys: bool = True,
    ):
        if copy_sep_sys:
            sep_sys = copy.deepcopy(sep_sys)
        self._sep_sys_ord = SetSeparationSystemOrderFunc(sep_sys, order_func)
        self._initial_sep_ids = self._sep_sys_ord.sorted_ids.copy()
        self._number_of_initial_seps_added = 0
        self._sweep = TangleSweep(
            agreement_func=agreement_func(sep_sys),
            le_func=sep_sys.is_le,
            forbidden_tuple_size=forbidden_tuple_size,
        )
        self._uncrossing = EfficientDistinguisherUncrossing(
            self._sweep, self._sep_sys_ord, agreement=sep_sys.datasize
        )
        self._max_num_uncrossing_steps_in_lower_agreement = (
            np.inf
        )  # it's the best we have: uncrossing less often is slower...

    @property
    def sep_sys(self):
        return self._sep_sys_ord.sep_sys

    @property
    def all_oriented_feature_ids(self):
        return self.tree.sep_ids

    @property
    def original_feature_ids(self):
        return self._initial_sep_ids[: self._number_of_initial_seps_added]

    def oriented_feature_ids_for_agreement(self, agreement: int):
        return self.tree.sep_ids[: self.tree.tree_height_for_agreement(agreement)]

    def order_of_feature(
        self, level_idx: Optional[int] = None, feat_id: Optional[int] = None
    ):
        if (level_idx is None) == (feat_id is None):
            raise ValueError(
                "We need either an index or an id, neither none, nor both."
            )
        if feat_id is None:
            feat_id = self._sep_sys_ord.sorted_ids[level_idx]
        return self._sep_sys_ord.get_order(feat_id)

    def set_uncrossing_strategy_only_most_balanced_corner(self, only_balanced: bool):
        self._uncrossing._choose_most_balanced_corner = only_balanced

    def initialize(
        self,
        agreement: Optional[int] = None,
        number_of_seps=None,
        progress_callback=None,
    ):
        self.append_next_features(agreement, number_of_seps, progress_callback)

    def append_next_features(
        self, agreement=None, number_of_seps=None, progress_callback=None
    ) -> int:
        if agreement is None:
            if self._number_of_initial_seps_added == 0:
                self._uncrossing._agreement = self._sep_sys_ord.sep_sys.datasize // 5
        else:
            self._uncrossing._agreement = agreement

        if (
            number_of_seps is None
            or number_of_seps
            >= len(self._initial_sep_ids) - self._number_of_initial_seps_added
        ):
            number_of_seps = (
                len(self._initial_sep_ids) - self._number_of_initial_seps_added
            )

        sep_ids_to_add = self._initial_sep_ids[
            self._number_of_initial_seps_added : self._number_of_initial_seps_added
            + number_of_seps
        ]
        if progress_callback:
            progress_callback(
                PROGRESS_TYPE_SOMETHING_STARTING,
                info="appending",
                sweep=self._sweep,
            )

        for i, sep_id in enumerate(sep_ids_to_add):
            self._sweep.append_separation(sep_id, self._uncrossing._agreement)
            if progress_callback:
                progress_callback(
                    PROGRESS_TYPE_SEP_APPENDING_RUNNING,
                    sweep=self._sweep,
                    num_total_seps=len(sep_ids_to_add),
                    num_seps_added=i + 1,
                )
            self._uncrossing.add_known_ids([sep_id])
            self._uncrossing.uncross(progress_callback)

        self._number_of_initial_seps_added += number_of_seps

        if progress_callback:
            progress_callback(
                PROGRESS_TYPE_SOMETHING_FINISHED,
                info="appending finished",
                sweep=self._sweep,
            )

        return self._number_of_initial_seps_added

    def sweep_one(self, progress_callback=None) -> int:
        """Extends nodes in the tree until the agreement search limit has decreased.
        The resulting tangle search tree is uncrossed.

        Returns
        -------
        int
            The new, decreased, agreement search limit.
        """

        return self.sweep_below(self._sweep.tree.limit, progress_callback)

    def sweep_below(self, agreement: int, progress_callback=None) -> int:
        """Extends nodes in the tree until the agreement search limit has decreased below the specified agreement value.
        The resulting tangle search tree is uncrossed.

        Parameters
        ----------
        agreement : int
            The value below which the agreement search limit should fall.
        progress_callback: TangleSearchProgressType
            a progress callback

        Returns
        -------
        int
            The new agreement search limit.
        """

        self._sweep.sweep_below(agreement, progress_callback=progress_callback)
        self._uncrossing._agreement = agreement
        self._uncrossing.uncross(progress_callback=progress_callback)
        return self._sweep.tree.limit

    def sweep_stepwise(
        self, agreement: int, step_size=1, sweep_progress_callback=None
    ) -> int:
        if sweep_progress_callback:
            sweep_progress_callback(
                PROGRESS_TYPE_SOMETHING_STARTING,
                info="sweep (stepwise)",
                sweep=self._sweep,
            )
        limit = self.tree.limit
        while limit >= agreement:
            limit = self.sweep_below(
                max(self._sweep.tree.limit - (step_size - 1), agreement),
                sweep_progress_callback,
            )
        if sweep_progress_callback:
            sweep_progress_callback(
                PROGRESS_TYPE_SOMETHING_FINISHED,
                info="sweep (stepwise)",
                sweep=self._sweep,
            )
        return limit

    def lower_agreement(self, min_agreement: int, progress_callback=None):
        sweep_step_size = max(
            1,
            np.floor(
                (self.tree.limit + 1 - min_agreement)
                / self._max_num_uncrossing_steps_in_lower_agreement
            ),
        )
        self.sweep_stepwise(min_agreement, sweep_step_size, progress_callback)

    @property
    def tree(self) -> TangleSearchTree:
        """The tangle search tree on which this TangleSweep operates."""

        return self._sweep.tree

    @property
    def search_object(self):
        return self._sweep

    @property
    def sep_orders(self):
        return self._sep_sys_ord.get_order(self.tree.sep_ids)

    def tangle_matrix(
        self, min_agreement: Optional[int] = None, only_initial_seps: bool = True
    ):
        r"""
        Returns a matrix containing all the found tangles in rows.

        Every row of the returned matrix is a :math:`\{-1,1\}`-orientation-vector, every column corresponds to a separation.
        The columns are ordered in the same way the corresponding separations appear in the tree.

        Parameters
        ----------
        min_agreement:
            All tangles of at least this agreement value are returned. If None, the search tree's limit is used.
        only_initial_seps : bool
            If False, the resulting matrix contains one column for every separation that was oriented, including all the corners introduced by uncrossing.
            If True, only columns corresponding to the explicitly appended separations are included.

        Returns
        -------
        tangle_mat :  numpy.ndarray,
            A numpy :math:`(m,k)`-matrix with entries in :math:`\{-1,1\}` containing one row for each of the :math:`m`
            tangles and a column for each of the :math:`k` oriented separations.
        """

        if min_agreement is None:
            min_agreement = self.tree.limit

        mat = self.tree.tangle_matrix(agreement=min_agreement)
        if only_initial_seps:
            mat = mat[
                :, np.isin(self.tree.sep_ids[: mat.shape[1]], self.original_feature_ids)
            ]

        return mat

    def create_tot(self, min_agreement, max_level=None, id_at_max_level=None):
        """
        Create a tree of tangles.

        Note:
        You should choose a min_agreement value which is not smaller than the limit of the tree of the sweep attribute.
        Furthermore you should not set a value for both the max_level and the id_at_max_level.

        Parameters
        ----------
        min_agreement
            Tangles which appear in the tree of tangles will have at least this agreement value.
        max_level
            Only consider tangles up to the max level. Optional.
        id_at_max_level
            Only consider tangles which do not orient separations below the separation with given id. Optional.

        Returns
        -------
        TreeOfTangle
            The tree of tangles containing the maximal tangles.

        """
        assert min_agreement >= self._sweep.tree.limit
        assert ((max_level is None) and (id_at_max_level is None)) or (
            (max_level is None) != (id_at_max_level is None)
        )

        if id_at_max_level is not None:
            max_level = (self._sweep.tree.sep_ids == id_at_max_level).argmax()
            if max_level == 0:
                assert (
                    self._sweep.tree.sep_ids[0] == id_at_max_level
                )  # bail out, if id is not found

        levels, sep_ids = self._sweep.tree.get_efficient_distinguishers(
            agreement=min_agreement, max_level=max_level
        )
        if levels.shape[0] == 0:
            levels = np.array([0])
            sep_ids = self._sweep.tree.sep_ids[[0]]
        tangle_mat = self._sweep.tree.tangle_matrix(
            agreement=min_agreement, max_level=max_level
        )
        tangles_reduced = tangle_mat[:, levels]
        bin_tree = BinTreeNode.from_indicator_matrix(tangles_reduced)
        some_node = create_tot(
            bin_tree, sep_ids, 0, tangles_reduced, self.sep_sys.is_le
        )
        tot = TreeOfTangles(sep_ids=sep_ids, nodes=some_node.all_nodes())
        tot.tangle_matrix = tangle_mat
        return tot
