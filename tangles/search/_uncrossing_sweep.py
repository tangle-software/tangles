import copy
import numpy as np

from ._sweep import TangleSweep
from tangles._typing import SetSeparationOrderFunction
from typing import Optional
from ._uncrossing import EfficientDistinguisherUncrossing
from tangles.separations.system import SetSeparationSystemBase, SetSeparationSystemOrderFunc
from tangles.agreement import agreement_func
from ._tree import TangleSearchTree
from tangles.util.tree import BinTreeNode
from ._tree_of_tangles import TreeOfTangles, create_tot

import tangles.search.progress as tsp



class UncrossingSweep:
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

    def __init__(self,
                 sep_sys: SetSeparationSystemBase,
                 order_func: SetSeparationOrderFunction,
                 forbidden_tuple_size: int = 3,
                 copy_sep_sys: bool = True):
        if copy_sep_sys:
            sep_sys = copy.deepcopy(sep_sys)
        self.sep_sys_ord = SetSeparationSystemOrderFunc(sep_sys, order_func)
        self.initial_sep_ids = self.sep_sys_ord.sorted_ids.copy()
        self.number_of_initial_seps_added = 0
        self.sweep = TangleSweep(agreement_func=agreement_func(sep_sys), le_func=sep_sys.is_le, forbidden_tuple_size=forbidden_tuple_size)
        self.uncrossing = EfficientDistinguisherUncrossing(self.sweep, self.sep_sys_ord, agreement=sep_sys.datasize)
        self.max_num_uncrossing_steps_in_lower_agreement = np.inf   # it's the best we have: uncrossing less often is slower...

    @property
    def sep_sys(self):
        return self.sep_sys_ord.sep_sys

    @property
    def all_oriented_sep_ids(self):
        return self.tree.sep_ids

    @property
    def original_sep_ids(self):
        return self.initial_sep_ids[:self.number_of_initial_seps_added]

    def oriented_sep_ids_for_agreement(self, agreement: int):
        return self.tree.sep_ids[:self.tree.tree_height_for_agreement(agreement)]

    def order_of_feature(self, level_idx: Optional[int] = None, feat_id: Optional[int] = None):
        if (level_idx is None) == (feat_id is None):
            raise ValueError("We need either an index or an id, neither none, nor both.")
        if feat_id is None:
            feat_id = self.sep_sys_ord.sorted_ids(level_idx)
        return self.sep_sys_ord.get_order(feat_id)

    def initialize(self, agreement: Optional[int] = None, number_of_seps=None, progress_callback=None):
        self.append_next_separations(agreement, number_of_seps, progress_callback)

    def append_next_separations(self, agreement=None, number_of_seps=None, progress_callback=None) -> int:
        if agreement is None:
            if self.number_of_initial_seps_added == 0:
                self.uncrossing._agreement = self.sep_sys_ord.sep_sys.datasize//5
        else:
            self.uncrossing._agreement = agreement

        if number_of_seps is None or number_of_seps >= len(self.initial_sep_ids) - self.number_of_initial_seps_added:
            number_of_seps = len(self.initial_sep_ids) - self.number_of_initial_seps_added

        sep_ids_to_add = self.initial_sep_ids[self.number_of_initial_seps_added:self.number_of_initial_seps_added + number_of_seps]
        if progress_callback:
            progress_callback(tsp.PROGRESS_TYPE_SOMETHING_STARTING, info="appending", sweep=self.sweep)

        for i, sep_id in enumerate(sep_ids_to_add):
            self.sweep.append_separation(sep_id, self.uncrossing._agreement)
            if progress_callback:
                progress_callback(tsp.PROGRESS_TYPE_SEP_APPENDING_RUNNING, sweep=self.sweep, num_total_seps=len(sep_ids_to_add), num_seps_added=i+1)
            self.uncrossing.add_known_ids([sep_id])
            self.uncrossing.uncross(progress_callback)

        self.number_of_initial_seps_added += number_of_seps

        if progress_callback:
            progress_callback(tsp.PROGRESS_TYPE_SOMETHING_FINISHED, info="appending finished", sweep=self.sweep)

        return self.number_of_initial_seps_added

    def sweep_one(self, progress_callback=None) -> int:
        """Extends nodes in the tree until the agreement search limit has decreased.
        The resulting tangle search tree is uncrossed.

        Returns
        -------
        int
            The new, decreased, agreement search limit.
        """

        return self.sweep_below(self.sweep.tree.limit, progress_callback)

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

        self.sweep.sweep_below(agreement, progress_callback=progress_callback)
        self.uncrossing._agreement = agreement
        self.uncrossing.uncross(progress_callback=progress_callback)
        return self.sweep.tree.limit

    def sweep_stepwise(self, agreement: int, step_size=1, sweep_progress_callback=None) -> int:
        if sweep_progress_callback:
            sweep_progress_callback(tsp.PROGRESS_TYPE_SOMETHING_STARTING, info="sweep (one by one)", sweep=self.sweep)
        limit = self.tree.limit
        while limit >= agreement:
            limit = self.sweep_below(max(self.sweep.tree.limit - (step_size-1), agreement), sweep_progress_callback)
        if sweep_progress_callback:
            sweep_progress_callback(tsp.PROGRESS_TYPE_SOMETHING_FINISHED, info="sweep (one by one)", sweep = self.sweep)
        return limit


    def lower_agreement(self, min_agreement: int, progress_callback=None):
        sweep_step_size = max(1,np.floor((self.tree.limit+1 - min_agreement)/self.max_num_uncrossing_steps_in_lower_agreement))
        self.sweep_stepwise(min_agreement, sweep_step_size, progress_callback)



    @property
    def tree(self) -> TangleSearchTree:
        """The tangle search tree on which this TangleSweep operates."""

        return self.sweep.tree

    @property
    def sep_orders(self):
        return self.sep_sys_ord.get_order(self.tree.sep_ids)

    def tangle_matrix(self, min_agreement: Optional[int] = None, only_original_seps:bool = True):
        """
        Returns a matrix containing all the found tangles in rows.
        
        Every row of the returned matrix is a :math:`\{-1,1\}`-orientation-vector, every column corresponds to a separation.
        The columns are ordered in the same way the corresponding separations appear in the tree.

        Parameters
        ----------
        only_original_seps : bool
            If False, the resulting matrix contains one column for every separation that was oriented, including all the corners introduced by uncrossing. 
            If True, only columns corresponding to the explicitly appended separations are included.
        min_agreement:
            All tangles of at least this agreement value are returned. If None, the search tree's limit is used.

        Returns
        -------
        tangle_mat :  numpy.ndarray,
            A numpy :math:`(m,k)`-matrix with entries in :math:`\{-1,1\}` containing one row for each of the :math:`m`
            tangles and a column for each of the :math:`k` oriented separations.
        """

        if min_agreement is None:
            min_agreement = self.tree.limit

        mat = self.tree.tangle_matrix(agreement=min_agreement)
        if only_original_seps:
            mat = mat[:, np.isin(self.tree.sep_ids[:mat.shape[1]],self.original_sep_ids)]

        return mat

    def create_tot(self, min_agreement, max_level=None, id_at_max_level=None):
        assert min_agreement >= self.sweep.tree.limit
        assert ((max_level is None) and (id_at_max_level is None)) or ((max_level is None) != (id_at_max_level is None))

        if id_at_max_level is not None:
            max_level = (self.sweep.tree.sep_ids == id_at_max_level).argmax()
            if max_level == 0:
                assert self.sweep.tree.sep_ids[0] == id_at_max_level  # bail out, if id is not found


        levels, sep_ids = self.sweep.tree.get_efficient_distinguishers(agreement=min_agreement, max_level=max_level)
        if levels.shape[0] == 0:
            levels = np.array([0])
            sep_ids = self.sweep.tree.sep_ids[[0]]
        tangle_mat = self.sweep.tree.tangle_matrix(agreement=min_agreement, max_level=max_level)
        tangles_reduced = tangle_mat[:, levels]
        bin_tree = BinTreeNode.from_indicator_matrix(tangles_reduced)
        some_node = create_tot(bin_tree, sep_ids, 0, tangles_reduced, self.sep_sys.is_le)
        tot = TreeOfTangles(sep_ids=sep_ids, nodes=some_node.all_nodes())
        tot.tangle_matrix = tangle_mat
        return tot
