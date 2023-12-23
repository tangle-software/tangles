from typing import Optional
import heapq
import numpy as np
from tangles._typing import LessOrEqFunc, AgreementFunc
from .extending_tangles import ExtendingTangles
from ._tree import TangleSearchTree
from tangles import Tangle

import tangles.search.progress as tsp


def default_sweep_progress_callback(sweep, level_idx):
    print(f"\rSweeping: {level_idx+1}/{sweep.tree.number_of_separations} finished", end="      ")


class TangleSweep:
    """
    Search object which builds and extends the tangle search tree.

    Parameters
    ----------
    agreement_func : AgreementFunc
        A function which takes a list of separations and returns a non-negative, numerical agreement value.
    le_func : LessOrEqFunc
        A partial order on the oriented separations.
    sep_ids : np.array, optional
        The separation ids. Can be used to start the algorithm with some empty levels.
    forbidden_tuple_size : int
        The maximum size of forbidden tuples. The standard tangles use a maximum size of 3 (i.e. forbidden triples).
    """

    def __init__(self,
                 agreement_func: AgreementFunc,
                 le_func: LessOrEqFunc,
                 sep_ids: Optional[np.array] = None,
                 forbidden_tuple_size: int = 3):
        
        self._search_tree = TangleSearchTree(Tangle(agreement_func.max_value), sep_ids if sep_ids is not None else np.empty(0, dtype=int))
        self._algorithm = ExtendingTangles(agreement_func, le_func, forbidden_tuple_size=forbidden_tuple_size)

    def sweep_one(self, progress_callback=None) -> int:
        """Extend nodes in the tree until the agreement search limit has decreased.

        Returns
        -------
        int
            The new, decreased, agreement search limit.
        """

        return self.sweep_below(self._search_tree.limit, progress_callback)

    def sweep_below(self, agreement: int, progress_callback=None) -> int:
        """Extend nodes in the tree until the agreement search limit has decreased below the specified agreement value.

        Parameters
        ----------
        agreement : int
            The value below which the agreement search limit should fall.

        Returns
        -------
        int
            The new agreement search limit.
        """

        if progress_callback:
            progress_callback(tsp.PROGRESS_TYPE_SOMETHING_STARTING, info="sweep", sweep=self)
        for i, level in self._search_tree._levels(self._search_tree.root, agreement, self.tree.number_of_separations):
            level_extend = [node for node in level if node.is_leaf()]
            self._algorithm.extend_tangles(level_extend, self._search_tree.sep_ids[i], np.empty(0))
            if progress_callback:
                progress_callback(tsp.PROGRESS_TYPE_SWEEP_RUNNING, sweep=self, level=i)
        if progress_callback:
            progress_callback(tsp.PROGRESS_TYPE_SOMETHING_FINISHED, info="sweep finished", sweep=self)
        return self._search_tree.limit

    def greedy_search(self, max_width: int, explore_agreement_lower_bound: int = 1, max_depth:Optional[int] = None, start_node: Optional[Tangle] = None):
        """Greedily search for tangles. 
        
        What this means is that the search goes through all of the levels in the tree, optionally starting from a
        different node than the root node. And, if there are as of yet not extended nodes, it extends at most the
        specified number of nodes which have the highest agreement values of those that are not yet extended.

        Parameters
        ----------
        max_width : int
            The maximum number of tangles to extend each level.
        explore_agreement_lower_bound : int
            Only extend tangles which have at least this agreement value.
        max_depth : int, optional
            The maximum depth to extend to. The depth is specified relatively to the root.
        start_node : Tangle, optional
            Can be used to set a different starting node than the root node.
        """

        for i, level in self._search_tree._levels(start_node or self._search_tree.root, explore_agreement_lower_bound, max_depth or self.tree.number_of_separations):
            potential_extend = [node for node in level if node.is_leaf()]
            highest_agreement_nodes = heapq.nlargest(max_width, potential_extend, key=lambda node: node.agreement)
            self._algorithm.extend_tangles(highest_agreement_nodes, self.tree.sep_ids[i], np.empty(0))

    def append_separation(self, new_sep_id: int, agreement_lower_bound: Optional[int]=None) -> int:
        """Append a new separation to the tree. If the final level is empty nothing happens.
        
        By default, only nodes which have agreement value of at least the agreement search limit are extended.
        This is to ensure that by default appending a separation does not change the tangle search limit.

        Parameters
        ----------
        new_sep_id : int
            The id of the separation to append.
        agreement_lower_bound : int, optional
            Only nodes with agreement at least `agreement_lower_bound` are extended in the appending step.

        Returns
        -------
        int
            Number of separations in the last level after appending. If this number is 0 then appending
            will not add any more tangles.
        """

        self.insert_separation(self._search_tree.number_of_separations, new_sep_id, agreement_lower_bound=agreement_lower_bound)
        return len(self._search_tree.k_tangles(self._search_tree.number_of_separations, agreement_lower_bound or self.tree.limit))

    def insert_separation(self, insertion_idx: int, new_sep_id: int, agreement_lower_bound: Optional[int] = None):
        """Insert a new separation into a specified level in the tree. 
        
        Below the insertion level, nodes which have parents that have an agreement value which lies below 
        `agreement_lower_bound` are discarded. By default this bound is the agreement search limit. This is to ensure
        that by default an insertion does not increase the limit.

        This causes all previously existing tangles that are of a higher order than the new separation
        to attempt to add the new separation as well. If this fails they are removed from the tree.

        Inserting the separations has the same result as going back in time and simply appending the
        separations in the correct order.

        Parameters
        ----------
        insertion_idx : int
            The level at which the new separation is inserted.
        new_sep_id : int
            The id of the separation to append.
        agreement_lower_bound : int, optional
            Only nodes with agreement at least `agreement_lower_bound` are extended in the appending step.
        """

        min_agreement = agreement_lower_bound or self._search_tree.limit
        insert_layer = self._search_tree.k_tangles(insertion_idx, 0)
        filtered_insert_layer = []
        for node in insert_layer:
            if node.agreement < min_agreement:
                node.open()
            else:
                filtered_insert_layer.append(node)
        insert_layer = filtered_insert_layer
        self._search_tree._insert_sep_id(insertion_idx, new_sep_id)
        sep_ids_needing_updates = self._search_tree._sep_ids_to_update_after_insertion(insertion_idx)
        self._algorithm.extend_tangles(insert_layer, new_sep_id, sep_ids_needing_updates, forbidden_agreement=min_agreement)

    @property
    def tree(self) -> TangleSearchTree:
        """The TangleSearchTree on which this TangleSweep operates."""

        return self._search_tree
