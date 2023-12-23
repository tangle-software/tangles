from typing import Optional
import numpy as np
from tangles._tangle import Tangle
from tangles.util._subsets import all_subsets
from tangles.agreement import CalculateAgreement
from tangles._typing import AgreementFunc, LessOrEqFunc, OrientedSep


class ExtendingTangles:
    """
    The Algorithm responsible for extending tangles.

    Parameters
    ----------
    agreement_func : AgreementFunc
        A function to calculate the agreement value for checked tuples.
    le_func : LessOrEqFunc
        A function which answers whether one oriented separation is less than or equal
        another oriented separation.
    forbidden_tuple_size : int
        The maximum size of tuples that are checked in the algorithm.
    """

    def __init__(self,
                 agreement_func: AgreementFunc,
                 le_func: LessOrEqFunc,
                 forbidden_tuple_size: int = 3):
        self._find_agreement = CalculateAgreement(agreement_func, forbidden_tuple_size)
        self._core_logic = _CoreLogic(le_func)

    def extend_tangles(self, level: list[Tangle], new_sep_id:int, sep_ids:np.ndarray, forbidden_agreement: int = 0):
        """Extend a list of tangles of the same order by adding `new_sep_id` into it.
        The separation is also added into every tangle which is a superset of a tangle from the list.

        Parameters
        ----------
        level : list of Tangle
            A list of tangles to extend.
        new_sep_id : int
            The id of the separation that will be inserted.
        sep_ids : np.ndarray
            The ids of the separations not oriented by the list of tangles,
            but oriented by other tangles, sorted by order.
        forbidden_agreement : int
            If a node has this agreement value its children are removed.
        """

        level_to_extend = _Level(level)
        forced_orientations = self._core_logic.which_nodes_are_forced(level_to_extend, new_sep_id)
        level_to_extend.extend_nodes_but_restrict_side(forced_orientations)
        nodes_to_test = level_to_extend.nodes_with_two_children()
        self._core_logic.remove_shadowed_seps(nodes_to_test, new_sep_id)
        surviving_nodes_left = self._add_seps_to_tangles(nodes_to_test.left_children(), {(new_sep_id, -1)}, forbidden_agreement)
        surviving_nodes_right = self._add_seps_to_tangles(nodes_to_test.right_children(), {(new_sep_id, 1)}, forbidden_agreement)
        self._update_descendant_cores(_Level(surviving_nodes_left), (new_sep_id, -1), sep_ids, forbidden_agreement)
        self._update_descendant_cores(_Level(surviving_nodes_right), (new_sep_id, 1), sep_ids, forbidden_agreement)

    def _update_descendant_cores(self, pending_level:'_Level', new_sep: OrientedSep, sep_ids: np.ndarray, forbidden_agreement: int):
        for level_sep_id in sep_ids:
            if pending_level.is_empty():
                break
            forced_side = self._core_logic.which_level_side_is_forced(new_sep, level_sep_id)
            if forced_side == 1:
                pending_level.detach_left_children()
                pending_level.copy_data_into_children()
                pending_level = _Level(pending_level.right_children())
            elif forced_side == -1:
                pending_level.detach_right_children()
                pending_level.copy_data_into_children()
                pending_level = _Level(pending_level.left_children())
            else:
                forced_nodes, nodes_to_test = self._core_logic.which_nodes_were_forced(pending_level, level_sep_id)
                forced_nodes.copy_data_into_children()
                next_level = forced_nodes.left_children() + forced_nodes.right_children()
                shadowing_side = self._core_logic.by_which_orientation_is_new_sep_shadowed(new_sep, level_sep_id)
                if shadowing_side != 1:
                    self._core_logic.remove_shadowed_seps_using_shadowed_parent(nodes_to_test.right_children())
                    next_level += self._add_seps_to_tangles(nodes_to_test.right_children(), {new_sep, (level_sep_id, 1)}, forbidden_agreement)
                if shadowing_side != -1:
                    self._core_logic.remove_shadowed_seps_using_shadowed_parent(nodes_to_test.left_children())
                    next_level += self._add_seps_to_tangles(nodes_to_test.left_children(), {new_sep, (level_sep_id, -1)}, forbidden_agreement)
                pending_level = _Level(next_level)

    def _add_seps_to_tangles(self, tangles: list[Tangle], add_seps:set[OrientedSep], forbidden_agreement: int) -> list[Tangle]:
        if len(tangles) == 0:
            return []
        self._find_agreement(tangles, add_seps)
        for node in tangles:
            if node.agreement < forbidden_agreement:
                node.open()
        self._core_logic.add_seps_to_cores(tangles, add_seps)
        return tangles


class _CoreLogic:
    def __init__(self, le_func:LessOrEqFunc):
        self._le_func = le_func

    def which_nodes_are_forced(self, level:'_Level', new_sep_id:int) -> np.ndarray[int]:
        subsets, inclusions = all_subsets([tangle.core for tangle in level.get_nodes()], 1)
        good_orientation = np.zeros(len(level), dtype=int)
        if subsets.shape[1] == 0:
            return good_orientation
        for i, subset in enumerate(subsets):
            if np.any(good_orientation[inclusions[i] == 1] == 0):
                current_sep = subset[0]
                if self._le_func(current_sep[0], current_sep[1], new_sep_id, 1):
                    good_orientation[inclusions[i] == 1] = 1
                elif self._le_func(current_sep[0], current_sep[1], new_sep_id, -1):
                    good_orientation[inclusions[i] == 1] = -1
        return good_orientation

    def remove_shadowed_seps(self, level: '_Level', new_sep_id:int):
        all_seps = set()
        for node in level.get_nodes():
            all_seps = all_seps.union(node.core)
        seps_not_dominated_left = {sep for sep in all_seps if not self._le_func(new_sep_id, -1, sep[0], sep[1])}
        seps_not_dominated_right = {sep for sep in all_seps if not self._le_func( new_sep_id, 1, sep[0], sep[1])}
        for node in level.get_nodes():
            node.left_child.core = node.left_child.core.intersection(seps_not_dominated_left)
            node.right_child.core = node.right_child.core.intersection(seps_not_dominated_right)

    def which_level_side_is_forced(self, new_sep:OrientedSep, level_sep_id:int) -> Optional[int]:
        if self._le_func(new_sep[0], new_sep[1], level_sep_id, -1):
            return -1
        if self._le_func(new_sep[0], new_sep[1], level_sep_id, 1):
            return 1
        return None

    def which_nodes_were_forced(self, level: '_Level', level_sep_id:int) -> tuple['_Level', '_Level']:
        nodes_needing_tests = []
        nodes_that_were_forced = []
        for node in level.get_nodes():
            if node.right_child is not None and (level_sep_id, 1) in node.right_child.core:
                nodes_needing_tests.append(node)
            elif node.left_child is None or (level_sep_id, -1) not in node.left_child.core:
                nodes_that_were_forced.append(node)
        return _Level(nodes_that_were_forced), _Level(nodes_needing_tests)

    def remove_shadowed_seps_using_shadowed_parent(self, nodes:list[Tangle]):
        for node in nodes:
            node.core = node.core.intersection(node.parent.core)

    def by_which_orientation_is_new_sep_shadowed(self, new_sep:OrientedSep, level_sep_id:int) -> Optional[int]:
        if self._le_func(level_sep_id, -1, new_sep[0], new_sep[1]):
            return -1
        if self._le_func(level_sep_id, 1, new_sep[0], new_sep[1]):
            return 1
        return None

    def add_seps_to_cores(self, nodes:list[Tangle], add_seps:set[OrientedSep]):
        for node in nodes:
            node.core = node.core.union(add_seps)

class _Level:
    def __init__(self, nodes:list[Tangle]):
        self._nodes = nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def extend_nodes_but_restrict_side(self, side_restricted_to:np.ndarray):
        for node, side in zip(self._nodes, side_restricted_to):
            node.copy_subtree_into_children(left=side!=1, right=side!=-1)

    def detach_left_children(self):
        for node in self._nodes:
            node.set_left_child(None)

    def detach_right_children(self):
        for node in self._nodes:
            node.set_right_child(None)

    def copy_data_into_children(self):
        for child in self.right_children() + self.left_children():
            child.core = child.parent.core
            child.agreement = child.parent.agreement

    def is_empty(self) -> bool:
        return len(self._nodes) == 0

    def left_children(self) -> list[Tangle]:
        return [node.left_child for node in self._nodes if node.left_child]

    def right_children(self) -> list[Tangle]:
        return [node.right_child for node in self._nodes if node.right_child]

    def get_nodes(self) -> list[Tangle]:
        return self._nodes

    def nodes_with_two_children(self) -> '_Level':
        return _Level([node for node in self._nodes if node.left_child and node.right_child])