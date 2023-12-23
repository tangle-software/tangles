import numpy as np
import itertools

from tangles.util.tree import BinTreeNode
from tangles.search._tree import TangleSearchTree
from tangles._typing import AgreementFunc, OrientedSep
from typing import Optional
from collections import defaultdict

class FTreeNode(BinTreeNode):
    """
    An Forbidden-Tuples-Tree (FTT). A binary tree containing forbidden tuples in every leaf.
    """
    
    def __init__(self,
                 new_sep_id: int,
                 parent=None,
                 left_child = None,
                 right_child = None):
        super().__init__(parent=parent, left_child=left_child, right_child=right_child)
        self.forbidden_tuples = []
        self._new_sep_id = new_sep_id

    def next_sep_id(self) -> Optional[int]:
        chosen_child = self.left_child or self.right_child
        return chosen_child and chosen_child._new_sep_id

    @property
    def core(self) -> list[OrientedSep]:
        my_core = []
        node = self
        while node.parent:
            if node is node.parent.left_child:
                my_core.append((node.parent.next_sep_id(), -1))
            if node is node.parent.right_child:
                my_core.append((node.parent.next_sep_id(), 1))
            node = node.parent
        return my_core

    def make_irreducible(self):
        stack = [self]
        while stack:
            node = stack.pop()
            while (w := node._search_child_whose_leaves_do_not_require_parent_to_be_forbidden()) is not None:
                next_sep_id = node.next_sep_id()
                node._take_everything_from(w)
                node.remove_sep_from_leaves(next_sep_id)

            if not node.is_leaf():
                stack.append(node.left_child)
                stack.append(node.right_child)

    def _search_child_whose_leaves_do_not_require_parent_to_be_forbidden(self) -> Optional['FTreeNode']:
        if self.is_leaf():
            return None

        parent_sep_id = self.next_sep_id()

        left_forbidden_tuples = [leaf.forbidden_tuples for leaf in self.left_child.leaves_in_subtree()]
        right_forbidden_tuples = [leaf.forbidden_tuples for leaf in self.right_child.leaves_in_subtree()]

        can_take_left = True
        for tuples in left_forbidden_tuples:
            if all((parent_sep_id, 1) in forbidden_tuple or (parent_sep_id, -1) in forbidden_tuple for forbidden_tuple in tuples):
                can_take_left = False
                break

        if can_take_left:
            return self.left_child

        can_take_right = True
        for tuples in right_forbidden_tuples:
            if all((parent_sep_id, 1) in forbidden_tuple or (parent_sep_id, -1) in forbidden_tuple for forbidden_tuple in tuples):
                can_take_right = False
                break

        if can_take_right:
            return self.right_child

        return None

    def _take_everything_from(self, other_node: 'FTreeNode'):
        self._new_sep_id = other_node._new_sep_id
        self.set_left_child(other_node.left_child)
        self.set_right_child(other_node.right_child)
        self.forbidden_tuples = other_node.forbidden_tuples

    def remove_sep_from_leaves(self, sep_id: int):
        leaves = self.leaves_in_subtree()
        for leaf in leaves:
            leaf.forbidden_tuples = [forbidden_tuple
                for forbidden_tuple in leaf.forbidden_tuples
                if (sep_id, 1) not in forbidden_tuple and (sep_id, -1) not in forbidden_tuple
            ]


def _which_nodes_have_certain_subset(node_list:list[FTreeNode], subset_size:int) -> dict:
    subset_occurrence = defaultdict(list)
    for node in node_list:
        for subset in itertools.combinations(sorted(node.core), subset_size):
            subset_occurrence[subset].append(node)
    return subset_occurrence


def _create_next_level(level_nodes: list[FTreeNode], allowed_agreement: float, sep_id_before_new_level: int):
    next_level_nodes = []
    for f_tree_node, tst_node in level_nodes:
        if tst_node.agreement >= allowed_agreement:
            if tst_node.left_child and tst_node.right_child:
                f_tree_node.set_left_child(FTreeNode(sep_id_before_new_level))
                f_tree_node.set_right_child(FTreeNode(sep_id_before_new_level))
                next_level_nodes.extend([(f_tree_node.left_child, tst_node.left_child),
                                         (f_tree_node.right_child, tst_node.right_child)])
            elif tst_node.left_child:
                next_level_nodes.append((f_tree_node, tst_node.left_child))
                if tst_node.left_child.agreement < allowed_agreement:
                    forced_decreases_agreement_warning(sep_id_before_new_level, -1, tst_node.core, tst_node.agreement, tst_node.left_child.agreement)
            elif tst_node.right_child:
                next_level_nodes.append((f_tree_node, tst_node.right_child))
                if tst_node.right_child.agreement < allowed_agreement:
                    forced_decreases_agreement_warning(sep_id_before_new_level, 1, tst_node.core, tst_node.agreement, tst_node.right_child.agreement)
    return next_level_nodes

def forced_decreases_agreement_warning(sep_id_before_new_level, ori, previous_core, prev_agreement, new_agreement):
    raise Exception(f"""
          Error!
          Adding {(sep_id_before_new_level, ori)} to {previous_core} reduces the agreement value
          from {prev_agreement} to {new_agreement} - below the minimal allowed value - despite only
          one child existing in the TST. This is either a problem with the agreement function, if a
          forced separations reduce the agreement value of a tangle, or it is a problem with the tangle
          search tree, that a node has only one child despite the other, missing, child also being consistent.
          """)

def _find_forbidden_tuples(parents: list[FTreeNode], oriented_sep: OrientedSep, ag_func: AgreementFunc, allowed_agreement: float, tuple_size: int):
    if tuple_size == 0:
        if ag_func.max_value < allowed_agreement:
            n.child(oriented_sep[1]).forbidden_tuples.append(tuple([]))
        return
    buf = np.empty((tuple_size, 2), dtype=int)
    buf[0,:] = oriented_sep
    if tuple_size == 1:
        if ag_func(buf[:, 0], buf[:, 1]) < allowed_agreement:
            for n in parents:
                n.child(oriented_sep[1]).forbidden_tuples.append((oriented_sep,))
    else:
        for subset, inclusion in _which_nodes_have_certain_subset(parents, subset_size=tuple_size-1).items():
            buf[1:, :] = subset
            if ag_func(buf[:, 0], buf[:, 1]) < allowed_agreement:
                forbidden_tuple = (oriented_sep,) + subset
                for n in inclusion:
                    n.child(oriented_sep[1]).forbidden_tuples.append(forbidden_tuple)


def _check_level_nodes(level_nodes, sep_id, allowed_agreement, agreement_func, tuple_size):
    forbidden_nodes = [f_t_node for f_t_node, tst_node in level_nodes if tst_node.agreement < allowed_agreement]
    left_parents = [n.parent for n in forbidden_nodes if n.parent.left_child == n]
    right_parents = [n.parent for n in forbidden_nodes if n.parent.right_child == n]
    for ts in range(tuple_size+1):
        _find_forbidden_tuples(left_parents, (sep_id, -1), agreement_func, allowed_agreement, ts)
        _find_forbidden_tuples(right_parents, (sep_id, 1), agreement_func, allowed_agreement, ts)


def createFTree(tst: TangleSearchTree, agreement_func: AgreementFunc, allowed_agreement: float, forbidden_tuple_size=3):
    root = FTreeNode(new_sep_id=None)
    level_nodes = [(root, tst.root)]
    for level_num, sep_id in enumerate(tst.sep_ids):
        if not (level_nodes := _create_next_level(level_nodes, allowed_agreement, sep_id)): break
        _check_level_nodes(level_nodes, sep_id, allowed_agreement, agreement_func, min(level_num+1, forbidden_tuple_size))
    return root
