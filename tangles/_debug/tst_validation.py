import numpy as np
from tangles._typing import LessOrEqFunc, AgreementFunc, OrientedSep
from tangles.search import TangleSearchTree
from tangles import Tangle
from tangles.agreement import memoize_agreement_func

def is_tst_valid(tangle_search_tree: TangleSearchTree, le_func: LessOrEqFunc, agreement_func: AgreementFunc,
                 intended_agreement: int):
    if not check_limit(tangle_search_tree, intended_agreement):
        print("Limit is wrong: ", tangle_search_tree.limit)
        return False
    if not check_number_of_children(tangle_search_tree):
        print("Tree Structure is wrong")
        return False
    if not check_core(tangle_search_tree, le_func):
        print("Core error")
        return False
    if not check_agreement_falling(tangle_search_tree):
        print("Agreement increases going down tree")
        return False
    if not check_agreement_correct(tangle_search_tree, agreement_func):
        print("Wrong Agreement Value")
        return False
    return True

def check_limit(tangle_search_tree: TangleSearchTree, agreement:int):
    return tangle_search_tree.limit < agreement

def check_number_of_children(tangle_search_tree: TangleSearchTree):
    for idx, level in tangle_search_tree._levels(tangle_search_tree.root, 0):
        for node in level:
            if node.left_child and node.right_child:
                if (tangle_search_tree.sep_ids[idx], 1) not in node.right_child.core:
                    return False
                if (tangle_search_tree.sep_ids[idx], -1) not in node.left_child.core:
                    return False
            elif not node.is_leaf():
                if node.right_child:
                    if (tangle_search_tree.sep_ids[idx], 1) in node.right_child.core:
                        return False
                    if (node.core is not node.right_child.core) or (node.agreement != node.right_child.agreement):
                        return False
                if node.left_child:
                    if (tangle_search_tree.sep_ids[idx], -1) in node.left_child.core:
                        return False
                    if (node.core is not node.left_child.core) or (node.agreement != node.left_child.agreement):
                        return False
    return True

def check_core(tangle_search_tree: TangleSearchTree, le_func:LessOrEqFunc):
    for idx, level in tangle_search_tree._levels(tangle_search_tree.root, 0):
        for node in level:
            if node.parent:
                new_sep = (tangle_search_tree.sep_ids[idx-1], 1)
                if node.parent.left_child is node:
                    new_sep = (tangle_search_tree.sep_ids[idx-1], -1)
                if new_sep not in node.core:
                    new_sep_shadowed = False
                    for sep in node.core:
                        if le_func(sep[0], sep[1], new_sep[0], new_sep[1]):
                            new_sep_shadowed = True
                            break
                    if not new_sep_shadowed:
                        return False
                    if not node.core is node.parent.core:
                        return False
                else:
                    for sep in node.parent.core:
                        if sep in node.core:
                            if le_func(new_sep[0], new_sep[1], sep[0], sep[1]):
                                return False
                        else:
                            if not le_func(new_sep[0], new_sep[1], sep[0], sep[1]):
                                return False
                    if len(node.core.difference(node.parent.core)) != 1:
                        return False
    return True

def check_agreement_falling(tangle_search_tree: TangleSearchTree):
    for _, level in tangle_search_tree._levels(tangle_search_tree.root, 0):
        for node in level:
            if node.parent and not node.parent.agreement >= node.agreement:
                print(node.parent.agreement, node.agreement)
                return False
    return True

def check_agreement_correct(tangle_search_tree: TangleSearchTree, agreement_func: AgreementFunc):
    mem_agreement = memoize_agreement_func(agreement_func)
    for idx, level in tangle_search_tree._levels(tangle_search_tree.root, 0):
        for node in level:
            if node.parent:
                new_sep = (tangle_search_tree.sep_ids[idx-1], 1)
                if node.parent.left_child is node:
                    new_sep = (tangle_search_tree.sep_ids[idx-1], -1)
                if new_sep in node.core:
                    if (node.parent.agreement >= node.agreement and node.agreement != node.parent.agreement):
                        calc = _calculate_node_agreement(node, mem_agreement, new_sep)
                        if not calc == node.agreement:
                            return False
                    else:
                        calc = _calculate_node_agreement(node, mem_agreement, new_sep)
                        if node.agreement > calc:
                            return False
    return True

def _calculate_node_agreement(node: Tangle, agreement_func: AgreementFunc, new_sep: OrientedSep):
    if len(node.core) == 1:
        return agreement_func(np.array([new_sep[0]]), np.array([new_sep[1]]))
    sep_ids = np.empty(3, dtype=int)
    orientations = np.empty(3, dtype=int)
    sep_ids[0] = new_sep[0]
    orientations[0] = new_sep[1]
    min_agreement = np.iinfo(int).max
    for sep1 in node.core:
        sep_ids[1] = sep1[0]
        orientations[1] = sep1[1]
        for sep2 in node.core:
            sep_ids[2] = sep2[0]
            orientations[2] = sep2[1]
            agreement = agreement_func(sep_ids, orientations)
            if agreement < min_agreement:
                min_agreement = agreement
    return min_agreement
