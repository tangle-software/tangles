import pytest
import numpy as np
from typing import Optional

from tangles.search import TangleSearchTree, FTreeNode, createFTree
from tangles import Tangle
from tangles._typing import AgreementFunc

def test_createftree(tst, agreement_func, ftree):
    new_ftree = createFTree(tst, agreement_func, 1)
    assert ftree == _ftree_to_list(new_ftree)

def test_makeirreducible(tst, agreement_func, ftree_irreducible):
    new_ftree = createFTree(tst, agreement_func, 1)
    new_ftree.make_irreducible()
    assert ftree_irreducible == _ftree_to_list(new_ftree)

def test_empty_forbidden_set(tst, agreement_func, ftree_only_root):
    new_ftree = createFTree(tst, agreement_func, 11)
    new_ftree.make_irreducible()
    assert ftree_only_root == _ftree_to_list(new_ftree)

def test_forced_reduces_agreement_error(tst, agreement_func):
    threw_exception = False
    try:
        _ = createFTree(tst, agreement_func, 2)
    except:
        threw_exception = True
    assert threw_exception

@pytest.fixture
def tst() -> TangleSearchTree:
    return _build_tst([10, 2, 1, 0, 1, 0, 1, 0, 0, None, None, None, None, None, None, 0, 0,
                       None, None, None, None, None, 1, 0, 0, None, None, None, None])

@pytest.fixture
def agreement_func() -> AgreementFunc:
    return _build_agreement_func([
        (set([(0, -1)]), 2), #ensuring the values in the TST are correct
        (set([(1, 1)]), 1),
        (set([(0, 1)]), 1),

        (set([(0, -1), (2, 1)]), 0), #the left side may not be picked
        (set([(0, -1), (2, -1)]), 0), #when removing 0

        (set([(1, -1)]), 0), #ensuring that 0 is nowhere necessary on the right side

        (set([(0, 1), (3, 1)]), 0), #only 0 and 3 would be sufficient
        (set([(0, 1), (3, -1)]), 0), #but since 0 is removed first this is not realized

        (set([(1, 1), (2, -1)]), 0),
        (set([(2, 1), (3, -1)]), 0), #2 may help but its not necessary
        (set([(1, 1), (3, -1)]), 0), #there are multiple options for the
        (set([(1, 1), (3, 1)]), 0), #last leaf, 1 is only necessary if 0 is deleted
    ])

@pytest.fixture
def ftree() -> list[Optional[int]]:
    return [0, 1, 2, 3, None, None, None, None, 2, None, None]

@pytest.fixture
def ftree_irreducible() -> list[Optional[int]]:
    return [1, 3, None, None, None]

@pytest.fixture
def ftree_only_root() -> list[Optional[int]]:
    return [None]

def _build_tst(agreement_list: list[Optional[int]]) -> TangleSearchTree:
    root = Tangle(agreement_list.pop(0), set())
    stack = [root]
    while stack:
        node = stack.pop()
        agreement = agreement_list.pop(0)
        if agreement is not None:
            new_core = node.core.copy()
            new_core.add((node.level(), -1))
            node.set_left_child(Tangle(agreement, new_core))
            stack.append(node.left_child)
        agreement = agreement_list.pop(0)
        if agreement is not None:
            new_core = node.core.copy()
            new_core.add((node.level(), 1))
            node.set_right_child(Tangle(agreement, new_core))
            stack.append(node.right_child)
    return TangleSearchTree(root, np.arange(10))


def _build_agreement_func(agreement_value_subsets: list[tuple[set, int]], max_value: int = 10) -> AgreementFunc:
    def func(sep_ids: np.ndarray, sep_ori: np.ndarray) -> int:
        min_value = max_value
        input_set = set(zip(sep_ids, sep_ori))
        for subset, value in agreement_value_subsets:
            if value < min_value and subset.issubset(input_set):
                min_value = value
        return min_value
    func.max_value = 10

    return func

def _ftree_to_list(root: FTreeNode) -> list[int]:
    stack = [root]
    tree_list = []
    while stack:
        node = stack.pop()
        tree_list.append(node.next_sep_id())
        stack.extend(node.children())
    return tree_list
