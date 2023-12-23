import pytest
import numpy as np
from tangles.search import TangleSearchTree
from tangles import Tangle
from tangles.separations import FeatureSystem

def _tangles_fruit() -> dict[str, Tangle]:
    return {
        'root': Tangle(6, set()),
        'citrus': Tangle(4, {(0, -1)}),
        'tropical': Tangle(2, {(0, 1)}),
        'citrus_forced': Tangle(3, {(0, -1), (1, 1)}),
        'tropical_forced': Tangle(2, {(0, 1)}),
        'tropical_forced_again': Tangle(2, {(0, 1)}),
        'orange': Tangle(2, {(0, -1), (1, 1), (2, -1)}),
        'lemon': Tangle(1, {(0, -1), (1, 1), (2, 1)})
    }

def _tree_fruit(tangles: dict[str, Tangle]) -> TangleSearchTree:
    tangles['root'].set_left_child(tangles['citrus'])
    tangles['root'].set_right_child(tangles['tropical'])
    tangles['citrus'].set_right_child(tangles['citrus_forced'])
    tangles['tropical'].set_right_child(tangles['tropical_forced'])
    tangles['tropical_forced'].set_left_child(tangles['tropical_forced_again'])
    tangles['citrus_forced'].set_left_child(tangles['orange'])
    tangles['citrus_forced'].set_right_child(tangles['lemon'])
    return TangleSearchTree(tangles['root'], np.arange(3))

def _feat_sys_fruit():
    """
    The Fruit are, in order: Apple, Orange, Mandarine, Lemon, Dragonfruit, Pineapple
    """
    features = np.array(
        [
            [-1, -1 , 1],
            [-1, 1 , -1],
            [-1 , 1, -1],
            [-1 , 1, 1],
            [1 , 1, 1 ],
            [1 , 1, 1 ]
        ]
    )
    return FeatureSystem.with_array(features)

@pytest.fixture
def fruit() -> dict:
    """
    Small fruit example.
    """
    tangles = _tangles_fruit()
    return {
        'tangles': tangles,
        'tangle_search_tree': _tree_fruit(tangles),
        'feat_sys': _feat_sys_fruit()
    }

def build_random(min_number_of_nodes: int, height: int, average_agreement_change: int = 3) -> TangleSearchTree:
    root = Tangle(2 * average_agreement_change * height)
    number_of_nodes = 1
    tree_height = 0
    while number_of_nodes < min_number_of_nodes or tree_height < height:
        leaf = _choose_random_extendable_leaf(root, height)
        new_leaf_left, _ = _extend_node(leaf, average_agreement_change)
        number_of_nodes += 2
        new_nodes_height = new_leaf_left.level()
        if new_nodes_height > tree_height:
            tree_height = new_nodes_height
    return TangleSearchTree(root, np.arange(height))

def _choose_random_extendable_leaf(root: Tangle, max_height: int) -> Tangle:
    leaves = root.leaves_in_subtree()
    leaves = [leaf for leaf in leaves if leaf.agreement > 0 and leaf.level() < max_height]
    return leaves[np.random.randint(len(leaves))]

def _extend_node(node: Tangle, average_agreement_change: int) -> tuple[Tangle, Tangle]:
    node.set_left_child(Tangle(node.agreement - np.random.randint(2*average_agreement_change + 1)))
    node.set_right_child(Tangle(node.agreement - np.random.randint(2*average_agreement_change + 1)))
    return node.left_child, node.right_child