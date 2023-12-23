import pytest
import numpy as np
from tangles.util.tree import BinTreeNode

@pytest.fixture
def bin_tree() -> BinTreeNode:
  #         8
  #        / \
  #      6    7
  #    /   \
  #   2     5
  #  / \   / \
  # 0   1 3   4
  n = [BinTreeNode() for i in range(9)]
  n[8].set_left_child(n[6])
  n[8].set_right_child(n[7])
  n[6].set_left_child(n[2])
  n[2].set_left_child(n[0])
  n[2].set_right_child(n[1])
  n[6].set_right_child(n[5])
  n[5].set_left_child(n[3])
  n[5].set_right_child(n[4])
  return n[8]

def test_set_child(bin_tree: BinTreeNode):
  new_right_node = BinTreeNode()
  new_left_node = BinTreeNode()
  bin_tree.set_right_child(new_right_node)
  bin_tree.set_left_child(new_left_node)
  assert bin_tree.right_child is new_right_node
  assert bin_tree.left_child is new_left_node
  assert new_left_node.parent is bin_tree
  assert new_right_node.parent is bin_tree

def test_set_child_detaches_only_when_responsible(bin_tree: BinTreeNode):
  new_parent = BinTreeNode()
  previous_right_child = bin_tree.right_child
  previous_left_child = bin_tree.left_child
  previous_left_child.parent = new_parent
  previous_right_child.parent = new_parent
  bin_tree.set_left_child(None)
  bin_tree.set_right_child(None)
  assert previous_left_child.parent is new_parent
  assert previous_right_child.parent is new_parent

def test_detach(bin_tree: BinTreeNode):
  previous_right_child = bin_tree.right_child
  previous_left_child = bin_tree.left_child
  bin_tree.left_child.detach()
  bin_tree.right_child.detach()
  assert bin_tree.left_child is None
  assert bin_tree.right_child is None
  assert previous_left_child.parent is None
  assert previous_right_child.parent is None

def test_copy(bin_tree: BinTreeNode):
  copy = bin_tree.copy()
  assert isinstance(copy, BinTreeNode)
  assert copy.parent is None
  assert copy.left_child is None
  assert copy.right_child is None

def test_copy_subtree(bin_tree: BinTreeNode):
  assert _compare_subtrees(bin_tree.copy_subtree(), bin_tree)
  assert _compare_subtrees(bin_tree.right_child.copy_subtree(), bin_tree.right_child)

def test_copy_subtree_into_children(bin_tree: BinTreeNode):
  bin_tree_copy = bin_tree.copy_subtree()
  bin_tree.copy_subtree_into_children()
  assert _compare_subtrees(bin_tree.left_child, bin_tree_copy)
  assert _compare_subtrees(bin_tree.right_child, bin_tree_copy)
  assert _do_children_point_to_parents(bin_tree)

def _do_children_point_to_parents(node: BinTreeNode) -> bool:
  left_correct, right_correct = True, True
  if node.left_child:
    if not node.left_child.parent is node or not _do_children_point_to_parents(node.left_child):
      left_correct = False
  if node.right_child:
    if not node.right_child.parent is node or not _do_children_point_to_parents(node.right_child):
      right_correct = False
  return left_correct and right_correct

def _compare_subtrees(node: BinTreeNode, other_node: BinTreeNode) -> bool:
  if node.left_child:
    if not other_node.left_child:
      return False
    if not _compare_subtrees(node.left_child, other_node.left_child):
      return False
  else:
    if other_node.left_child:
      return False

  if node.right_child:
    if not other_node.right_child:
      return False
    if not _compare_subtrees(node.right_child, other_node.right_child):
      return False
  else:
    if other_node.right_child:
      return False

  return True

def test_children(bin_tree: BinTreeNode):
  assert bin_tree.children() == [bin_tree.left_child, bin_tree.right_child]
  one_sided_node = BinTreeNode()
  one_sided_node.set_left_child(BinTreeNode())
  assert len(one_sided_node.children()) == 1

def test_level_in_subtree(bin_tree: BinTreeNode):
  expected_number = [1, 2, 2, 4, 0]
  for depth in range(0, 5):
    nodes = bin_tree.level_in_subtree(depth)
    assert len(nodes) == expected_number[depth]
    for node in nodes:
      assert node.level() == depth

def test_level(bin_tree: BinTreeNode):
  assert bin_tree.level() == 0
  assert bin_tree.right_child.level() == 1

def test_path_from_root_indicator(bin_tree: BinTreeNode):
  assert bin_tree.left_child.right_child.path_from_root_indicator() == [-1, 1]
  assert bin_tree.path_from_root_indicator() == []

def test_to_indicator_matrix(bin_tree: BinTreeNode):
  print(BinTreeNode.to_indicator_matrix(bin_tree.leaves_in_subtree()))
  assert np.all(
    BinTreeNode.to_indicator_matrix(bin_tree.leaves_in_subtree()) ==
    np.array([
      [1, 0, 0],
      [-1, -1, -1],
      [-1, -1, 1],
      [-1, 1, -1],
      [-1, 1, 1]], dtype=int
    )
  )

def test_leaves_in_subtree(bin_tree: BinTreeNode):
  max_depth_values = [None, 0, 1, 2, 3]
  expected_number_of_leaves = [5, 1, 2, 3, 5]
  for i in range(len(max_depth_values)):
    leaves = bin_tree.leaves_in_subtree(max_depth=max_depth_values[i])
    assert len(leaves) == expected_number_of_leaves[i]
    if max_depth_values[i] is None or max_depth_values[i] == 3:
      for leaf in leaves:
        assert leaf.is_leaf()