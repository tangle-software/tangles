import numpy as np
from tangles.convenience import search_tangles_uncrossed
from tangles.separations import SetSeparationSystem
from tangles.tests.path import get_test_data_path
from tangles._typing import SetSeparationOrderFunction
from tangles.util.matrix_order import matrix_order, covariance_order
from itertools import combinations, product
from tangles.analysis import tangle_score


# this should be the convenience-test case, and it should check the tangle matrix before verifying the tree
class ConvenienceTestCase:
  def __init__(self, name: str, seps: np.ndarray, order_func: SetSeparationOrderFunction):
    self._name = name
    self._seps = seps
    self._order_func = order_func

  @staticmethod
  def build_test_case(name:str, seps: np.ndarray, L: np.ndarray):
    np.savez(get_test_data_path(f"agreement_test_cases/{name}.npz"), seps=seps, L=L)

  @staticmethod
  def load_test_case(name: str):
    test_npz = np.load(get_test_data_path(f"agreement_test_cases/{name}.npz"))
    M = test_npz['L'].astype(float)
    return ConvenienceTestCase(
      name=name,
      seps=test_npz['seps'],
      order_func=lambda seps: matrix_order(M, seps)
    )

  # not sure if the covariance test cases (if any) should be handled like this. The test case should contain information about the used order function
  @staticmethod
  def load_test_case_covariance(name: str, shift=0):  # shift is needed nearly always if a covariance matrix is used to build a ToT (submodularity!)
    test_npz = np.load(get_test_data_path(f"agreement_test_cases/{name}.npz"))
    M = test_npz['seps'].astype(float)

    return ConvenienceTestCase(
      name=name,
      seps=test_npz['seps'],
      order_func=lambda s: -covariance_order(M.T, s, shift=shift)
    )

  def test_tst(self, agreement:int):
    sep_sys = SetSeparationSystem.with_array(self._seps)
    assert self._less_original_seps_than_max_number_of_seps(sep_sys, agreement, max_number_of_seps=10)

    uncrossed_sweep = search_tangles_uncrossed(sep_sys, agreement, self._order_func)
    assert self._tangle_matrix_only_original_seps_parameter(uncrossed_sweep, len(sep_sys))

    tangle_matrix = uncrossed_sweep.tree.tangle_matrix(agreement)
    scores = tangle_score(tangle_matrix, uncrossed_sweep.tree.sep_ids[:tangle_matrix.shape[1]], uncrossed_sweep.sep_sys, normalize_rows=True)
    assert self._tangle_scores_of_a_datapoint_sum_up_to_one(scores)
    assert self._minimal_tangle_score_of_a_datapoint_is_zero(scores)

  def _tangle_scores_of_a_datapoint_sum_up_to_one(self, scores):
    return np.all(np.isclose(scores.sum(axis=1),1))

  def _minimal_tangle_score_of_a_datapoint_is_zero(self, scores):
    return not np.any(scores.min(axis=1))

  def _less_original_seps_than_max_number_of_seps(self, sep_sys, agreement, max_number_of_seps):
    uncrossed_sweep = search_tangles_uncrossed(sep_sys, agreement, self._order_func, max_number_of_seps)
    return uncrossed_sweep.number_of_initial_seps_added <= max_number_of_seps

  def _tangle_matrix_only_original_seps_parameter(self, uncrossed_sweep, number_of_original_seps):
    tangle_matrix = uncrossed_sweep.tangle_matrix(only_original_seps=True)
    return tangle_matrix.shape[1] <= number_of_original_seps

  def test_tot(self, agreement:int):
    sep_sys = SetSeparationSystem.with_array(self._seps)
    uncrossed_sweep = search_tangles_uncrossed(sep_sys, agreement, self._order_func)
    tot = uncrossed_sweep.create_tot(agreement)

    self._verify_tot_tree_structure(tot)
    self._verify_tot_neighbours(tot)
    self._verify_tot_splits(tot)
    self._verify_tot_respects_sep_le_ordering(tot, uncrossed_sweep.sep_sys.is_le)

  def _verify_tot_tree_structure(self, tot):
    for edge in tot.edges:
      nodes_in_subtree_1 = set(edge.node1.node_iter(blocked_nodes={edge.node2}))
      nodes_in_subtree_2 = set(edge.node2.node_iter(blocked_nodes={edge.node1}))
      assert self._subtrees_share_no_nodes(nodes_in_subtree_1, nodes_in_subtree_2)
      assert self._subtrees_contain_all_tot_nodes(nodes_in_subtree_1, nodes_in_subtree_2, tot.nodes)

  def _subtrees_share_no_nodes(self, nodes_in_subtree_1, nodes_in_subtree_2):
    return len(nodes_in_subtree_1 & nodes_in_subtree_2) == 0

  def _subtrees_contain_all_tot_nodes(self, nodes_in_subtree_1, nodes_in_subtree_2, all_tot_nodes):
    return set(nodes_in_subtree_1 | nodes_in_subtree_2) == set(all_tot_nodes)

  def _verify_tot_neighbours(self, tot):
    for edge in tot.edges:
      assert self._tangles_at_edge_have_exactly_one_oppositely_oriented_sep(edge)
      assert self._sep_id_of_edge_matches_oppositely_oriented_sep(edge, tot)

  def _tangles_at_edge_have_exactly_one_oppositely_oriented_sep(self, edge):
    diff = edge.node1.reduced_tangle != edge.node2.reduced_tangle
    diff[edge.node1.reduced_tangle==0] = 0
    diff[edge.node2.reduced_tangle==0] = 0
    return np.sum(diff) == 1

  def _sep_id_of_edge_matches_oppositely_oriented_sep(self, edge, tot):
    diff = edge.node1.reduced_tangle != edge.node2.reduced_tangle
    diff[edge.node1.reduced_tangle==0] = 0
    diff[edge.node2.reduced_tangle==0] = 0
    return tot.sep_ids[diff] == edge.sep_id

  def _verify_tot_splits(self, tot):
    for edge in tot.edges:
      assert self._edge_sep_idx_points_to_correct_sep_id(edge, tot.sep_ids)

      nodes_in_subtree_1 = list(edge.node1.node_iter(blocked_nodes={edge.node2}))
      nodes_in_subtree_2 = list(edge.node2.node_iter(blocked_nodes={edge.node1}))
      assert self._two_nodes_from_different_subtrees_orient_sep_of_edge_differently(nodes_in_subtree_1, nodes_in_subtree_2, edge.sep_idx)
      assert self._two_nodes_from_same_subtree_orient_sep_of_edge_equally(nodes_in_subtree_1, edge.sep_idx)
      assert self._two_nodes_from_same_subtree_orient_sep_of_edge_equally(nodes_in_subtree_2, edge.sep_idx)

  def _edge_sep_idx_points_to_correct_sep_id(self, edge, sep_ids):
    return sep_ids[edge.sep_idx] == edge.sep_id

  def _two_nodes_from_different_subtrees_orient_sep_of_edge_differently(self, nodes_in_subtree_1, nodes_in_subtree_2, sep_idx_of_edge):
    for n1, n2 in product(nodes_in_subtree_1, nodes_in_subtree_2):
      if n1.reduced_tangle[sep_idx_of_edge] == 0 and n2.reduced_tangle[sep_idx_of_edge] == 0:
        continue
      elif n1.reduced_tangle[sep_idx_of_edge] == n2.reduced_tangle[sep_idx_of_edge]:
        return False
    return True

  def _two_nodes_from_same_subtree_orient_sep_of_edge_equally(self, nodes_in_subtree, sep_idx_of_edge):
    for n_a, n_b in combinations(nodes_in_subtree, 2):
      if n_a.reduced_tangle[sep_idx_of_edge] == 0 or n_b.reduced_tangle[sep_idx_of_edge] == 0:
        continue
      elif n_a.reduced_tangle[sep_idx_of_edge] != n_b.reduced_tangle[sep_idx_of_edge]:
        return False
    return True

  def _verify_tot_respects_sep_le_ordering(self, tot, le_func):
    start_edge = tot.edges[0]
    node_direction_1 = start_edge.node1
    node_direction_2 = start_edge.node2

    self._check_correct_le_ordering_to_children(node_direction_1, start_edge, le_func)
    self._check_correct_le_ordering_to_children(node_direction_2, start_edge, le_func)

  def _check_correct_le_ordering_to_children(self, parent, parent_edge, le_func):
    p_orient = parent.reduced_tangle[parent_edge.sep_idx]
    neighbour = parent_edge.other_end(parent)
    neighbour_edges = [e for e in neighbour.edges if e != parent_edge]

    for n_edge in neighbour_edges:
      n_orient = neighbour.reduced_tangle[n_edge.sep_idx]
      assert le_func(parent_edge.sep_id, p_orient, n_edge.sep_id, n_orient)

      self._check_correct_le_ordering_to_children(neighbour, n_edge, le_func)