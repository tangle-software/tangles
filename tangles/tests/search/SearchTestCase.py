import pickle
import numpy as np
from typing import List
from tangles.search import TangleSweep
from tangles import agreement_func, Tangle
from tangles.separations.system import SetSeparationSystem
from tangles.tests.path import get_test_data_path
from tangles._debug.tst_validation import is_tst_valid
from tangles.search import TangleSearchTree

class SimpleTSTestCase:
  def __init__(self, name: str, feats: np.ndarray, agreements: List[int], tree: TangleSearchTree):
    self._name = name
    self._feats = feats
    self._agreements = agreements
    self._tree = tree

  @staticmethod
  def build(name: str) -> 'SimpleTSTestCase':
    test_npz = np.load(get_test_data_path(f"agreement_test_cases/{name}.npz"))
    agreements = list(test_npz['agreements'])
    feats = test_npz['seps']
    feat_sys, sep_info = SetSeparationSystem.with_array(feats, return_sep_info=True)
    search = TangleSweep(agreement_func(feat_sys), le_func=feat_sys.is_le)
    for i in sep_info[0]:
      if search.append_separation(i, min(agreements)) == 0:
        break
    if not is_tst_valid(search.tree, feat_sys.is_le, agreement_func(feat_sys), min(agreements)):
      return None
    return SimpleTSTestCase(name=name, feats=feats, agreements=agreements, tree=search.tree)

  @staticmethod
  def load(name: str) -> 'SimpleTSTestCase':
    feats = np.load(get_test_data_path(f"tst_data/{name}/feats.npy"))
    with open(get_test_data_path(f"tst_data/{name}/agreements.pickle"), "rb") as f:
      agreements = pickle.load(f)
    with open(get_test_data_path(f"tst_data/{name}/parent_index.pickle"), "rb") as f:
      parent_index =pickle.load(f)
    with open(get_test_data_path(f"tst_data/{name}/which_child.pickle"), "rb") as f:
      which_child =pickle.load(f)
    with open(get_test_data_path(f"tst_data/{name}/cores.pickle"), "rb") as f:
      cores = pickle.load(f)
    with open(get_test_data_path(f"tst_data/{name}/tangle_agreements.pickle"), "rb") as f:
      tangle_agreements = pickle.load(f)
    nodes = []
    for i in range(len(parent_index)):
      parent = None if parent_index[i] == -1 else nodes[parent_index[i]]
      node = Tangle(tangle_agreements[i], cores[i], parent)
      if parent:
        if which_child[i] == -1:
          parent.left_child = node
        else:
          parent.right_child = node
      nodes.append(node)
    feat_sys, sep_info = SetSeparationSystem.with_array(feats, return_sep_info=True)
    tree = TangleSearchTree(nodes[0], np.arange(feats.shape[1]))
    if not is_tst_valid(tree, feat_sys.is_le, agreement_func(feat_sys), min(agreements)):
      print('Corrupted Files', name)
      return None
    return SimpleTSTestCase(name, feats, agreements, tree)

  def save(self) -> 'SimpleTSTestCase':
    np.save(get_test_data_path(f"tst_data/{self._name}/feats"), self._feats)
    with open(get_test_data_path(f"tst_data/{self._name}/agreements.pickle"), "wb") as f:
      pickle.dump(self._agreements, f)
    nodes = []
    parent_index = []
    which_child = []
    cores = []
    tangle_agreements = []
    for _, level in self._tree._levels(self._tree.root, -2):
      for node in level:
        nodes.append(node)
        cores.append(node.core)
        tangle_agreements.append(node.agreement)
        if node.parent:
          parent_index.append(nodes.index(node.parent))
          if node is node.parent.left_child:
            which_child.append(-1)
          else:
            which_child.append(1)
        else:
          parent_index.append(-1)
          which_child.append(0)
    with open(get_test_data_path(f"tst_data/{self._name}/parent_index.pickle"), "wb") as f:
      pickle.dump(parent_index, f)
    with open(get_test_data_path(f"tst_data/{self._name}/which_child.pickle"), "wb") as f:
      pickle.dump(which_child, f)
    with open(get_test_data_path(f"tst_data/{self._name}/cores.pickle"), "wb") as f:
      pickle.dump(cores, f)
    with open(get_test_data_path(f"tst_data/{self._name}/tangle_agreements.pickle"), "wb") as f:
      pickle.dump(tangle_agreements, f)
    return self

  def run_append(self, agreement: int) -> bool:
    feat_sys, sep_info = SetSeparationSystem.with_array(self._feats, return_sep_info=True)
    search = TangleSweep(agreement_func(feat_sys), le_func=feat_sys.is_le)
    for i in sep_info[0]:
      if search.append_separation(i, agreement) == 0:
        break

    return search.tree.is_subtree_of(self._tree) and search.tree.limit < agreement

  def run_insert_half(self, agreement: int) -> bool:
    np.set_printoptions(linewidth=100000)
    _, num_seps = self._feats.shape
    half = num_seps // 2
    feat_sys, sep_info = SetSeparationSystem.with_array(self._feats, return_sep_info=True)
    search = TangleSweep(agreement_func(feat_sys), le_func=feat_sys.is_le)
    for i in range(half, num_seps):
      if search.append_separation(sep_info[0][i], agreement) == 0:
        break
    for i in range(half):
      search.insert_separation(sep_info[0][i], sep_info[0][i], agreement)

    return search.tree.is_subtree_of(self._tree) and search.tree.limit < agreement

  def run_insert_interleaved(self, agreement: int) -> bool:
    np.set_printoptions(linewidth=100000)
    _, num_seps = self._feats.shape
    feat_sys, sep_info = SetSeparationSystem.with_array(self._feats, return_sep_info=True)
    search = TangleSweep(agreement_func(feat_sys), le_func=feat_sys.is_le)
    max = num_seps
    for i in range(1, num_seps,2):
      if search.append_separation(sep_info[0][i], agreement) == 0:
        max = i
        break
    for i in range(0, num_seps,2):
      if i >= max:
        break
      search.insert_separation(sep_info[0][i], sep_info[0][i], agreement)

    return search.tree.is_subtree_of(self._tree) and search.tree.limit < agreement

  def run_agreement_interaction(self, agreement: int) -> bool:
    np.set_printoptions(linewidth=100000)
    _, num_seps = self._feats.shape
    feat_sys, sep_info = SetSeparationSystem.with_array(self._feats, return_sep_info=True)
    search = TangleSweep(agreement_func(feat_sys), le_func=feat_sys.is_le)
    max = num_seps
    min_agreement = min(self._agreements)
    for i in range(1, num_seps,2):
      if search.append_separation(sep_info[0][i], min_agreement) == 0:
        max = i
        break
    for i in range(0, num_seps,2):
      if i >= max:
        break
      search.insert_separation(sep_info[0][i], sep_info[0][i], agreement)
    search.sweep_below(min_agreement)

    return search.tree.is_subtree_of(self._tree) and search.tree.limit < agreement

  def run_sweep(self):
    self._sweep_test()
    self._sweep_append_test()
    self._sweep_insert_test()
    self._sweep_greedy_test()

  def _sweep_test(self):
    feat_sys, [sep_ids, _] = SetSeparationSystem.with_array(self._feats, return_sep_info=True)
    sweep = TangleSweep(agreement_func(feat_sys), feat_sys.is_le, sep_ids)
    assert sweep.sweep_one() >= 0
    max_test_agreement = max(self._agreements)
    limit = sweep.sweep_below(max_test_agreement)
    assert limit == sweep.sweep_below(max_test_agreement)
    min_test_agreement = min(self._agreements)
    sweep.sweep_below(min_test_agreement)
    assert sweep.tree.limit < min_test_agreement
    assert sweep.tree.is_subtree_of(self._tree)
    assert sweep.sweep_one() >= 0

  def _sweep_append_test(self):
    num_seps = self._feats.shape[1]
    feat_sys, [sep_ids, _] = SetSeparationSystem.with_array(self._feats, return_sep_info=True)
    sweep = TangleSweep(agreement_func(feat_sys), feat_sys.is_le)
    min_test_agreement = min(self._agreements)
    for sep_id in range(num_seps):
      sweep.append_separation(sep_ids[sep_id], agreement_lower_bound= min_test_agreement)
    assert sweep.tree.is_subtree_of(self._tree)

  def _sweep_insert_test(self):
    num_seps = self._feats.shape[1]
    feat_sys, [sep_ids, _] = SetSeparationSystem.with_array(self._feats, return_sep_info=True)
    sweep = TangleSweep(agreement_func(feat_sys), feat_sys.is_le)
    min_test_agreement = min(self._agreements)
    for i in range(1, num_seps,2):
      sweep.append_separation(sep_ids[i], agreement_lower_bound= min_test_agreement)
    for i in range(0, num_seps,2):
      sweep.insert_separation(sep_ids[i], sep_ids[i], agreement_lower_bound= min_test_agreement)
    assert sweep.tree.is_subtree_of(self._tree)

  def _sweep_greedy_test(self):
    feat_sys, [sep_ids, _] = SetSeparationSystem.with_array(self._feats, return_sep_info=True)

    sweep = TangleSweep(agreement_func(feat_sys), feat_sys.is_le, sep_ids)
    sweep.sweep_one()
    sweep.greedy_search(1, 1, start_node = sweep.tree.root.left_child)
    sweep.greedy_search(1, 1, start_node = sweep.tree.root.right_child)
    for i in range(len(sep_ids)-1):
      assert len(sweep.tree.root.left_child.level_in_subtree(i)) <= 2
      assert len(sweep.tree.root.right_child.level_in_subtree(i)) <= 2

    min_test_agreement = min(self._agreements)
    max_width = self._tree.tangle_matrix(min_test_agreement).shape[0]
    sweep.greedy_search(max_width, min_test_agreement)
    assert is_tst_valid(sweep.tree, feat_sys.is_le, agreement_func(feat_sys), min_test_agreement)