import pytest
import numpy as np
from tangles.guiding import guided_tangle, get_tangle_by_path, is_guiding
from tangles import Tangle
from tangles.search import TangleSearchTree
from tangles.tests._test_data.tst_examples import fruit, build_random

Subset = np.ndarray
GuidedTestCase = dict

######### TEST GUIDED TANGLE ##########

def subsets() -> dict[str, Subset]:
    return {
        'empty': np.zeros(6, dtype=bool),
        'citrus': np.array([False, True, True, True, False, False]),
        'tropical_apple_orange': np.array([True, True, False, False, True, True]),
        'all_elements': np.ones(6, dtype=bool)
    }

def agreements() -> list[int]:
    return [1, 3]

def subset_names() -> list[str]:
    return subsets().keys()

def expected_tangles(tangles: dict[str, Tangle]) -> dict[tuple[str, int], Tangle]:
    return {
        ('empty', 1): tangles['root'],
        ('empty', 3): tangles['root'],
        ('all_elements', 1): tangles['lemon'],
        ('all_elements', 3): tangles['citrus_forced'],
        ('tropical_apple_orange', 1): tangles['root'],
        ('tropical_apple_orange', 3): tangles['root'],
        ('citrus', 1): tangles['orange'],
        ('citrus', 3): tangles['citrus_forced']
    }

@pytest.fixture
def guided_test_case(fruit) -> GuidedTestCase:
    return {
        'tree': fruit['tangle_search_tree'],
        'subsets': subsets(),
        'expected': expected_tangles(fruit['tangles']),
        'feat_sys': fruit['feat_sys']
    }

@pytest.mark.parametrize('agreement', agreements())
@pytest.mark.parametrize('subset', subset_names())
def test_guided_tangle(guided_test_case: GuidedTestCase, subset: str, agreement: int):
    assert guided_tangle(guided_test_case['tree'],
                         guided_test_case['feat_sys'],
                         guided_test_case['subsets'][subset],
                         min_agreement=agreement) == guided_test_case['expected'][(subset, agreement)]

@pytest.mark.parametrize('agreement', [1])
@pytest.mark.parametrize('subset', subset_names())
def test_is_guiding(guided_test_case: GuidedTestCase, subset: str, agreement: int):
    tangle = guided_test_case['expected'][(subset, agreement)]
    subset_vector = guided_test_case['subsets'][subset]
    feature_system = guided_test_case['feat_sys']
    assert is_guiding(subset_vector, tangle, feature_system)
    if tangle.parent:
        assert is_guiding(subset_vector, tangle.parent, feature_system)
    if tangle.left_child:
        assert not is_guiding(subset_vector, tangle.left_child, feature_system)
    if tangle.right_child:
        assert not is_guiding(subset_vector, tangle.right_child, feature_system)

########### TEST GET TANGLE BY PATH ##########

def _test_subpaths(tree: TangleSearchTree, path: np.ndarray):
    test_path = np.zeros(path.shape, dtype=int)
    found_leaf_on_path = None
    for i in range(len(path)):
        node = get_tangle_by_path(tree, test_path)
        if not found_leaf_on_path:
            path_from_root = node.path_from_root_indicator()
            path_from_root = path_from_root + [0] * (len(test_path) - len(path_from_root))
            assert np.all(path_from_root == test_path)
            if node.is_leaf():
                found_leaf_on_path = node
        else:
            assert node == found_leaf_on_path
        test_path[i] = path[i]

def _get_path_from_root(tangle: Tangle) -> list[Tangle]:
    node = tangle
    path = [node]
    while node.parent:
        node = node.parent
        path.append(node)
    path.reverse()
    return path

def _test_agreements_along_path(tree: TangleSearchTree, path: np.ndarray):
    tangles_along_path = _get_path_from_root(get_tangle_by_path(tree, path))
    agreements_along_path = [tangle.agreement for tangle in tangles_along_path] + [0]
    for i in range(len(tangles_along_path)):
        agreement_must_be_at_most = agreements_along_path[i]
        agreement_must_be_greater_than = agreements_along_path[i+1]
        for agreement in range(agreement_must_be_greater_than+1, agreement_must_be_at_most+1):
            assert get_tangle_by_path(tree, path, min_agreement=agreement) == tangles_along_path[i]

def _test_path(tree: TangleSearchTree, path: np.ndarray):
    _test_subpaths(tree, path)
    _test_agreements_along_path(tree, path)

def test_get_tangle_by_path():
    tree = build_random(200, 10)
    num_tests = 100
    for _ in range(num_tests):
        path = np.random.choice([-1, 1], size=(10,))
        _test_path(tree, path)