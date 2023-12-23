import pytest
import numpy as np
from tangles.util.preprocessing import (
    standardize,
    balance,
    normalize_length
)

@pytest.fixture
def standardize_data() -> np.ndarray:
    return np.array([
        [1, 656],
        [3, 100]
    ])

@pytest.fixture
def standardized_result() -> np.ndarray:
    return np.array([
        [-1, 1],
        [1, -1],
    ], dtype=float)

def test_standardize_axis_0(standardize_data: np.ndarray, standardized_result: np.ndarray):
    assert np.all(standardize(standardize_data, axis=0) == standardized_result)

def test_standardize_axis_1(standardize_data: np.ndarray, standardized_result: np.ndarray):
    assert np.all(standardize(standardize_data.T, axis=1) == standardized_result.T)

@pytest.fixture
def standardize_data_std_0() -> np.ndarray:
    return np.array([
        [1, 1],
        [3, 1]
    ])

@pytest.fixture
def standardized_result_std_0() -> np.ndarray:
    return np.array([
        [-1, 0],
        [1, 0],
    ], dtype=float)

def test_standardize_removes_equal_values(standardize_data_std_0: np.ndarray, standardized_result_std_0: np.ndarray):
    assert np.all(standardize(standardize_data_std_0.T, axis=1) == standardized_result_std_0.T)

@pytest.fixture
def balance_data() -> np.ndarray:
    return np.array([
        [0, -1],
        [1, 1],
        [100, -1],
    ])

@pytest.fixture
def balance_result_axis_0() -> np.ndarray:
    return np.array([
        [-1, 0],
        [0, 2],
        [99, 0],
    ], dtype=float)

@pytest.fixture
def balance_result_axis_1() -> np.ndarray:
    return np.array([
        [0.5, -0.5],
        [0, 0],
        [50.5, -50.5],
    ], dtype=float)

def test_balance_axis_0(balance_data: np.ndarray, balance_result_axis_0: np.ndarray):
    assert np.all(balance(balance_data, axis=0) == balance_result_axis_0)

def test_balance_axis_1(balance_data: np.ndarray, balance_result_axis_1: np.ndarray):
    assert np.all(balance(balance_data, axis=1) == balance_result_axis_1)

@pytest.fixture
def normalize_data() -> np.ndarray:
    return np.array([
        [0, 1],
        [1, 1],
    ])

@pytest.fixture
def normalize_result() -> np.ndarray:
    return np.array([
        [0, 1/np.sqrt(2)],
        [1, 1/np.sqrt(2)]
    ], dtype=float)

def test_normalize_axis_0(normalize_data: np.ndarray, normalize_result: np.ndarray):
    assert np.all(normalize_length(normalize_data, axis=0) == normalize_result)

def test_normalize_axis_1(normalize_data: np.ndarray, normalize_result: np.ndarray):
    assert np.all(normalize_length(normalize_data.T, axis=1) == normalize_result.T)
