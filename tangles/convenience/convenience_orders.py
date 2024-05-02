from typing import Union

import numpy as np
import scipy as sp
import pandas as pd

from tangles._typing import SetSeparationOrderFunction
from tangles.util.graph.cut_weight import CutWeightOrder, RatioCutOrder, NCutOrder
from tangles.util.matrix_order import matrix_order

# TODO: it might not be the perfect place for these order function classes...


class OrderFunctionO1:
    """
    Order function "O1" from the book
    """

    def __init__(self, features: np.ndarray):
        self.features_pos = features >= 0
        self.features_neg = features <= 0

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        orders = np.empty(features.shape[1])
        for s in range(features.shape[1]):
            orders[s] = (
                self.features_pos[features[:, s] >= 0, :].sum(axis=0)
                * self.features_pos[features[:, s] <= 0, :].sum(axis=0)
            ).sum() + (
                self.features_neg[features[:, s] >= 0, :].sum(axis=0)
                * self.features_neg[features[:, s] <= 0, :].sum(axis=0)
            ).sum()
        return orders


class OrderFunctionO1biased:
    """
    Biased version of order function "O1" from the book
    """

    def __init__(self, features: np.ndarray):
        self.features_pos = features >= 0

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        orders = np.empty(features.shape[1])
        for s in range(features.shape[1]):
            orders[s] = (
                self.features_pos[features[:, s] >= 0, :].sum(axis=0)
                * self.features_pos[features[:, s] <= 0, :].sum(axis=0)
            ).sum()
        return orders


class OrderFunctionO2:
    """
    Order function "O2" from the book
    """

    def __init__(self, features: np.ndarray):
        self.features_pos = features >= 0
        self.features_neg = features <= 0

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        orders = np.empty(features.shape[1])
        for s in range(features.shape[1]):
            orders[s] = (
                np.minimum(
                    self.features_pos[features[:, s] >= 0, :].sum(axis=0),
                    self.features_pos[features[:, s] <= 0, :].sum(axis=0),
                ).sum()
                + np.minimum(
                    self.features_neg[features[:, s] >= 0, :].sum(axis=0),
                    self.features_neg[features[:, s] <= 0, :].sum(axis=0),
                ).sum()
            )
        return orders


class OrderFunctionO3:
    """
    Order function "O3" from the book
    """

    def __init__(self, features: np.ndarray):
        self.features = features >= 0

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        orders = np.empty(features.shape[1])
        for s in range(features.shape[1]):
            orders[s] = np.minimum(
                self.features[features[:, s] >= 0, :].sum(axis=0),
                self.features[features[:, s] <= 0, :].sum(axis=0),
            ).sum()
        return orders


class OrderFunctionO4:
    """
    Order function "O4" from the book
    """

    def __init__(self, sigma: Union[np.ndarray, sp.sparse.spmatrix]):
        self.mat = -sigma
        if isinstance(self.mat, np.ndarray):
            np.fill_diagonal(self.mat, 0)
        elif isinstance(self.mat, sp.sparse.spmatrix):
            if (self.mat.diagonal() != 0).any():
                self.mat.setdiag(0)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        return 0.5 * matrix_order(self.mat, features)


def order_works_on_features(name: str) -> bool:
    """
    Helper function to find out if a named order function works on features

    Parameters
    ----------
    name: str
        The name of an order function, currently supported: "O1"-"O5" (see tangles-book), "O1-biased" (see tangles book),
                                                            "cut", "ratiocut", "normcut"

    Returns
    -------
    bool:
        True if the named order functino works with a matrix of features.
    """
    return name in {"O1", "O1-biased", "O2", "O3"}


def create_order_function(
    name: str, mat: Union[np.ndarray, pd.DataFrame]
) -> SetSeparationOrderFunction:
    """
    Create a standard order function given by name.

    Parameters
    ----------
    name: str
        The name of an order function, currently supported: "O1"-"O5" (see tangles-book), "O1-biased" (see tangles book),
                                                            "cut", "ratiocut", "normcut"
    mat: np.ndarray
        Data used to compute the order. This is either a set of features/advanced features or a similarity matrix (depending on the type of order function, see :order_works_on_features)

    Returns
    -------
    SetSeparationOrderFunction:
        a callable implementing the chosen order function
    """
    if name == "O1":
        return OrderFunctionO1(mat)
    if name == "O1-biased":
        return OrderFunctionO1biased(mat)
    if name == "O2":
        return OrderFunctionO2(mat)
    if name == "O3":
        return OrderFunctionO3(mat)
    if name in {"O4", "sim_O4", "O6", "sim_O6"}:
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"The order function {name} needs a similarity matrix")
        return OrderFunctionO4(mat)
    if name in {"cut", "sim_cut", "O5"}:
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"The order function {name} needs a similarity matrix")
        return CutWeightOrder(mat)
    if name in {"ratiocut", "sim_ratiocut"}:
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"The order function {name} needs a similarity matrix")
        return RatioCutOrder(mat)
    if name in {"normcut", "sim_normcut"}:
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"The order function {name} needs a similarity matrix")
        return NCutOrder(mat)
    # TODO: need more order functions...

    raise ValueError(f"unknown order function: {name}")
