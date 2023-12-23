import numpy as np
from abc import ABC

class OrderFuncDerivative(ABC):
    """
    Abstract Base Class for use with the minimize cut method.

    Must overwrite method discrete derivative.
    """

    def discrete_derivative(self, feature: np.ndarray) -> np.ndarray:
        """
        Returns an array, the same shape as the feature, containing how the order of the feature
        would change if the corresponding value in the feature was flipped.

        This method is abstract and needs to be overwritten.
        """
        pass

    def change_discrete_derivative(self, feat: np.ndarray, derivative: np.ndarray, change_index: int) -> np.ndarray:
        """
        This method does not need to be overwritten but it could help performance to overwrite it.
        """
        new_sep = feat.copy()
        new_sep[change_index] = -new_sep[change_index]
        return self.discrete_derivative(new_sep)

def minimize_cut(starting_feature: np.ndarray, order_derivative: OrderFuncDerivative, max_steps: int = int(1e8)) -> np.ndarray:
    """Find a locally minimal cut in a graph starting with the cut specified by `starting_feature`.

    Parameters
    ----------
    starting_feature : np.ndarray
        -1/1-indicator vector of an initial cut to start the local search with.
    order_derivative: OrderFuncDerivative
        A function that calculates the discrete derivative (and optionally for better performance the
        change of the discrete derivative) for finding local minima.
    max_steps : int, optional
        The maximal number of optimization steps.

    Returns
    -------
    np.ndarray
        A -1/1-indicator vector of the found locally minimal cut.
    """

    feature = starting_feature.copy()
    derivative = order_derivative.discrete_derivative(feature)

    for _ in range(max_steps):
        best_change = np.argmin(derivative)
        if derivative[best_change] >= 0:
            break
        derivative = order_derivative.change_discrete_derivative(feature, derivative, best_change)
        feature[best_change] = -feature[best_change]
    return feature
