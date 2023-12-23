import numpy as np

def random_features(num_features: int, num_elements: int, probability: float = 0.5) -> np.ndarray:
    """
    Generates an array of features randomly. For each feature, each element of the groundset is independently
    chosen to be contained within the feature with the given probability.

    Parameters
    ----------
    num_features : int
        The number of features that will be generate.
    num_elements : int
        The size of the ground set from which elements are chosen for each feature.
    probability : float
        The probability of a single element to be contained within a feature.

    Returns
    -------
    np.ndarray
        An array of features.
    """
    
    return np.random.choice([1, -1], size=(num_elements, num_features), p=[probability, 1-probability])