import numpy as np
import itertools

def all_subsets(sets: list[set], subset_size:int) -> tuple[np.ndarray, np.ndarray]:
    subset_occurrence = _create_subset_dictionary(sets, subset_size)
    number_of_subsets = len(subset_occurrence)

    if number_of_subsets == 0:
        return np.empty((1, 0, 2), dtype=int), np.ones((1, len(sets)), dtype=bool)

    subsets = np.empty((number_of_subsets,subset_size,2), dtype=int)
    subsets[:,:,:] = np.array(list(subset_occurrence.keys()))

    inclusion = np.empty((number_of_subsets, len(sets)), dtype=bool)
    inclusion[:,:] = np.array(list(subset_occurrence.values()))

    return subsets, inclusion


def _create_subset_dictionary(core_list:list[set], subset_size:int) -> dict:
    if subset_size == 0:
        return {}
    subset_occurrence = {}
    for i,core in enumerate(core_list):
        core_sorted = sorted(core)
        for subset in itertools.combinations(core_sorted, subset_size):
            lies_in_core = subset_occurrence.get(subset)
            if lies_in_core is None:
                lies_in_core = np.zeros(len(core_list), dtype=bool)
                lies_in_core[i] = True
                subset_occurrence[subset] = lies_in_core
            else:
                lies_in_core[i] = True
    return subset_occurrence
