from abc import ABC, abstractmethod
import numpy as np
from tangles._typing import AgreementFunc, OrientedSep
from tangles._tangle import Tangle
from tangles.separations import SetSeparationSystem, FeatureSystem, SetSeparationSystemBase
from tangles.util._subsets import all_subsets
from typing import Union

def memoize_agreement_func(agreement_func: AgreementFunc) -> AgreementFunc:
    cache: dict[tuple[int, ...], int] = {}

    def agreement(sep_ids: Union[np.ndarray, list, tuple], orientations: Union[np.ndarray, list, tuple]) -> int:
        key = tuple(sep_ids) + tuple(orientations)
        good = cache.get(key)
        if good is None:
            good = agreement_func(sep_ids, orientations)
            cache[key] = good
        return good

    agreement.max_value = agreement_func.max_value
    return agreement

class AgreementFunc(ABC):
    def __init__(self, max_value: int):
        self.tuple_cache: dict[tuple[int, ...], np.ndarray] = {}
        self.max_value = max_value

    def __call__(self, sep_ids: Union[np.ndarray, list, tuple], orientation: Union[np.ndarray, list, tuple]) -> int:
        return self.agreement(sep_ids, orientation)

    @abstractmethod
    def agreement(self, sep_ids: Union[np.ndarray, list, tuple], orientation: Union[np.ndarray, list, tuple]) -> int:
        pass

class AgreementFuncBitarray(AgreementFunc):
    def __init__(self, sep_sys: SetSeparationSystem, memoized=False):
        super().__init__(sep_sys.datasize)
        self.sep_sys = sep_sys
        self.agreement = memoize_agreement_func(self._compute_agreement) if memoized else self._compute_agreement

    def agreement(self, sep_ids: Union[np.ndarray, list, tuple], orientation: Union[np.ndarray, list, tuple]) -> int:
        pass

    def _compute_agreement(self, sep_ids: Union[np.ndarray, list, tuple], orientations: Union[np.ndarray, list, tuple]) -> int:
        small_side_union = (self.sep_sys.seps_ba[1][sep_ids[0]] if orientations[0] > 0 else self.sep_sys.seps_ba[0][sep_ids[0]]).copy()
        for i in range(1, len(sep_ids)):
            small_side_union |= self.sep_sys.seps_ba[1][sep_ids[i]] if orientations[i] > 0 else self.sep_sys.seps_ba[0][sep_ids[i]]
        return len(small_side_union) - small_side_union.count()

class AgreementFuncBitarrayBips(AgreementFunc):
    def __init__(self, sep_sys: FeatureSystem, memoized=False):
        super().__init__(sep_sys.datasize)
        self.sep_sys = sep_sys
        self.agreement = memoize_agreement_func(self._compute_agreement) if memoized else self._compute_agreement

    def agreement(self, sep_ids: Union[np.ndarray, list, tuple], orientation: Union[np.ndarray, list, tuple]) -> int:
        pass

    def _compute_agreement(self, sep_ids: Union[np.ndarray, list, tuple], orientations: Union[np.ndarray, list, tuple]) -> int:
        intersection = (self.sep_sys.seps_ba[sep_ids[0]] if orientations[0] > 0 else ~self.sep_sys.seps_ba[sep_ids[0]]).copy()
        for i in range(1, len(sep_ids)):
            intersection &= self.sep_sys.seps_ba[sep_ids[i]] if orientations[i] > 0 else ~self.sep_sys.seps_ba[sep_ids[i]]
        return intersection.count()

def agreement_func(sep_sys: SetSeparationSystemBase, **options) -> AgreementFunc:
    """
    Return an agreement function that calculates the agreement value for separations in the given separation system.
    
    Parameters
    ----------
    sep_sys
        The separation system in which to look up the sep ids.

    Returns
    -------
    agreement_func
        A function to calculate the agreement value for a tuple of separations.
    """

    if isinstance(sep_sys, SetSeparationSystem):
        return AgreementFuncBitarray(sep_sys, **options)
    if isinstance(sep_sys, FeatureSystem):
        return AgreementFuncBitarrayBips(sep_sys, **options)

    raise Exception(f"agreement_func does not support {type(sep_sys)}")

class CalculateAgreement:
    def __init__(self,
                 agreement_func: AgreementFunc,
                 forbidden_tuple_size: int):
        self._agreement_func = agreement_func
        self._forbidden_tuple_size = forbidden_tuple_size

    def _calculate_subset_agreements(self,
                                     subsets: np.ndarray,
                                     inclusions: np.ndarray,
                                     tail: np.ndarray) -> np.ndarray:
        subset_size = subsets.shape[1]
        sep_ids = np.empty(subset_size + len(tail), dtype=int)
        sep_orientations = np.empty(subset_size + len(tail), dtype=np.int8)
        sep_ids[subset_size:] = tail[:, 0]
        sep_orientations[subset_size:] = tail[:, 1]
        agreements = np.empty(len(subsets), dtype=int)
        for i, subset in enumerate(subsets):
            sep_ids[:subset_size] = subset[:, 0]
            sep_orientations[:subset_size] = subset[:, 1]
            agreements[i] = self._agreement_func(sep_ids, sep_orientations)
        core_agreements = inclusions.astype(int) * (agreements[:, np.newaxis]+1)-1
        core_agreements[core_agreements == -1] = np.iinfo(int).max
        return np.min(core_agreements, axis=0)

    def _lower_agreement_value(self, tangle: Tangle, new_value: float):
        tangle.agreement = min(new_value, tangle.agreement)

    def __call__(self, tangles: list[Tangle], add_seps: set[OrientedSep]):
        tail = np.empty((len(add_seps), 2), dtype=int)
        tail[:, :] = list(add_seps)
        core_agreement = np.empty(len(tangles))
        for i, tangle in enumerate(tangles):
            if tangle.parent:
                core_agreement[i] = tangle.parent.agreement
            else:
                core_agreement[i] = np.iinfo(int).max

        for subset_size in range(0, self._forbidden_tuple_size - len(add_seps)+1):
            subsets, inclusions = all_subsets([tangle.core for tangle in tangles], subset_size)
            new_core_agreement =  self._calculate_subset_agreements(subsets, inclusions, tail)
            core_agreement = np.minimum(core_agreement, new_core_agreement)

        for i, tangle in enumerate(tangles):
            self._lower_agreement_value(tangle, core_agreement[i])