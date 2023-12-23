import numpy as np
from tangles.separations.system import SetSeparationSystemOrderFunc
from ._sweep import TangleSweep
from typing import Optional, Union

import tangles.search.progress as tsp

def uncross_distinguishers(search: TangleSweep,
                           sys_ord: SetSeparationSystemOrderFunc,
                           agreement: int,
                           verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Uncross the efficient distinguishers of tangles of at least the specified `agreement` value.

    This is done by adding in suitable corners to the separation system and the tangle search tree.
    The corners we chose are corners of lower order than one of the previous efficient distinguishers which
    also distinguish the tangles distinguished by that distinguisher. Thus the previous efficient
    distinguisher is replaced by the new, more nested, corner.

    For this to work we need to have two requirements:

    1. the order function must be submodular and injective, otherwise there might not exist a corner
       to better efficiently distinguish tangles in the uncrossing step.
    2. the agreement function must be such that no tangle contains an empty star: a triple of oriented separations
       such that the tangle contains two of those oriented separations but also the inverse of their infimum.

    Parameters
    ----------
    search : TangleSweep
        The search object we use to manage the tangle search.
    sys_ord : SetSeparationSystemOrderFunc
        todo: dont use sys ord
    agreement : int
        The minimum agreement value for tangles who's efficient distinguishers we attempt to uncross.
    verbose : bool
        If True, print the steps the program performs.

    Returns
    -------
    efficient_distinguisher_levels : np.ndarray
        The levels, in the tangle search tree, at which the efficient distinguishers appear.
    efficient_distinguisher_ids : np.ndarray
        The separation ids of the efficient distinguishers, sorted by order, the same way as the levels.
    """

    return EfficientDistinguisherUncrossing(search, sys_ord, agreement).uncross()


def _no_submodularity_error_message(sep_1: int, sep_2: int, corners: Union[list[int], np.ndarray]) -> str:
    return f"""
        We have encountered a critical error with your order function:
        The two separations which ids {sep_1} and {sep_2} have
        corners in {corners} where all but at most one corner have higher order than both seps.
    """


def _empty_star_error_message(sep_1: int, sep_2: int, corners: Union[list[int], np.ndarray]) -> str:
    return f"""
        There is a problem with your agreement function:
        There exists a tangle which contains an empty star containing
        {sep_1} and {sep_2} as well as a corner from {corners} oriented towards
        that tangles orientation of {sep_1} and {sep_2}.
        That's no good for trees of tangles.
    """


class EfficientDistinguisherUncrossing:
    def __init__(self, search: TangleSweep, sys_ord: SetSeparationSystemOrderFunc, agreement: int):
        self._search = search
        self._sys_ord = sys_ord
        self._sep_sys = sys_ord.sep_sys
        self._agreement = agreement
        self._known_ids = set(search.tree.sep_ids)

    def add_known_ids(self, ids):
        self._known_ids.update(ids)

    def find_first_cross(self) -> Optional[tuple[int, int]]:
        _, efficient_d_id = self._search.tree.get_efficient_distinguishers(return_ids=True, agreement=self._agreement)
        cross1, cross2 = self._sep_sys.find_first_cross(efficient_d_id)
        if cross1 is None:
            return None
        return cross1, cross2

    def uncross(self, progress_callback=None):
        num_corners_added = 0
        if progress_callback:
            progress_callback(tsp.PROGRESS_TYPE_SOMETHING_STARTING, info="uncrossing", sweep=self._search)
        while True:
            if (potential_cross := self.find_first_cross()) is None:
                break

            new_corners = self.uncrossing_step(potential_cross[0], potential_cross[1])
            if new_corners is None:
                break

            if progress_callback:
                num_corners_added += len(new_corners)
                progress_callback(tsp.PROGRESS_TYPE_UNCROSSING_RUNNING, uncrossing=self, num_corners_added=num_corners_added, sweep=self._search)

        if progress_callback:
            progress_callback(tsp.PROGRESS_TYPE_SOMETHING_FINISHED, info="uncrossing", sweep=self._search)
        return self._search.tree.get_efficient_distinguishers(return_ids=True, agreement=self._agreement)

    def uncrossing_step(self, sep_id_1: int, sep_id_2: int) -> Optional[list[int]]:
        corner_ids, _ = self._sep_sys.get_corners(sep_id_1,sep_id_2)

        chosen_corner_ids = self._choose_corner_ids(sep_id_1, sep_id_2, corner_ids)

        if len(chosen_corner_ids) < 2:
            print(_no_submodularity_error_message(sep_id_1, sep_id_2, corner_ids))
            return None

        filtered_chosen_corner_ids = [id for id in chosen_corner_ids if id not in self._known_ids]
        if not filtered_chosen_corner_ids:
            print(_empty_star_error_message(sep_id_1, sep_id_2, corner_ids))
            return None

        self._insert_into_search(filtered_chosen_corner_ids)
        return filtered_chosen_corner_ids

    def _choose_corner_ids(self, sep_1: int, sep_2: int, corner_ids: Union[list[int], np.ndarray]):
        max_order = max(self._sys_ord.injective_order_value([sep_1, sep_2]))
        return np.array(corner_ids)[self._sys_ord.injective_order_value(corner_ids) < max_order]

    def _insert_into_search(self, corner_ids: Union[list[int], np.ndarray]):
        for corner_id in corner_ids:
            self._known_ids.add(corner_id)
            self._search.insert_separation(self._sys_ord.get_insertion_index(self._search.tree.sep_ids, corner_id), corner_id, self._agreement)
