from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np
from tangles.analysis import tangle_score


class TangleSearchWidget(ABC):

    @property
    @abstractmethod
    def sep_sys(self):
        """
        the separation system (feature system) used by this widget
        """

    @property
    @abstractmethod
    def tree(self):
        """
        the tangle search tree used by this widget
        """

    @property
    @abstractmethod
    def search_object(self):
        """
        the low level search object used by this search object
        """

    @property
    @abstractmethod
    def original_feature_ids(self) -> Union[list, np.ndarray]:
        """
        A list of ids of separations/features that were appended to the tree (without corners)
        """

    @property
    @abstractmethod
    def all_oriented_feature_ids(self) -> Union[list, np.ndarray]:
        """
        A list of ids of separations/features that were appended to the tree (including corners)
        """

    @abstractmethod
    def oriented_feature_ids_for_agreement(
        self, agreement: int
    ) -> Union[list, np.ndarray]:
        """
        A list of ids of separations/features that could be oriented at the given agreement level
        """

    @abstractmethod
    def tangle_matrix(
        self, min_agreement: Optional[int] = None, only_initial_seps: bool = True
    ):
        """Return the tangle matrix that describes all maximal tangles of at least the specified agreement.

        Guaranteed to return every tangle (on the set of separation ids the sweep knows about) if the limit of the
        :class:`~tangles.search._tree.TangleSearchTree` is below the specified `min_agreement`.

        Parameters
        ----------
        min_agreement : int
            All tangles of at least this agreement value are returned.
        only_inital_seps : bool
            whether to include auxiliary features into the tangle matrix

        Returns
        -------
        np.ndarray
            Tangle matrix.
        """

    @abstractmethod
    def lower_agreement(self, min_agreement: int, progress_callback=None):
        """
        Extend nodes in the tangle search tree until the agreement search limit has decreased below the
        specified agreement value.

        This method just forwards to :meth:`tangles.TangleSweep.sweep_below`.

        Parameter
        ---------
        min_agreement : int
            The new agreement search limit.
        """

    def tangle_score(self, min_agreement: int, normalize: str = "none") -> np.ndarray:
        """
        compute the tangles scores for given agreement

        (forwards to :meth:`tangles.analysis.tangle_score`)

        Parameter
        ---------
        min_agreement : int
           The new agreement search limit.
        normalize: str
            one of 'none', 'rows', 'cols', 'both'

        Returns
        -------
        np.ndarray
            Tangle score matrix M, where M(i,j) is the score of data point i for tangle j
        """
        if min_agreement is None:
            min_agreement = self.tree.limit
        mat = self.tangle_matrix(min_agreement, only_initial_seps=False)
        ids = self.oriented_feature_ids_for_agreement(min_agreement)[: mat.shape[1]]
        return tangle_score(
            mat,
            ids,
            self.sep_sys,
            normalize_rows=normalize in ("both", "rows"),
            normalize_cols=normalize in ("both", "cols"),
        )
