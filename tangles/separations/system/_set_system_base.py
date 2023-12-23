import numpy as np
from typing import Dict, Tuple, final, Sequence, List, Generator
from abc import ABC, abstractmethod

import warnings
import scipy.sparse as sp

from tangles.util.logic import append, simplify, distribute
from tangles._typing import OrientedSep

from typing import Union, Optional
import numbers

class MetaData:
    """
    Metadata for separations.

    Can be used, for example, to save what a separation intuitively means,
    what question in a questionnaire the separation was generated from and
    can be used to save whether it was generated as a corner of other separations.
    """

    def __init__(self, info, orientation: int = 0, type='custom'):
        """
        Build a piece of metadata for a separation.

        Metadata can also be associated to a certain orientation of a separation (specification of a feature).

        Parameters
        ----------
        info : Any
            The information to be saved to the separation.
        orientation : int, optional
            If the piece of information should be associated just to one
            orientation of the separation set this variable to the orientation.
            Defaults to 0.
        type : str
            The type of this metadata object. Available predefined values are 'custom' and 'inf'.
        """

        self.type = type
        self.info = info
        self.orientation = orientation
        self.next = None

    @staticmethod
    def from_dict(dcit: dict):
        m = MetaData(dcit["type"], dcit["orientation"], dcit["type"])
        next = dcit.get("next")
        if next:
            m.next = MetaData.from_dict(next)
        return m

    def to_dict(self):
        dcit = {"type":self.type, "info":self.info, "orientation":self.orientation}
        if self.next:
            dcit["next"] = self.next.todict()
        return dcit

    def append(self, metadata: 'MetaData'):
        """
        Append another piece of metadata to the same separation.

        Parameters
        ----------
        metadata : MetaData
            Another piece of metadata to be appended to the same separation.
        """

        if self.next is None:
            self.next = metadata
        else:
            n = self.next
            while n.next is not None:   # warning: this list can become huge!
                n = n.next
            n.next = metadata

    def tail_as_list(self):
        tail = [self]
        while (d:=tail[-1].next): tail.append(d)
        return tail

    def tail_as_gen(self):
        metadata = self
        while metadata:
            yield metadata
            metadata = metadata.next


class SetSeparationSystemBase(ABC):
    def __init__(self, datasize):
        self.datasize = datasize
        self.le_cache: Dict[tuple[int, ...], int] = {}
        self.sup_cache: Dict[tuple[int, ...], Tuple[int, int]] = {}

        self.sep_metadata: List[MetaData] = []

    @classmethod
    def with_array(cls, S: np.ndarray, return_sep_info: bool = False, metadata=None) -> Union[Tuple['SetSeparationSystemBase', np.ndarray], 'SetSeparationSystemBase']:
        """
        Create a new SetSeparationSystem from a separation matrix.

        Parameters
        ----------
        S : np.ndarray
            Matrix of shape (number of points, number of separations) representing the separations.
        return_sep_info : bool, optional
            Whether to return the ids and orientations of the incoming separations as they will appear in the separation system.
        metadata : arraylike, optional
            An optional piece of metadata. Should be an arraylike of length (number of separations).

        Returns
        -------
        SetSeparationSystem or tuple (SetSeparationSystem, np.ndarray)
            The generated set separation system or, if `return_sep_info` is set to True,
            a tuple additionally containing information about the ids and orientations of the input separations.
        """

        sep_sys = cls(S.shape[0])
        sep_ids = sep_sys.add_seps(S, metadata=metadata)
        return (sep_sys, sep_ids) if return_sep_info else sep_sys

    @classmethod
    def with_sparse_array(cls, S, return_sep_info: bool = False, metadata=None) -> Union[Tuple['SetSeparationSystemBase', Tuple[np.ndarray, np.ndarray]], 'SetSeparationSystemBase']:
        """
        Create a new SetSeparationSystem from a sparse separation matrix.

        Parameters
        ----------
        S : scipy.sparse.csc_matrix
            Sparse matrix of shape (number of points, number of separations) representing the separations.
        return_sep_info : bool, optional
            Whether to return the ids and orientations of the incoming separations as they will appear in the separation system.
        metadata : arraylike, optional
            An optional piece of metadata. Should be an arraylike of length (number of separations).

        Returns
        -------
        SetSeparationSystem or tuple (SetSeparationSystem, np.ndarray)
            The generated set separation system or, if `return_sep_info` is set to True,
            a tuple additionally containing information about the ids and orientations of the input separations.
        """

        sep_sys = cls(S.shape[0])
        if not isinstance(S, sp.csc_matrix):
            warnings.warn("converting to  csc format...")
            S = sp.csc_matrix(S)

        tmp = np.empty((S.shape[0],1), dtype=np.int8)
        ids, orientations = np.empty(S.shape[1], dtype=int), np.empty(S.shape[1], dtype=np.int8)
        for i in range(S.shape[1]):
            tmp.fill(-1)
            tmp[S.indices[S.indptr[i]:S.indptr[i+1]],0] = 1
            ids[i], orientations[i] = sep_sys.add_seps(tmp, None if metadata is None else metadata[i])
        return (sep_sys, (ids, orientations)) if return_sep_info else sep_sys

    @abstractmethod
    def __getitem__(self, sep_ids: Union[int, np.ndarray]) -> np.ndarray:
        """
        Access the data of the separation or separations.

        Parameters
        ----------
        sep_ids : int or np.ndarray
            The separation id or list of separation ids to access.

        Returns
        -------
        np.ndarray
            Matrix of shape (number of datapoints, number of separations) containing the information of the separation.
        """

        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Number of separations in the separation system.
        """

        pass

    def all_sep_ids(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The ids of all currently stored separations.
        """

        return np.arange(len(self))

    @abstractmethod
    def get_sep_ids(self, seps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        If the separations are already contained inside of the separation system then return the
        id and orientation of the sep. Otherwise -1 is returned.

        Parameters
        ----------
        seps : np.ndarray
            A matrix of shape (number of datapoints, number of separations) encoding the separations.

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            The separation ids and orientations.
        """

        pass

    @abstractmethod
    def _add_sep(self, new_seps: np.ndarray) -> Tuple[int, int]:
        pass

    @abstractmethod
    def _compute_le(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> bool:
        pass

    @abstractmethod
    def _compute_infimum_of_two(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> tuple[int, np.int8]:
        pass

    def is_nested(self, sep_id_1: int, sep_id_2: int) -> bool:
        """
        Checks whether two separations are nested.

        Note that being nested does not depend on the orientations of the separations.

        Parameters
        ----------
        sep_id_1 : int
            The id of the first separation.
        sep_id_2 : int
            The id of the second separation.

        Returns
        -------
        bool
            Whether the separations are nested.
        """

        return self.is_le(sep_id_1, 1, sep_id_2, 1) or self.is_le(sep_id_1, 1, sep_id_2, -1) or \
               self.is_le(sep_id_1, -1, sep_id_2, 1) or self.is_le(sep_id_1, -1, sep_id_2, -1)

    @final
    def separation_metadata(self, sep_ids: Union[int, list, np.ndarray, None] = None) -> Union[MetaData, list[MetaData]]:
        """
        Returns the metadata of the separation `sep_id`.

        Parameters
        ----------
        sep_id : int, list, np.ndarray or None
            The id(s) of the separation to get the metadata from.

        Returns
        -------
        MetaData
            A metadata object.
        """
        if sep_ids is None:
            return self.sep_metadata
        if isinstance(sep_ids, numbers.Integral):
            return self.sep_metadata[sep_ids]
        return [self.sep_metadata[sep_id] for sep_id in sep_ids]

    @final
    def add_seps(self, new_seps: np.ndarray, metadata=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add separations to the separation system.

        Parameters
        ----------
        new_seps : np.ndarray
            A matrix of shape (number of points, number of separations) representing the separations.
        metadata : arraylike, optional
            An optional piece of metadata. Should be an arraylike of length (number of separations).

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            The ids of the added separations in the separation system in its first entry
            and the orientations of the added separations in the separation system in the second entry.
        """

        if len(new_seps.shape) < 2:
            new_seps = new_seps[:, np.newaxis]
        if new_seps.shape[1] == 1:
            metadata = [metadata]
        elif metadata is None:
            metadata = [None]*new_seps.shape[1]

        ids, orientations = np.empty(new_seps.shape[1], dtype=int), np.empty(new_seps.shape[1], dtype=np.int8)
        for i in range(new_seps.shape[1]):
            ids[i], orientations[i] = self._add_sep(new_seps[:,i:i+1])
            if ids[i] < len(self.sep_metadata):
                self.sep_metadata[ids[i]].append(MetaData(metadata[i], orientations[i]))
            else:
                self.sep_metadata.append(MetaData(metadata[i], orientations[i]))
        return ids, orientations

    @abstractmethod
    def compute_infimum(self, sep_ids: np.ndarray, orientations: np.ndarray) -> np.ndarray:
        """
        Calculate the infimum of a list of separation ids and orientations. Used not to get separations but
        to see what lies in the intersection of a list of oriented separations. Hence the result is also not
        added into the separation system.

        Parameters
        ----------
        sep_ids : np.ndarray
            The ids of the separations to calculate the infimum from.
        orientations : np.ndarray
            The orientations of the separations.

        Returns
        -------
        np.ndarray
            A -1/1-indicator vector for whether a datapoint lies inside the infimum.
        """

        pass

    def infimum_of_two(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> Tuple[int, int]:
        """
        Calculate the infimum of two oriented separations from the separation system and add this infimum as a
        new separation to the separation system.

        Parameters
        ----------
        sep_id_a : int
            The id of the first separation.
        orientation_a : int
            The orientation of the first separation.
        sep_id_b : int
            The id of the second separation.
        orientation_b : int
            The orientation of the second separation.

        Returns
        -------
        tuple (int, int)
            Separation id of the infimum and orientation of the infimum.
        """

        key = (sep_id_a, orientation_a, sep_id_b, orientation_b) if sep_id_a < sep_id_b else (sep_id_b, orientation_b, sep_id_a, orientation_a)
        sup = self.sup_cache.get(key)
        if sup is None:
            sup = self._compute_infimum_of_two(*key)
            self.sup_cache[key] = sup
        return sup

    def get_corners(self, sep_id_a: int, sep_id_b: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the four corners of two separations in the separation system.

        The corners are added into the separation system as new separations.

        Parameters
        ----------
        sep_id_a : int
            The id of the first separation.
        sep_id_b : int
            The id of the second separation.

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            Tuple with the separation ids of the four separations in the first entry
            and the orientations of the four separations in the second entry.
        """

        sep_ids = np.empty(4, dtype=int)
        orientations = np.empty(4, dtype=np.int8)
        sep_ids[0], orientations[0] = self.infimum_of_two(sep_id_a, -1, sep_id_b, -1)
        sep_ids[1], orientations[1] = self.infimum_of_two(sep_id_a, 1, sep_id_b, -1)
        sep_ids[2], orientations[2] = self.infimum_of_two(sep_id_a, -1, sep_id_b, 1)
        sep_ids[3], orientations[3] = self.infimum_of_two(sep_id_a, 1, sep_id_b, 1)
        return sep_ids, orientations

    def is_le(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> bool:
        """
        Check if separation :math:`a` specified by `sep_id_a` is less than or equal to separation :math:`b` specified by `sep_id_b`,
        i.e. if :math:`a \le b`.

        Parameters
        ----------
        sep_id_a : int
            The id of the first separation.
        orientation_a : int
            The orientation of the first separation.
        sep_id_b : int
            The id of the second separation.
        orientation_b : int
            The orientation of the second separation.

        Returns
        -------
        bool
            Whether the first separation is less than or equal to the second separation.
        """

        return self._compute_le(sep_id_a, orientation_a, sep_id_b, orientation_b)

    @final
    def crossing_seps(self, sep_ids: Sequence[int]) -> Generator[Tuple[int, int], None, None]:
        """
        Generator to get all crossing separations from the list of separations.

        Parameters
        ----------
        sep_ids : sequence of int
            The ids of the separations that are checked for crossings.

        Returns
        -------
        tuple (int, int) or tuple (None, None)
            The ids of two crossing separation if there exists such a pair.
        """

        for i, sep_id1 in enumerate(sep_ids):
            for sep_id2 in sep_ids[i + 1:]:
                if not self.is_nested(sep_id1, sep_id2):
                    yield sep_id1, sep_id2

    @final
    def find_first_cross(self, sep_ids: Sequence[int]) -> Union[Tuple[int, int], Tuple[None, None]]:
        for i, sep_id1 in enumerate(sep_ids):
            for sep_id2 in sep_ids[i + 1:]:
                if not self.is_nested(sep_id1, sep_id2):
                    return sep_id1, sep_id2
        return None, None

    def metadata_matrix(self,
                        sep: OrientedSep,
                        data_list: list,
                        normal_form: 'str' = 'disjunctive',
                        _known_sep_matrices: dict[int, tuple] = None) -> np.ndarray:
        """
        Explain the meaning of a separation, generated by repeatedly taking corners of separations,
        by calculating a simplified logical term which explains the separation.

        Parameters
        ----------
        sep : OrientedSep
            The separation to explain.
        data_list : list
            The possible values of info_objects for separations. The rows of the matrix are also in this order.
        normal_form : {'disjunctive', 'conjunctive'}
            Disjunctive Normal Form (DNF) means that the result is a union of intersections.
            Conjunctive Normal Form (CNF) means that the result is an intersection of unions.
        _known_sep_matrices : dict
            Cache already computed results. This avoids computing the same stuff twice.

        Returns
        -------
        np.ndarray
            Matrix in CNF or DNF explaining the feature.
        """

        if _known_sep_matrices is None:
            _known_sep_matrices = {}

        if sep in _known_sep_matrices:
            return _known_sep_matrices[sep]

        metadata_list = self.separation_metadata(sep[0]).tail_as_list()
        metadata = next((meta for meta in metadata_list if meta.type == 'custom'), None) or \
                   next((meta for meta in metadata_list if meta.type == 'inf'), None) or \
                   metadata_list[0]
        info_object = metadata.info
        orientation = metadata.orientation*sep[1]
        my_matrix = None
        if metadata.type == 'inf':
            term1 = self.metadata_matrix((info_object[0][0], info_object[0][1]*orientation), data_list, normal_form, _known_sep_matrices)
            term2 = self.metadata_matrix((info_object[1][0], info_object[1][1]*orientation), data_list, normal_form, _known_sep_matrices)
            my_matrix = simplify(distribute(term1, term2)) if (orientation == 1 and normal_form == 'disjunctive') or \
                                                           (orientation == -1 and normal_form == 'conjunctive') else \
                        simplify(append(term1, term2))
        elif metadata.type == 'custom':
            my_matrix = np.zeros((len(data_list), 1), dtype=np.int8)
            my_matrix[np.where(data_list == info_object), 0] = orientation

        _known_sep_matrices[sep] = my_matrix
        return my_matrix

    def assemble_meta_info(self, sep_id: int, known_meta_info: dict[int, tuple] = None):
        """
        If the user has entered custom meta info for a separation, then that is prioritised before everything else.
        Otherwise, we check for the possibility of this separation merely being a corner of other separations,
        in that case the info_object becomes a four-tuple (orientation1, info_object_1, orientation2, info_object2),
        the implication being that our separation is the corner of those two separations (the seps being represented by the info object).
        If everything else fails, we simply return the first meta_info saved for the separation.

        Parameters
        ----------
        sep_id : int
            The id of the separation whose meta_info we are interested in.
        known_meta_info : dict
            Already computed meta info. Providing this avoids computing the same stuff twice.

        Returns
        -------
        tuple (int, any)
            Contains the specification in its first entry and the info object of the metadata in its second entry.
        """

        if known_meta_info is None:
            known_meta_info = {}

        if sep_id in known_meta_info:
            return known_meta_info[sep_id]

        metadata_list = self.separation_metadata(sep_id).tail_as_list()
        metadata = next((meta for meta in metadata_list if meta.type == 'custom'), None) or \
                   next((meta for meta in metadata_list if meta.type == 'inf'), None) or \
                   metadata_list[0]
        orientation = metadata.orientation
        info_object = metadata.info
        if metadata.type == 'inf':
            info_object = ((info_object[0][1], self.assemble_meta_info(info_object[0][0], known_meta_info=known_meta_info)),
                           (info_object[1][1], self.assemble_meta_info(info_object[1][0], known_meta_info=known_meta_info)))

        known_meta_info[sep_id] = orientation, info_object
        return orientation, info_object

