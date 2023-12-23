import bitarray as ba
import bitarray.util as ba_util
import numpy as np
from ._set_system_base import SetSeparationSystemBase, MetaData
from tangles._typing import Feature
from typing import Optional, Union
import copy as cp
import numbers


class BitarrayHashToIdMultimap:
    def __init__(self, n, hash_size, max_mask_size=0.66, seed=99, do_scramble=True):
        self.params = (n, hash_size, max_mask_size, seed, do_scramble)
        self.masks = []
        np.random.seed(seed)
        mask_size = max(1, min(int(n * max_mask_size), hash_size // 2))
        num_masks = max(1, hash_size // mask_size)
        for i in range(num_masks):
            mask = np.zeros(n, dtype=bool)
            mask[np.random.choice(n, mask_size, replace=False)] = 1
            self.masks.append(ba.frozenbitarray(mask.tolist()))

        self.hash_size = num_masks * mask_size
        self.hash = []
        for i in range(hash_size + 1):
            self.hash.append([])

        if do_scramble:
            mask = np.zeros(n, dtype=bool)
            mask[np.random.choice(n, n // 8, replace=False)] = 1
            self.scramble = ba.frozenbitarray(mask.tolist())
        else:
            self.scramble = None

    def copy(self) -> 'BitarrayHashToIdMultimap':
        copy = BitarrayHashToIdMultimap(*self.params)
        copy.hash = cp.deepcopy(self.hash)
        return copy

    def _key(self, sep_bit):
        k = sep_bit if self.scramble is None else sep_bit ^ self.scramble
        return sum(ba_util.count_and(k, m) for m in self.masks)

    def get(self, sep_bit):
        k = self._key(sep_bit)
        return self.hash[k], self.hash[self.hash_size - k]

    def add(self, sep_bit, sep_id):
        self.hash[self._key(sep_bit)].append(sep_id)

    def bucket_sizes(self):
        return np.array([len(b) for b in self.hash])

    def print(self):
        for b in self.hash:
            print(len(b), end=",")
        print()


class SetSeparationSystem(SetSeparationSystemBase):
    def __init__(self, data_size):
        super().__init__(data_size)
        self.seps_ba = ([], [])
        self.sep_hash_to_id = BitarrayHashToIdMultimap(data_size, 10000)

    def __len__(self) -> int:
        return len(self.seps_ba[0])

    def copy(self) -> 'SetSeparationSystem':
        copy = SetSeparationSystem(self.datasize)
        copy.seps_ba = (self.seps_ba[0].copy(), self.seps_ba[1].copy())
        copy.sep_hash_to_id = self.sep_hash_to_id.copy()
        return copy

    def get_single_sep_id_bit(self, sep_bit_a, sep_bit_b) -> tuple[int, np.int8]:
        a_match_pos, a_match_neg = self.sep_hash_to_id.get(sep_bit_a)
        b_match_pos, b_match_neg = self.sep_hash_to_id.get(~sep_bit_b)
        match_pos = set(a_match_pos) & set(b_match_pos)
        for s in match_pos:
            if (self.seps_ba[0][s] == sep_bit_a) and (self.seps_ba[1][s] == sep_bit_b):
                return s, np.int8(1)
        match_neg = set(a_match_neg) & set(b_match_neg)
        for s in match_neg:
            if (self.seps_ba[1][s] == sep_bit_a) and (self.seps_ba[0][s] == sep_bit_b):
                return s, np.int8(-1)
        return -1, np.int8(1)

    def get_sep_ids(self, seps) -> tuple[np.ndarray, np.ndarray]:
        if len(seps.shape) < 2:
            seps = seps[:, np.newaxis]
        n_seps = seps.shape[1]
        sep_ids = np.empty(n_seps, dtype=int)
        orientations = np.empty(n_seps, dtype=np.int8)
        for i in range(seps.shape[1]):
            s_a = ba.bitarray((seps[:, i] >= 0).tolist())
            s_b = ba.bitarray((seps[:, i] <= 0).tolist())
            sep_ids[i], orientations[i] = self.get_single_sep_id_bit(s_a, s_b)
        return sep_ids, orientations

    def compute_infimum(self, sep_ids: np.ndarray, orientations: np.ndarray) -> tuple[np.ndarray]:
        off = (orientations[0] + 1) >> 1
        sup_A = (self.seps_ba[1 - off][sep_ids[0]]).copy()
        sup_B = (self.seps_ba[off][sep_ids[0]]).copy()
        for i in range(1,len(sep_ids)):
            off = (orientations[i] + 1) >> 1
            sup_A &= self.seps_ba[1-off][sep_ids[i]]
            sup_B |= self.seps_ba[off][sep_ids[i]]
        sep = np.frombuffer(sup_A.unpack(), dtype=np.int8) - np.frombuffer(sup_B.unpack(), dtype=np.int8)
        return sep

    def _add_sep_bit(self, new_sep_bit_a, new_sep_bit_b) -> tuple[int, int]:
        sep_id, orientation = self.get_single_sep_id_bit(new_sep_bit_a, new_sep_bit_b)
        if sep_id == -1:
            sep_id = len(self.seps_ba[0])
            self.seps_ba[0].append(new_sep_bit_a)
            self.seps_ba[1].append(new_sep_bit_b)
            self.sep_hash_to_id.add(new_sep_bit_a, sep_id)
            self.sep_hash_to_id.add(~new_sep_bit_b, sep_id)
        return sep_id, orientation

    def _add_sep(self, new_sep: np.ndarray) -> tuple[int, int]:
        new_sep_bit_a = ba.bitarray((new_sep[:, 0] >= 0).tolist())
        new_sep_bit_b = ba.bitarray((new_sep[:, 0] <= 0).tolist())
        return self._add_sep_bit(new_sep_bit_a, new_sep_bit_b)

    def _compute_infimum_of_two(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> tuple[int, np.int8]:
        off_a = (orientation_a + 1) >> 1
        off_b = (orientation_b + 1) >> 1
        sup_A = (self.seps_ba[1-off_a][sep_id_a]) & (self.seps_ba[1-off_b][sep_id_b])
        sup_B = (self.seps_ba[off_a][sep_id_a]) | (self.seps_ba[off_b][sep_id_b])
        id, o = self.get_single_sep_id_bit(sup_A, sup_B)

        metadata = MetaData(((sep_id_a, orientation_a), (sep_id_b, orientation_b)), orientation=1, type='inf')
        if id == -1:
            id = len(self.seps_ba[0])
            self.seps_ba[0].append(sup_A)
            self.seps_ba[1].append(sup_B)
            self.sep_hash_to_id.add(sup_A, id)
            self.sep_hash_to_id.add(~sup_B, id)
            self.sep_metadata.append(metadata)
        elif id != sep_id_a and id != sep_id_b:
            metadata.orientation = o
            self.sep_metadata[id].append(metadata)

        return id, o


    def _compute_le(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> bool:
        off_a = (orientation_a + 1) >> 1
        off_b = (orientation_b + 1) >> 1
        a = (self.seps_ba[1-off_a][sep_id_a], self.seps_ba[off_a][sep_id_a])
        b = (self.seps_ba[1-off_b][sep_id_b], self.seps_ba[off_b][sep_id_b])
        return (a[0] | b[0]).count() == b[0].count() and (a[1] | b[1]).count() == a[1].count()

    def __getitem__(self, sep_ids) -> np.ndarray:
        if isinstance(sep_ids, numbers.Integral):
            seps = np.frombuffer(self.seps_ba[0][sep_ids].unpack(), dtype=np.int8) - np.frombuffer(self.seps_ba[1][sep_ids].unpack(), dtype=np.int8)
        elif isinstance(sep_ids, slice):
            start, stop, step = sep_ids.indices(len(self.seps_ba[0]))
            seps = np.empty((self.datasize, (stop - start) // step), dtype=np.int8)
            for i, id in enumerate(range(start, stop, step)):
                seps[:, i] = np.frombuffer(self.seps_ba[0][id].unpack(), dtype=np.int8) - np.frombuffer(self.seps_ba[1][id].unpack(), dtype=np.int8)
            return seps
        else:
            seps = np.empty((self.datasize, len(sep_ids)), dtype=np.int8)
            for i, id in enumerate(sep_ids):
                seps[:, i] = np.frombuffer(self.seps_ba[0][id].unpack(), dtype=np.int8) - np.frombuffer(self.seps_ba[1][id].unpack(), dtype=np.int8)
        return seps


class FeatureSystem(SetSeparationSystemBase):
    def __init__(self, data_size):
        super().__init__(data_size)
        self.seps_ba: list[ba.bitarray] = []
        self.sep_hash_to_id = BitarrayHashToIdMultimap(data_size, 10000)

    def all_feature_ids(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            The ids of all currently stored features.
        """

        return self.all_sep_ids()

    def is_nested(self, feature_id_1: int, feature_id_2: int) -> bool:
        """
        Check whether two features are nested. Note that inverting one, or both, features
        does not change whether they are nested or not.

        Parameters
        ----------
        feature_id_1 : int
            The id of the first feature.

        feature_id_2 : int
            The id of the second feature.

        Returns
        -------
        bool
            Whether the features are nested.
        """

        return super().is_nested(feature_id_1, feature_id_2)

    def feature_metadata(self, feature_ids: Union[int, list, np.ndarray, None] = None) -> MetaData:
        """
        Return a list of all metadata of the feature `feature_id`.

        Parameters
        ----------
        feature_ids : int, list, np.ndarray or None
            The id of the feature to get the metadata from.

        Returns
        -------
        MetaData
            A metadata object (i.e. a linked list of metadata objects).
        """

        return self.separation_metadata(feature_ids)

    def add_features(self, new_features: np.ndarray, metadata=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Add features to the feature system.

        Parameters
        ----------
        new_features : np.ndarray
             A matrix of shape (number of points, number of features) representing the features.
        metadata : arraylike, optional
            An optional piece of metadata. Should be an arraylike of length (number of features).

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            The ids of the added features in the feature system in its first entry
            and an np.ndarray in the second entry that is 1 if the new feature is contained in the feature system,
            or -1 if its inverse is contained in the feature system.
        """

        return self.add_seps(new_features, metadata)

    def infimum_of_two(self, feature_id_a: int, specification_a: int, feature_id_b: int, specification_b: int) -> tuple[int, int]:
        """
        Calculate the infimum of two features from the feature system and add this infimum as a
        new feature to the feature system.

        Parameters
        ----------
        feature_id_a : int
            The id of the first feature.
        specification_a : int
            Whether to take the feature (1) or its inverse (-1).
        feature_id_b : int
            The id of the second feature.
        specification_b : int
            Whether to take the feature (1) or its inverse (-1).

        Returns
        -------
        tuple (int, int)
            Feature id of the infimum and specification of the infimum.
        """

        return super().infimum_of_two(feature_id_a, specification_a, feature_id_b, specification_b)

    def get_corners(self, feature_id_a: int, feature_id_b: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the four corners of two features in the feature system.

        The corners are added into the feature system as new features.

        Parameters
        ----------
        feature_id_a : int
            The id of the first feature.
        feature_id_b : int
            The id of the second feature.

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            Tuple with the feature ids of the four features in the first entry and the specifications, whether the
            features or their inverses are contained in the feature system, in the second entry.
        """

        return super().get_corners(feature_id_a, feature_id_b)

    def is_le(self, feat_id_a: int, specification_a: int, feat_id_b: int, specification_b: int) -> bool:
        """
        Check if feature :math:`a` specified by `feat_id_a` is less than or equal to feature :math:`b` specified by `feat_id_b`,
        i.e. if :math:`a \le b`.

        If a feature is less than or equal another feature, then the inclusion of
        the first features in some specification prohibits the inverse of the latter features
        to be included in that specification.

        Parameters
        ----------
        feat_id_a : int
            The id of the first feature.
        specification_a : int
            Whether to take the feature (1) or its inverse (-1).
        feat_id_b : int
            The id of the second feature.
        specification_b : int
            Whether to take the feature (1) or its inverse (-1).

        Returns
        -------
        bool
            Whether the first feature is less than or equal to the second feature.
        """

        return super().is_le(feat_id_a, specification_a, feat_id_b, specification_b)

    def is_subset(self, feat_id_a: int, specification_a: int, feat_id_b: int, specification_b: int) -> bool:
        """
        Check if the feature `feat_id_a` is a subset of the feature `feat_id_b`.

        Parameters
        ----------
        feat_id_a : int
            The id of the first feature.
        specification_a : int
            Whether to take the feature (1) or its inverse (-1).
        feat_id_b : int
            The id of the second feature.
        specification_b : int
            Whether to take the feature (1) or its inverse (-1).

        Returns
        -------
        bool
            Whether the feature `feat_id_a` is a subset of the feature `feat_id_b`.
        """

        return self.is_le(feat_id_a, specification_a, feat_id_b, specification_b)



    def metadata_matrix(self,
                        feature: Feature,
                        data_list: list,
                        normal_form: 'str' = 'disjunctive',
                        _known_feature_matrices: dict[int, tuple] = None) -> np.ndarray:
        """
        Explain the meaning of a feature, generated by repeatedly taking corners of features,
        by calculating a simplified logical term which explains the feature.

        Parameters
        ----------
        feature : Feature
            The feature to explain.
        data_list : list
            The possible values of info_objects for separations. The rows of the matrix are also in this order.
        normal_form : {'disjunctive', 'conjunctive'}
            Disjunctive Normal Form (DNF) means that the result is a union of intersections.
            Conjunctive Normal Form (CNF) means that the result is an intersection of unions.
        _known_feature_matrices : dict
            Cache already computed results. This avoids computing the same stuff twice.

        Returns
        -------
        np.ndarray
            Matrix in CNF or DNF explaining the feature.
        """

        return super().metadata_matrix(feature, data_list, normal_form, _known_feature_matrices)


    def assemble_meta_info(self, feat_id: int, known_meta_info: dict[int, tuple] = None):
        """
        If the user has entered custom meta info for a feature, then that is prioritised before everything else.
        Otherwise, we check for the possibility of this feature merely being a corner of other features,
        in that case the info_object becomes a four-tuple (specification1, info_object_1, specification2, info_object2),
        the implication being that our separation is the corner of those two features (represented by their info object).
        If everything else fails, we simply return the first meta_info saved for the feature.

        Parameters
        ----------
        feat_id : int
            The id of the feature whose meta_info we are interested in.
        known_meta_info : dict
            Already computed meta info. Providing this avoids computing the same stuff twice.

        Returns
        -------
        tuple (int, any)
            Contains the specification in its first entry and the info object of the metadata in its second entry.
        """

        return super().assemble_meta_info(feat_id, known_meta_info)

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Number of features in the feature system.
        """

        return len(self.seps_ba)

    def copy(self) -> 'FeatureSystem':
        copy = FeatureSystem(self.datasize)
        copy.seps_ba = self.seps_ba.copy()
        copy.sep_hash_to_id = self.sep_hash_to_id.copy()
        return copy

    def _get_single_sep_id_bit(self, sep_bit) -> tuple[int, np.int8]:
        match_pos, match_neg = self.sep_hash_to_id.get(sep_bit)
        for s in match_pos:
            if self.seps_ba[s] == sep_bit:
                return s, np.int8(1)
        for s in match_neg:
            if self.seps_ba[s] == ~sep_bit:
                return s, np.int8(-1)
        return -1, np.int8(1)

    def get_sep_ids(self, seps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.get_feature_ids(seps)

    def get_feature_ids(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        If the features or their inverses are already contained inside of the feature system then return the
        id and specification of the feature. Otherwise -1 is returned.

        Parameters
        ----------
        features : np.ndarray
            A matrix of shape (number of datapoints, number of features) encoding the features.

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            The feature ids and specifications.
        """

        if len(features.shape) < 2:
            features = features[:, np.newaxis]
        n_seps = features.shape[1]
        sep_ids = np.empty(n_seps, dtype=int)
        orientations = np.empty(n_seps, dtype=np.int8)
        for i in range(features.shape[1]):
            s = ba.bitarray((features[:, i] > 0).tolist())
            sep_ids[i], orientations[i] = self._get_single_sep_id_bit(s)
        return sep_ids, orientations

    def compute_infimum(self, feat_ids: np.ndarray, specifications: np.ndarray) -> np.ndarray:
        sup = (self.seps_ba[feat_ids[0]] if specifications[0] > 0 else ~self.seps_ba[feat_ids[0]]).copy()
        for i in range(1,feat_ids.shape[0]):
            sup &= self.seps_ba[feat_ids[i]] if specifications[i] > 0 else ~self.seps_ba[feat_ids[i]]
        return 2 * np.frombuffer(sup.unpack(), dtype=np.int8) - 1

    def _add_sep_bit(self, new_sep_bit) -> tuple[int, int]:
        sep_id, orientation = self._get_single_sep_id_bit(new_sep_bit)
        if sep_id == -1:
            sep_id = len(self.seps_ba)
            self.seps_ba.append(new_sep_bit)
            self.sep_hash_to_id.add(new_sep_bit, sep_id)
        return sep_id, orientation

    def _add_sep(self, new_sep: np.ndarray) -> tuple[int, int]:
        new_sep_bit = ba.bitarray((new_sep[:, 0] > 0).tolist())
        return self._add_sep_bit(new_sep_bit)

    def _compute_infimum_of_two(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> tuple[int, int]:
        a = self.seps_ba[sep_id_a] if orientation_a > 0 else ~self.seps_ba[sep_id_a]
        b = self.seps_ba[sep_id_b] if orientation_b > 0 else ~self.seps_ba[sep_id_b]
        sup = a & b
        id, o = self._get_single_sep_id_bit(sup)
        metadata = MetaData(((sep_id_a, orientation_a), (sep_id_b, orientation_b)), orientation=1, type='inf')
        if id == -1:
            id = len(self.seps_ba)
            self.seps_ba.append(sup)
            self.sep_hash_to_id.add(sup, id)
            self.sep_metadata.append(metadata)
        elif id != sep_id_a and id != sep_id_b:
            metadata.orientation = o
            self.sep_metadata[id].append(metadata)
        return id, o

    def _compute_le(self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int) -> bool:
        a = self.seps_ba[sep_id_a] if orientation_a > 0 else ~self.seps_ba[sep_id_a]
        b = self.seps_ba[sep_id_b] if orientation_b > 0 else ~self.seps_ba[sep_id_b]
        return (a | b).count() == b.count()

    def __getitem__(self, feat_ids) -> np.ndarray:
        """
        Get the feature or features as a matrix of indicator columnvectors.

        Parameters
        ----------
        feat_ids : int or np.ndarray
            The feature id or list of feature ids to access.

        Returns
        -------
        np.ndarray
            Indicator-vector matrix of shape (number of datapoints, number of features) .
        """

        if isinstance(feat_ids, numbers.Integral):
            seps = 2 * np.frombuffer(self.seps_ba[feat_ids].unpack(), dtype=np.int8) - 1
        elif isinstance(feat_ids, slice):
            start, stop, step = feat_ids.indices(len(self.seps_ba))
            seps = np.empty((self.datasize, (stop - start) // step), dtype=np.int8)
            for i, id in enumerate(range(start, stop, step)):
                seps[:, i] = 2 * np.frombuffer(self.seps_ba[id].unpack(), dtype=np.int8) - 1
            return seps
        else:
            seps = np.empty((self.datasize, len(feat_ids)), dtype=np.int8)
            for i, id in enumerate(feat_ids):
                seps[:, i] = 2 * np.frombuffer(self.seps_ba[id].unpack(), dtype=np.int8) - 1
        return seps
