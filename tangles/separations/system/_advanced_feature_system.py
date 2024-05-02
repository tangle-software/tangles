from typing import Union
import numpy as np
from tangles._typing import Feature
from ._set_system_base import SetSeparationSystemBase, MetaData


class AdvancedFeatureSystem(SetSeparationSystemBase):
    def __init__(self, data_size: int, capacity: int = 1):
        """
        Create an Advanced FeatureSystem, able to store Advanced Features.

        Advanced features are functions mapping from some ground set to the reals.

        Parameters
        ----------
        data_size
            the number of elements in the ground set.
        capacity
            the starting size of the advanced feature system. defaults to 1.
            If one knows the rough number of features one desires to store, inputting
            this value here might help performance.
        """
        super().__init__(data_size)
        self.features = np.empty((data_size, capacity))
        self.size = 0
        self.feature_map = {}
        self.hash_weights = np.random.random(size=(1, data_size))

    def count_big_side(self, sep_id: int):
        if sep_id >= len(self):
            raise ValueError("unknown feature")
        return (self.features[:, sep_id] > 0).sum()

    def side_counts(self, sep_id: int) -> tuple[int, int]:
        if sep_id >= len(self):
            raise ValueError("unknown separation")
        s = (self.features[:, sep_id] > 0).sum()
        return s, self.datasize - s

    def feature_size(self, feat_id: int) -> int:
        """
        Returns
        -------
        int
            The size of the given feature
        """
        return self.count_big_side(feat_id)

    def feature_and_complement_size(self, feat_id: int) -> tuple[int, int]:
        """
        Returns
        -------
        pair of ints
            size of the feature and its complement
        """
        return self.side_counts(feat_id)

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

    def feature_metadata(
        self, feature_ids: Union[int, list, np.ndarray, range, None] = None
    ) -> Union[MetaData, list[MetaData]]:
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

    def add_features(
        self, new_features: np.ndarray, metadata=None
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def add_corner(
        self,
        feature_id_a: int,
        specification_a: int,
        feature_id_b: int,
        specification_b: int,
    ) -> tuple[int, int]:
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

        return super().add_corner(
            feature_id_a, specification_a, feature_id_b, specification_b
        )

    def get_corners(
        self, feature_id_a: int, feature_id_b: int
    ) -> tuple[np.ndarray, np.ndarray]:
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

    def is_le(
        self, feat_id_a: int, specification_a: int, feat_id_b: int, specification_b: int
    ) -> bool:
        r"""
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

    def is_subset(
        self, feat_id_a: int, specification_a: int, feat_id_b: int, specification_b: int
    ) -> bool:
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

    def metadata_matrix(
        self,
        feature: Feature,
        data_list: list,
        normal_form: "str" = "disjunctive",
        _known_feature_matrices: dict[int, tuple] = None,
    ) -> np.ndarray:
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

        return super().metadata_matrix(
            feature, data_list, normal_form, _known_feature_matrices
        )

    def assemble_meta_info(
        self, feat_id: int, known_meta_info: dict[int, tuple] = None
    ):
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

        return self.size

    def copy(self) -> "AdvancedFeatureSystem":
        """
        create a copy of this fesature system

        Returns
        -------
            AdvancedFeatureSystem: a copy
        """

        copy = AdvancedFeatureSystem(self.datasize)
        copy.features = self.features.copy()
        copy.size = self.size
        copy.feature_map = self.feature_map.copy()
        return copy

    def get_sep_ids(self, seps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.get_feature_ids(seps)

    def _hash_features(self, features: np.ndarray) -> np.ndarray:
        return np.floor(1000 * np.abs(self.hash_weights @ features)).reshape(-1)

    def get_feature_ids(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        If the advanced features or their inverses are already contained in the advanced feature system then return the
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
        feature_ids = -np.ones(n_seps, dtype=int)
        orientations = np.zeros(n_seps, dtype=np.int8)

        hash_values = self._hash_features(features)
        for i in range(features.shape[1]):
            existing_feature_ids = self.feature_map.get(hash_values[i], None)
            if existing_feature_ids is None:
                continue
            if (
                eq := (
                    self.features[:, existing_feature_ids] == features[:, i : i + 1]
                ).all(axis=0)
            ).any():
                feature_ids[i], orientations[i] = existing_feature_ids[eq.argmax()], 1
            elif (
                eq := (
                    self.features[:, existing_feature_ids] == -features[:, i : i + 1]
                ).all(axis=0)
            ).any():
                feature_ids[i], orientations[i] = existing_feature_ids[eq.argmax()], -1

        return feature_ids, orientations

    def compute_infimum(
        self, feat_ids: Union[np.ndarray, list], specifications: Union[np.ndarray, list]
    ) -> np.ndarray:
        return np.minimum.reduce(self.features[:, feat_ids] * specifications, axis=1)

    def _add_new_sep(self, new_sep: np.ndarray):
        new_id = self.size
        if new_id == self.features.shape[1]:
            new_size = int(self.features.shape[1] * 9 / 8 + 6)
            # new size formular is similar to what is found in the CPython source code
            new_features = np.empty((self.datasize, new_size))
            new_features[:, :new_id] = self.features
            self.features = new_features
        self.features[:, new_id] = new_sep
        self.size += 1

        hash_value = self._hash_features(new_sep)[0]
        mapped = self.feature_map.get(hash_value, None)
        if mapped is None:
            mapped = []
            self.feature_map[hash_value] = mapped
        mapped.append(new_id)

        return new_id

    def _add_sep(self, new_sep: np.ndarray) -> tuple[int, int]:
        feat_id, ori = self.get_feature_ids(new_sep)
        feat_id, ori = feat_id[0], ori[0]
        if feat_id == -1:
            feat_id, ori = self._add_new_sep(new_sep[:, 0]), 1
        return feat_id, ori

    def _compute_and_add_corner(
        self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int
    ) -> tuple[int, int]:
        infi = np.minimum(
            self.features[:, sep_id_a] * orientation_a,
            self.features[:, sep_id_b] * orientation_b,
        )
        feat_id, ori = self.get_feature_ids(infi)
        feat_id, ori = feat_id[0], ori[0]
        metadata = MetaData(
            ((sep_id_a, orientation_a), (sep_id_b, orientation_b)),
            orientation=1,
            dtype="inf",
        )
        if feat_id == -1:
            feat_id, ori = self._add_new_sep(infi), 1
            self.sep_metadata.append(metadata)
        elif feat_id != sep_id_a and feat_id != sep_id_b:
            metadata.orientation = ori
            self.sep_metadata[feat_id].append(metadata)
        return feat_id, ori

    def _compute_le(
        self, sep_id_a: int, orientation_a: int, sep_id_b: int, orientation_b: int
    ) -> bool:
        return bool(
            (
                self.features[:, sep_id_a] * orientation_a
                <= self.features[:, sep_id_b] * orientation_b
            ).all()
        )

    def __getitem__(self, feat_ids) -> np.ndarray:
        """
        Get the feature or features as a matrix of indicator column vectors.

        Parameters
        ----------
        feat_ids : int or np.ndarray
            The feature id or list of feature ids to access.

        Returns
        -------
        np.ndarray
            Indicator-vector matrix of shape (number of datapoints, number of features) .
        """

        return self.features[:, : self.size][:, feat_ids]
