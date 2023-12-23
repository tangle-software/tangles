import numpy as np
import scipy as sp
import pandas as pd

from typing import Optional, Union, Callable, Tuple
from warnings import warn
from copy import deepcopy
import scipy.sparse

from tangles.convenience import TangleSweepFeatureSys
from tangles.search import UncrossingSweep
from tangles.separations import FeatureSystem, SetSeparationSystem
from tangles.convenience.Survey import Survey, ColumnSelectionTypes
from tangles.convenience.convenience_functions import search_tangles, search_tangles_uncrossed
from tangles.convenience.SurveyFeatureFactory import SurveyFeatureFactory, SimpleSurveyFeatureFactory_MissingValuesBothSides
from tangles._typing import SetSeparationOrderFunction
from tangles.convenience.convenience_orders import create_order_function, order_works_on_features
from tangles.util.graph.similarity import cosine_similarity
from tangles.util.graph.cut_weight import CutWeightOrder
from tangles.util import unique_rows
from tangles.convenience.SurveyVariable import SurveyVariableValues



import sys
import os
import pickle
from datetime import datetime
from pathlib import Path


from tangles.search.progress import TangleSearchProgressType, DefaultProgressCallback, PROGRESS_TYPE_SOMETHING_STARTING, PROGRESS_TYPE_SOMETHING_FINISHED



# TODO: consider writing a class for the metadata.info field?

class SurveyTangles:
    """Manages a tangle search on survey data.

    Attributes
    ----------
    survey : :class:`~tangles.convenience.Survey.Survey`
        A survey object.
    sweep : :class:`~tangles.convenience.TangleSweepFeatureSys` or :class:`~tangles.search._uncrossing_sweep.UncrossingSweep`
        An object containing a sweep object and a feature system (or separation system).
    agreement : int
        The currently valid agreement lower bound.
    similarity_matrix : np.ndarray or scipy.sparse.spmatrix
        A similarity matrix.
    order : list, np.ndarray or :class:`SetSeparationOrderFunction`
        An object indicating in which order the features are used for the tangle search.
        A :class:`SetSeparationOrderFunction` is a ``Callable[[np.ndarray], np.ndarray]``.
    progress_callback : :class:`~tangles.search.progress.DefaultProgressCallback` or callable
        A callable providing a progress indication (see :class:`~tangles.search.progress.DefaultProgressCallback` for reference).
    """

    @classmethod
    def search(cls, survey: Survey, agreement: int,
               features_or_separations: Optional[np.ndarray] = None, metadata: Optional[Union[np.ndarray, list]] = None,
               order:Optional[Union[list,np.ndarray,SetSeparationOrderFunction, str]] = None,
               similarity_matrix: Union[np.ndarray, sp.sparse.spmatrix] = None,
               feature_factory: Optional[SurveyFeatureFactory] = None,
               uncross: bool = False,
               progress_callback=DefaultProgressCallback()) -> 'SurveyTangles':
        """Search tangles.
        
        Returns a :class:`SurveyTangles` object managing a tangle search on survey data.

        Parameters
        ----------
        survey : :class:`Survey`
            A survey object.
        agreement : int
            The currently valid agreement lower bound.
        features_or_separations : np.ndarray or None
            A matrix containing the features in its columns.
        metadata : np.ndarray or None
            A list of metadata corresponding to `features_or_separations`.
        order : list, np.ndarray, :class:`~tangles._typing.SetSeparationOrderFunction`, str or None
            A list or np.ndarray of indices, an order function or the name of an order function, indicating the order
            in which the features are used for building a tangle search tree.
        similarity_matrix : np.ndarray or None
            A similarity matrix possibly used by the order function.
        feature_factory : :class:`SurveyFeatureFactory` or None
            An object turning questions (i.e. the rows of `survey`) into features (or separations). 
            See also :class:`SurveyFeatureFactory`.
        uncross : bool
            If True, uncross the features (or separations) that distinguish at least two tangles.
        progress_callback : :class:`~tangles.search.progress.DefaultProgressCallback` or callable
            A callable providing a progress indication (see :class:`~tangles.search.progress.DefaultProgressCallback` for reference).

        Returns
        ----------
        :class:`SurveyTangles`
            An object managing a tangle search on survey data.
        """

        tangles = cls(survey)
        tangles.similarity_matrix = similarity_matrix

        if features_or_separations is None:
            if not feature_factory:
                feature_factory = SimpleSurveyFeatureFactory_MissingValuesBothSides(survey)
            if progress_callback:
                progress_callback(PROGRESS_TYPE_SOMETHING_STARTING, info="creating features...")
            features_or_separations, metadata = feature_factory.create_features()
            if progress_callback:
                progress_callback(PROGRESS_TYPE_SOMETHING_FINISHED, info="creating features... finished")

        if isinstance(order, str):
            if not order_works_on_features(order):
                if tangles.similarity_matrix is None:
                    warn("Your order needs a similarity matrix. We create one for you :-) [sorry, this may take a while...]")
                    if progress_callback:
                        progress_callback(PROGRESS_TYPE_SOMETHING_STARTING, info="computing similarity...")
                    tangles.similarity_matrix = cosine_similarity(survey.data.to_numpy(), sim_thresh=0.25, max_neighbours=min(max(survey.num_respondents//5,1), 50), sequential=True)
                    if progress_callback:
                        progress_callback(PROGRESS_TYPE_SOMETHING_FINISHED, info="computing similarity... finished")
                order_func = create_order_function(order, tangles.similarity_matrix)
            else:
                order_func = create_order_function(order, features_or_separations)
            tangles.order = order_func
        else:
            tangles.order = order

        tangles.progress_callback = progress_callback
        tangles.initialize_search(agreement=agreement, features_or_separations=features_or_separations, metadata=metadata, uncrossing=uncross)
        return tangles




    def __init__(self, survey: Survey):
        self.survey: Survey = survey
        self.sweep: Optional[Union[TangleSweepFeatureSys,UncrossingSweep]] = None
        self.agreement: Optional[int] = None
        self.similarity_matrix: Union[np.ndarray, scipy.sparse.spmatrix, None] = None
        self.order: Union[list, np.ndarray, SetSeparationOrderFunction, None] = None
        self.progress_callback: TangleSearchProgressType = DefaultProgressCallback()

    def initialize_search(self, agreement: int, features_or_separations: np.ndarray, metadata: Optional[np.ndarray] = None, uncrossing: bool = False):
        """This function starts a tangle search on survey data.

        Parameters
        ----------
        agreement : int
            The agreement lower bound for the search.
        features_or_separations : np.ndarray
            A matrix containing the features in columns.
        metadata : np.ndarray or None
            A list of metadata corresponding to `features_or_separations`.
        uncrossing : bool
            If True, uncross the features (or separations) that distinguish at least two tangles.
        """

        if metadata is None:
            warn("SurveyTangles.search: feature meta data missing: some functionality might not be available")
        elif (e := self._check_metadata(metadata)) is not None:
            warn("Your metadata might not have the right format to use the full functionality of the class SurveyTangles. "
                 "You should consider using a lower level interface."
                 f"Error: {e}")

        if self.order is None and self.similarity_matrix is not None:
            self.order = CutWeightOrder(self.similarity_matrix)

        if uncrossing:
            if self.order is None or not isinstance(self.order, Callable):
                raise ValueError("We need a submodular order function to be able to uncross!")
            self.sweep = search_tangles_uncrossed(features_or_separations, agreement, order_func=self.order,  sep_metadata=metadata, progress_callback=self.progress_callback)
        else:
            self.sweep = search_tangles(features_or_separations, agreement, order=self.order, sep_metadata=metadata, progress_callback=self.progress_callback)
        self.agreement = agreement
        self._normalize_meta_data()

    @property
    def valid_agreement_lower_bound(self):
        """Current agreement lower bound.

        Returns
        ----------
        int
            The lower bound.
        """

        return self.sweep.tree.limit

    @property
    def feature_system(self) -> Union[FeatureSystem, SetSeparationSystem]:
        """The feature system (or separation system) in use.

        Returns
        ----------
        FeatureSystem or SetSeparationSystem:
            The feature system (or separation system) in use.
        """
        return self.sweep.sep_sys

    def sepcified_features(self, only_original_features: bool = True)-> np.ndarray:
        """Determine which features could be specified by the tangles found for the current agreement.

        The resulting list of feature ids are sorted by order.

        Parameters
        ----------
        only_original_features : bool
            If True, the result does not contain any corners that were inserted while uncrossing distinguishing features

        Returns
        -------
        np.ndarray
            A list of feature ids, sorted by order.
        """

        oriented_seps = self.sweep.oriented_sep_ids_for_agreement(self.agreement)
        if only_original_features:
            oriented_seps = oriented_seps[np.isin(oriented_seps,self.sweep.original_sep_ids)]
        return oriented_seps


    def tangle_matrix(self, return_metadata: bool = False, remove_duplicate_rows: bool = True, remove_prefixing_tangles: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
        """Create a matrix indicating how the features (or separations) are specified (or oriented) by the tangles.

        Each row corresponds to a tangle and each column corresponds to a feature (or separation).

        Parameters
        ----------
        return_metadata : bool
            Whether to include metadata.
        remove_duplicate_rows : bool
            If the tangles were searched with uncrossing: remove rows corresponding to tangles that are identical on the originally oriented seps.
            Such rows appear whenever tangles that differ only in corners used to uncross distinguishing features.
        remove_prefixing_tangles : bool
            If the tangles were searched with uncrossing: whether a row should be remove it its non-zero entries are
            the prefix of another row, i.e. if another row starts with the same non-zero entries. 
            In other words, whether a tangle should be removed if it specifies its features exactly as another tangle
            of higher order does.

        Returns
        ----------
        np.ndarray or tuple[np.ndarray, list]
            The matrix of specifications (or orientations) or, if `return_metadata` is True, a tuple with that matrix
            in its first entry and a list of metadata corresponding to the columns of the matrix in its second entry.
        """

        tangle_mat = self.sweep.tangle_matrix(min_agreement=self.agreement)
        if remove_duplicate_rows and len(tangle_mat.shape) > 2 and tangle_mat.shape[0] > 1:
            tangle_mat = unique_rows(tangle_mat)
        if remove_prefixing_tangles:
            s = ((tangle_mat[:,np.newaxis,:] * tangle_mat[np.newaxis,:,:]) >= 0).all(axis=2)
            nonz = (tangle_mat != 0).sum(axis=1)
            tangle_mat = tangle_mat[~((s & (nonz[:,np.newaxis] < nonz[np.newaxis,:])).any(axis=1)),:]
        return (tangle_mat, self.feature_system.separation_metadata(self.sweep.original_sep_ids[:tangle_mat.shape[1]])) if return_metadata else tangle_mat

    def change_agreement(self, agreement:int, force_tree_update: bool = False):
        """Change the agreement lower bound.

        Parameters
        ----------
        agreement : int
            The new agreement lower bound.
        force_tree_update : bool
            If set to True, this parameter can prevent you from accidentally starting a time-consuming task.

            A time-consuming task may be necessary whenever the agreement lower bound is changed such that its new
            value is below the limit of the :class:`~tangles.search._tree.TangleSearchTree`. In that case, the tangle
            search tree is updated by the :class:`sweep <tangles.search._sweep.TangleSweep>`.
        """

        if agreement > self.valid_agreement_lower_bound:
            self.agreement = agreement
        elif force_tree_update:
            num_seps_before = len(self.feature_system)
            self.sweep.lower_agreement(agreement, progress_callback=self.progress_callback)
            self.agreement = agreement
            if len(self.feature_system) != num_seps_before:
                self._normalize_meta_data()
        else:
            warn("Agreement lower than or equal to current valid agreement lower bound. We have to update the tree: you can enforce an update of the tangle tree by setting <force_tree_update> = True")
            return

    def ordered_metadata(self, only_original_seps: bool = False, insert_labels: bool = False) -> list:
        """Return the metadata of the features (or separations) sorted corresponding to the order used for the tangle search.

        Parameters
        ----------
        only_original_seps : bool
            If True, only the original features (or separations) are returned, otherwise the result possibly contains
            corners needed for the uncrossing of distinguishing features (or separations).
        insert_labels : bool
            If True, the variable names in the metadata are replaced by the variables' labels. 
            Often the label is the question text.

        Returns
        -------
        list
            A list of metadata-lists. The metadata-list at index i contains all metadata for the i-th specified feature (or oriented separation).
        """

        sep_ids = self.sweep.original_sep_ids if only_original_seps else self.sweep.all_oriented_sep_ids
        meta_data = deepcopy([m.tail_as_list() for m in self.sweep.sep_sys.separation_metadata(sep_ids)])
        if insert_labels:
            variables = self.survey.variables
            for mlist in meta_data:
                for m in mlist:
                    if m.type == 'custom':
                        m.info = (variables[m.info[0]].label or m.info[0], m.info[1], variables[m.info[0]].valid_values.get(m.info[2]) or m.info[2])
        return meta_data

    def _normalize_meta_data(self):
        for m in self.sweep.sep_sys.separation_metadata():
            while m:
                if m.type == 'custom' and m.orientation < 0: # TODO: what about corners?
                    m.orientation, m.info = 1, (m.info[0], SurveyVariableValues.invert_op(m.info[1]), m.info[2])
                m = m.next


    def typical_answers(self, only_affected_questions: bool = True, column_selection: ColumnSelectionTypes = None,
                        insert_labels: bool = True, extract_const_answers:bool = False,
                        remove_incomplete_tangles: bool = False) -> Union[pd.DataFrame, None, Tuple[pd.DataFrame,pd.DataFrame]]:
        """Create a dataframe containing the 'typical' answers given by each tangle.
        
        Parameters
        ----------
        only_affected_questions : bool
            If True, the questions corresponding to features (or separations) that were not specified (or oriented) are not included in the result.
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        insert_labels : bool
            If True, the variable names in the metadata are replaced by the variables' labels. 
            Often the label is the question text.
        extract_const_answers : bool
            If True, two dataframes are returned, one containing the answers that are typical in every tangle and one containing the typical answers.
        remove_incomplete_tangles : bool
            If True, remove tangles that do not specify all features.

        Returns
        -------
        pandas.DataFrame or tuple (DataFrame, DataFrame)
            A dataframe containing the 'typical' answers in each tangle,
            or tuple with the answers that are typical in every tangle in its first component and the typical answers
            in its second component.
        """

        tangle_mat = self.tangle_matrix(return_metadata=False)
        if tangle_mat.shape[0] == 0:
            warn("no tangles found")
            return (None, None) if extract_const_answers else None
        if tangle_mat.shape[1] == 0:
            warn("no tangle contains the original separations, please try something different")
            return (None, None) if extract_const_answers else None

        if remove_incomplete_tangles:
            tangle_mat = tangle_mat[tangle_mat[:,-1]!=0,:]

        sep_ids = self.sweep.original_sep_ids[:tangle_mat.shape[1]]
        metadata = [[m for m in self.sweep.sep_sys.separation_metadata(sep_id).tail_as_gen() if m.type == 'custom'] for sep_id in sep_ids]
        selected_cols = self.survey.interpret_column_selection(column_selection)
        if only_affected_questions:
            included_vars = set().union(*[[m.info[0] for m in mlist] for mlist in metadata])
            selected_cols = [c for c in selected_cols if c in included_vars]
            included_vars.intersection_update(selected_cols)
        else:
            included_vars = None

        typical_answers = pd.DataFrame(index=range(tangle_mat.shape[0]), columns=selected_cols, data=None)
        for tangle_idx in range(tangle_mat.shape[0]):
            var_values = self._answers_for_orientation(tangle_mat[tangle_idx,:], metadata, included_vars)
            for v in var_values:
                value_rep = v.possible_values_representation(insert_labels)
                typical_answers.loc[tangle_idx, v.var.name] = value_rep

        if insert_labels:
            typical_answers.rename(columns={v.name: (v.label or v.name) + f"[{v.name}]" for v in self.survey.variables}, inplace=True)

        if extract_const_answers:
            const = typical_answers.nunique() == 1
            const_answers = pd.DataFrame(typical_answers.loc[0,const]).T
            typical_answers = typical_answers.loc[:,~const]
            return typical_answers, const_answers
        else:
            return typical_answers


    # -------------------------- private ------

    def _answers_for_orientation(self, orientation, metadata_list, included_vars=None):
        if included_vars:
            answers = {q.name:q.create_values() for q in self.survey.variables if q.name in included_vars}
        else:
            answers = {q.name:q.create_values() for q in self.survey.variables}
        for metadata,ori in zip(metadata_list,orientation[orientation != 0]):
            for m in metadata:
                if m.info[0] not in answers:
                    continue
                answers[m.info[0]].update_values_for_specification(ori, m.info, m.orientation)
        return list(answers.values())

    def _check_metadata(self, metadata) -> Union[str, None]:
        if not isinstance(metadata, (list, np.ndarray)):
            return "metadata is not a list or array"
        for m in metadata:
            if not isinstance(m, Tuple):
                return "metadatalist entry is not a tuple"
            if len(m) != 3:
                return "metadata tuple has wrong size"
            if not m[0] in self.survey.variables:
                return "first entry of metadata tuple is not a valid variable name"
            if not self.survey.variables[m[0]].is_allowed_operation(m[1]):
                return "second entry of metadata tuple is not a valid operation"
        return None

    def save(self, file_name:str, append_date_to_name=True):
        warn("Save functionality is very rudimentary, it might be a problem to load the stored file after the library changes...")
        folder = os.path.dirname(file_name)
        Path(folder).mkdir(parents=True, exist_ok=True)
        file, ext = os.path.splitext(file_name)
        if ext is None or len(ext) == 0:
            ext = ".tngl"
        if append_date_to_name:
            now = datetime.now()
            file += now.strftime("%Y-%m-%d-%H-%M-%S")+ext

        rec_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(1000000)
        with open(file, "wb") as f:
            pickle.dump(self, f)
        sys.setrecursionlimit(rec_limit)



