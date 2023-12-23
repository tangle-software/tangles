from typing import Tuple, Optional, Union, Callable
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from abc import ABC, abstractmethod
from .SurveyVariable import SurveyVariable, UnionOfIntervals, SurveyVariableValues
from .Survey import Survey, ColumnSelectionTypes

from warnings import warn


class SurveyFeatureFactory(ABC):
    """Abstract base class for feature factories.

    A feature factory can be used to create binary features from (possibly non-binary) survey variables.

    Parameters
    ----------
    survey : :class:`~tangles.convenience.Survey`
        The factory will be able to create features (or separations) for variables of this survey.
    """


    default_numeric_var_num_bins = 5
    """A lot of features for numeric variables divide the range of the variable in bins. This is the default number of bins."""

    def __init__(self, survey:Survey):
        self.survey: Survey = survey

    @abstractmethod
    def create_features_for_single_col(self, variable: SurveyVariable, col_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create a set of binary features given a variable and a column of the data containing all answers to the
        question corresponding to `variable`.
        
        A factory method, to be overwritten by sub classes.

        Parameters
        ----------
        variable : :class:`SurveyVariable`
            A survey variable.
        col_data : pandas.Series
            The respondents' answers to the question corresponding to `variable`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            In the first entry, a matrix of shape ``(len(col_data), num_features)``, that contains the features as
            (oriented) indicator vectors. Here, ``num_features`` is the number of features created by this method.
            
            In the second entry, a matrix of shape ``(num_features,)``, that contains metadata for each feature.

            The metadata of each feature is expected to be a tuple ``(operation, value)``. For example, the tuple
            ``('>=', 8)`` describes a feature (or separation) that splits the column into a group that answered less
            than 8 and one that answered at least 8.
        """

        ...

    def create_features(self, column_selection: ColumnSelectionTypes = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for a set of variables corresponding to the columns specified by `column_selection`.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            In the first entry, a matrix of shape ``(len(col_data), num_features)``, that contains the features as
            (oriented) indicator vectors. Here, ``num_features`` is the number of features created by this method.
            
            In the second entry, a matrix of shape ``(num_features,)``, that contains metadata for each feature.

            The metadata of each feature is expected to be a tuple ``(operation, value)``. For example, the tuple
            ``('>=', 8)`` describes a feature (or separation) that splits the column into a group that answered less
            than 8 and one that answered at least 8.
        """

        columns = self.survey.interpret_column_selection(column_selection)
        seps = []
        metadata = []
        for c in columns:
            if len(self.survey.data[c].unique()) < 2:
                continue

            s, meta = self.create_features_for_single_col(self.survey.variables[c], self.survey.data[c])
            if self.check_factory_func_result(s, meta):
                seps.append(s)
                metadata.extend((c, *m) for m in meta)

        sep_metadata = np.empty(len(metadata), dtype=object)
        sep_metadata[:] = metadata
        return np.hstack(seps), sep_metadata

    def check_factory_func_result(self, s: np.ndarray, meta: np.ndarray) -> bool:
        # make sure, the subclasses at least make no obvious errors that result in more or less cryptic error messages later...
        if s is None: # no error, just no features...
            return False
        if meta is None:
            warn("metadata missing completely")
            return False
        if len(s.shape)==1:
            if s.shape[0] == 0: # no error, just no features...
                return False
            if meta.shape[0] != 1:
                warn("number of features does not match number of metadata")
                return False
            if s.shape[0] != self.survey.num_respondents:
                warn("feature vector has wrong size")
                return False
        if len(s.shape)>2:
            warn("feature matrix has too much dimensions")
            return False
        if s.shape[1] == 0: # no error, just no features...
            return False
        if s.shape[1] != meta.shape[0]:
            warn("number of features does not match number of metadata")
            return False
        if s.shape[0] != self.survey.num_respondents:
            warn("feature matrix has wrong number of rows")
            return False

        return True




class SurveyFeatureFactoryDecorator(SurveyFeatureFactory, ABC):
    """
    A small decorator class that can extend the behaviour of another survey feature factory.

    Attributes
    ----------
    embedded_feature_factory : :class:`SurveyFeatureFactory`
        Another feature factory. The default is a :class:`SimpleSurveyFeatureFactory`.
    """

    def __init__(self, survey: Survey, embedded_feature_factory: Optional[SurveyFeatureFactory] = None):
        super().__init__(survey)
        self.embedded_feature_factory = embedded_feature_factory or SimpleSurveyFeatureFactory(survey)




class SimpleSurveyFeatureFactory(SurveyFeatureFactory):
    """A simple survey factory, used as default factory.

    Attributes
    ----------
    numvar_func : create_feature_func
        A feature factory function to create features for numeric variables. 
        The default function splits the range regularly (see :func:`numericvar_features_split_regular_ge`).
    ordvar_func : create_feature_func
        A feature factory function to create features for ordinal variables. 
        The default function splits the range regularly (see :func:`ordinalvar_features_ge_all_splits`).
    nomvar_func : create_feature_func
        A feature factory function to create features for nominal variables.
        The default function creates a feature for every possible value the variable can take (see :func:`nominalvar_features_all_cats`).
    """

    create_feature_func = Callable[[pd.Series, Union[list, np.ndarray, pd.Series]], Tuple[np.ndarray, np.ndarray]]
    """format of the feature functions that can be used as factory function pointers in this class"""

    def __init__(self, survey:Survey):
        super().__init__(survey)
        num_func = lambda c,i:  numericvar_features_split_regular_ge(c, num_bins=SurveyFeatureFactory.default_numeric_var_num_bins,
                                                                    max_num_values_for_extensive_seps=50,
                                                                    invalid_values=i)
        self.numvar_func: SimpleSurveyFeatureFactory.create_feature_func = num_func
        self.ordvar_func: SimpleSurveyFeatureFactory.create_feature_func = ordinalvar_features_ge_all_splits
        self.nomvar_func: SimpleSurveyFeatureFactory.create_feature_func = nominalvar_features_all_cats

    def create_features_for_single_col(self, variable: SurveyVariable, col_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        invalid = variable.invalid_values_as_list()
        if variable.type in SurveyVariable.numeric_types:
            return self.numvar_func(col_data, invalid)
        elif variable.type in SurveyVariable.ordinal_types:
            return self.ordvar_func(col_data, invalid)
        elif variable.type in SurveyVariable.nominal_types:
            return self.nomvar_func(col_data, invalid)
        else:
            warn("cannot create features for unknown variable types")


class SimpleSurveyFeatureFactory_MissingValuesOwnFeatures(SurveyFeatureFactoryDecorator):
    """A simple survey factory that takes missing and invalid values into account.

    This factory creates one feature for each missing value. This feature partitions the dataset in respondents who
    gave an invalid answer vs. respondents who gave a valid answer.
    """

    def create_features_for_single_col(self, variable: SurveyVariable, col_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        seps, meta = self.embedded_feature_factory.create_features_for_single_col(variable=variable, col_data=col_data)
        if variable.invalid_values:
            invalid_values_list = variable.invalid_values_as_list()
            valid_sep = -np.ones((len(col_data),1), dtype=np.int8)
            valid_sep[col_data.isin(invalid_values_list), 0] = 1
            seps = np.hstack((valid_sep, seps))
            new_meta = np.empty((1,), dtype=object)
            new_meta[0] = ("in", UnionOfIntervals.create_with_isolated_points(invalid_values_list) if variable.is_numeric_type() else invalid_values_list)
            meta = np.hstack((new_meta, meta))
        return seps, meta

class SimpleSurveyFeatureFactory_MissingValuesBothSides(SurveyFeatureFactoryDecorator):
    """A simple survey factory that takes missing and invalid values into account.

    This factory creates set separations and assigns respondents who gave invalid answers to both sides of the
    separation.
    """

    def create_features_for_single_col(self, variable: SurveyVariable, col_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        seps, meta = self.embedded_feature_factory.create_features_for_single_col(variable=variable, col_data=col_data)
        if variable.invalid_values:
            seps[col_data.isin(variable.invalid_values_as_list()), :] = 0
        return seps, meta


class SimpleSurveyFeatureFactory_MissingValuesImplicit(SimpleSurveyFeatureFactory):
    """A simple survey factory that takes missing and invalid values into account.

    This factory creates features that don't need to handle invalid and missing answers explicitly. 
    The invalid answers are assigned to each feature's complement.
    """

    def __init__(self, survey:Survey, numeric_var_num_bins: int = 5):
        super().__init__(survey)
        self.numeric_var_num_bins = numeric_var_num_bins

    def create_features_for_single_col(self, variable: SurveyVariable, col_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        invalid = variable.invalid_values_as_list()
        if variable.type in SurveyVariable.numeric_types:
            return numericvar_features_inside_regular_bins(col_data, num_bins=self.numeric_var_num_bins, invalid_values=invalid)
        elif variable.type in SurveyVariable.ordinal_types:
            return nominalvar_features_all_cats(col_data, invalid_values=None)
        elif variable.type in SurveyVariable.nominal_types:
            return nominalvar_features_all_cats(col_data, invalid_values=None)
        else:
            warn("cannot create features for unknown variable types")


class SurveyFeatureFactory_CherryPicker(SurveyFeatureFactoryDecorator):
    """A survey feature factory extending a default factory that allows to change the factory functions for individual 
    variables.

    Parameters
    ----------
    survey : :class:`~tangles.convenience.Survey`
        The factory will be able to create features (or separations) for variables of this survey.
    default : :class:`SurveyFeatureFactory`
        A survey feature factory used as default. This factory creates features for all variables that are not treated 
        individually.
    """

    def __init__(self, survey:Survey, default: Optional[SurveyFeatureFactory] = None):
        super().__init__(survey, default or SimpleSurveyFeatureFactory_MissingValuesBothSides(survey))
        self.cherries = {}

    def set_factoryfunc(self, var_name: str, func: Callable[[SurveyVariable, pd.Series], Tuple[np.ndarray, np.ndarray]]):
        """Assign a feature factory function with a variable.

        Parameters
        ----------
        var_name : str
            Name of the variable.
        func : Callable[[SurveyVariable, pd.Series], Tuple[np.ndarray, np.ndarray]])
            A feature factory function.
        """

        self.cherries[var_name] = func

    def create_features_for_single_col(self, variable: SurveyVariable, col_data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        func = self.cherries.get(variable.name, None)
        if func:
            return func(variable, col_data)
        else:
            return self.embedded_feature_factory.create_features_for_single_col(variable, col_data)


#########################################################################################################
# single feature functions.....


def binary_unique_value_features(single_col_data: Union[pd.Series,np.ndarray], unique_values: Union[list,np.ndarray], op: str = '==') -> Tuple[np.ndarray, np.ndarray]:
    """
    A feature factory function for binary variables.

    Parameters
    ----------
    single_col_data : pd.Series or np.ndarray
        The featured data.
    unique_values : list or np.ndarray
        Unique values in `single_col_data`.
    op : {'==', '!=', '<', '<=', '>', '>='}
        The operation used to describe the feature in its metadata. Defaults to '=='.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The features in the first entry and the corresponding metadata in the second entry.
    """

    if len(unique_values) != 2:
        raise ValueError("_binary_unique_value_features: variable is supposed to take exactly two values")
    feat = -np.ones((len(single_col_data),1), dtype=np.int8)
    if op == '==':
        v = max(unique_values)
        feat[single_col_data==v,0] = 1
    elif op == '!=':
        v = min(unique_values)
        feat[single_col_data!=v,0] = 1
    elif op == '<':
        v = max(unique_values)
        feat[single_col_data < v,0] = 1
    elif op == '<=':
        v = min(unique_values)
        feat[single_col_data<=v,0] = 1
    elif op == '>':
        v = min(unique_values)
        feat[single_col_data>v,0] = 1
    elif op == '>=':
        v = max(unique_values)
        feat[single_col_data>=v,0] = 1
    else:
        raise ValueError(f"unknown op: {op}")
    feat_metadata = np.empty(1, dtype=object)
    feat_metadata[0] = (op,v)
    return feat, feat_metadata

def simple_unique_value_features(single_col_data: Union[pd.Series,np.ndarray], unique_values:Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    A simple feature factory function for variables that take a (small) number of unique values.

    Parameters
    ----------
    single_col_data : pd.Series or np.ndarray
        The featured data.
    unique_values : list or np.ndarray
        Unique values in `single_col_data`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The features in the first entry and the corresponding metadata in the second entry.
    """

    if len(unique_values) == 2:
        return binary_unique_value_features(single_col_data, unique_values)
    seps = -np.ones((len(single_col_data), len(unique_values)), dtype=np.int8)
    sep_metadata = np.empty(len(unique_values), dtype=object)
    for i, v in enumerate(unique_values):
        seps[single_col_data == v, i] = 1
        sep_metadata[i] = ("==",v)
    return seps, sep_metadata

def numericvar_features_split_regular_ge(single_col_data: Union[pd.Series,np.ndarray], num_bins: int = 5,
                                      max_num_values_for_extensive_seps: Optional[int] = 50,
                                      invalid_values: Optional[Union[list, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    A  feature factory function for numeric variables. 
    
    The function usually creates multiple features for one question.
    The range of a variable is divided into regular sections, that is, into intervals of the same size. Their 
    boundaries are used as thresholds. 

    Each feature describes the subset of respondents who gave an answer at least as high as the thresholds.

    Parameters
    ----------
    single_col_data : pd.Series or np.ndarray
        The featured data.
    num_bins : int
        Number of bins.
    max_num_values_for_extensive_seps : int
        If a variable takes at most this number of different values, we fall back to ordinal style features
        (this functionality is useful if the survey metadata was configured inaccurately in that variables that are
        ordinal have been incorrectly declared to be numerical variables).
    invalid_values : list or np.ndarray
        The invalid values in `single_col_data`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The features in the first entry and the corresponding metadata in the second entry.
    """

    if not is_numeric_dtype(single_col_data):
        raise ValueError("Need a numeric dtype for a numeric variable!")
    unique_values = np.setdiff1d(single_col_data.unique(), invalid_values) if invalid_values else single_col_data.unique()
    if (l:=len(unique_values)) <= 2:
        return binary_unique_value_features(single_col_data, unique_values, op="==")
    elif max_num_values_for_extensive_seps and l <= max_num_values_for_extensive_seps:
        return ordinalvar_features_ge_all_splits(single_col_data, invalid_values)
    else:
        num_bins = min(num_bins, l)
        splits = np.arange(1,num_bins)/(num_bins)
        feat = -np.ones((len(single_col_data), num_bins-1), dtype=np.int8)
        feat_metadata = np.empty(num_bins-1, dtype=object)
        for i,v in enumerate((min(unique_values) * (1 - splits) + max(unique_values) * splits)):
            feat[single_col_data >= v, i] = 1
            feat_metadata[i] = (">=", v)
    return feat, feat_metadata

def numericvar_features_inside_regular_bins(single_col_data: Union[pd.Series,np.ndarray], num_bins: int = 5,
                                      invalid_values: Optional[Union[list, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    A feature factory function for numeric variables. 
    
    The variable's range is split into regular bins, that is, into intervals of the same size. One feature is created
    for each bin. 
    
    Each feature describes the subset of respondents whose answer to the variable's question was inside the bin.

    Parameters
    ----------
    single_col_data : pd.Series or np.ndarray
        The featured data.
    num_bins : int
        The data's range is split into this number of sections.
    invalid_values : list or np.ndarray
        The invalid values in `single_col_data`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The features in the first entry and the corresponding metadata in the second entry.
    """

    if not is_numeric_dtype(single_col_data):
        raise ValueError("Need a numeric dtype for a numeric variable!")
    unique_values = np.setdiff1d(single_col_data.unique(), invalid_values) if invalid_values else single_col_data.unique()
    if (l:=len(unique_values)) <= 2:
        return binary_unique_value_features(single_col_data, unique_values, op="==")
    else:
        num_bins, mi, ma = min(num_bins, l), unique_values.min(), unique_values.max()
        d = (ma-mi)/num_bins
        feat = -np.ones((len(single_col_data), num_bins), dtype=np.int8)
        feat_metadata = np.empty(num_bins, dtype=object)
        for i in range(num_bins):
            lo,  hi = mi+i*d, mi+(i+1)*d
            feat[(single_col_data >= lo) & ((single_col_data < hi) if i<num_bins-1 else (single_col_data <= hi)), i] = 1
            feat_metadata[i] = ("in", UnionOfIntervals(lo, hi, True, i + 1 == num_bins))
    return feat, feat_metadata


def extensive_numericvar_features(single_col_data: Union[pd.Series,np.ndarray], invalid_values: Optional[Union[list, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    A feature factory function for numeric variables that take a small number of unique values.
    
    This function treats a numeric variable as an ordinal one.
    The range of the variable is split at each unique value. 
    
    This function is useful if the survey metadata was configured inaccurately in that variables that are
    ordinal have been incorrectly declared to be numerical variables.

    Parameters
    ----------
    single_col_data : pd.Series or np.ndarray
        The featured data.
    invalid_values : list or np.ndarray
        The invalid values in `single_col_data`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The features in the first entry and the corresponding metadata in the second entry.
    """

    return ordinalvar_features_ge_all_splits(single_col_data, invalid_values)

def ordinalvar_features_ge_all_splits(single_col_data: Union[pd.Series,np.ndarray], invalid_values: Optional[Union[list, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    A feature factory function for ordinal variables. 
    
    The function creates one feature for every unique value the variable can take except the smallest.

    Each feature describes the respondents who gave an answer at least as high as the unique value used as threshold.

    Parameters
    ----------
    single_col_data : pd.Series or np.ndarray
        The featured data.
    invalid_values : list or np.ndarray
        The invalid values in `single_col_data`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The features in the first entry and the corresponding metadata in the second entry.
    """

    unique_values = np.setdiff1d(single_col_data.unique(), invalid_values)
    if (num_values := len(unique_values)) <= 2:
        return binary_unique_value_features(single_col_data, unique_values, op="==")
    else:
        unique_values.sort()
        feat = -np.ones((len(single_col_data), num_values-1), dtype=np.int8)
        feat_metadata = np.empty(num_values-1, dtype=object)
        for i,v in enumerate(unique_values[1:]):
            feat[single_col_data>=v,i] = 1
            feat_metadata[i] = (">=",v)
    return feat, feat_metadata

def nominalvar_features_all_cats(single_col_data: Union[pd.Series,np.ndarray], invalid_values: Optional[Union[list, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """ A feature factory function for nominal variables.
    
    This function forwards to :func:`simple_unique_value_features`.
    """

    return simple_unique_value_features(single_col_data, np.setdiff1d(single_col_data.unique(), invalid_values))

def feats2seps_invalids_to_both_sides(create_feature_func):
    """ A decorator turning a feature into a set separation by assigning invalid and missing answers to both sides of
    the set separation.
    """

    def func(single_col_data: Union[pd.Series,np.ndarray], invalid_values: Union[list, np.ndarray, pd.Series]):
        seps, meta = create_feature_func(single_col_data, invalid_values=invalid_values)
        seps[single_col_data.isin(invalid_values),:] = 0
        return seps, meta
    return func
