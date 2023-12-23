import numbers

import numpy as np
import scipy as sp
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_float_dtype, is_integer_dtype
from warnings import warn
from copy import deepcopy
from typing import Tuple, Optional, Union, Callable
import os
from pathlib import Path
import json


from .SurveyVariable import SurveyVariable, UnionOfIntervals


QuestionSelector = Callable[[SurveyVariable], bool]
RowSelector = Callable[[pd.Series], bool]
ColumnSelectionTypes = Union[str, int, list,np.ndarray,pd.Series,range,QuestionSelector,None]

class Survey:
    """
    Objects of this class represent survey data and provide functions to prepare, clean and subset survey data.

    This class manages a pandas dataframe and a data structure containing information about the variables.
    It makes sure that the information in both of these objects stays synchronized.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A dataframe containing the survey data.
    """

    @staticmethod
    def load(folder:str, name: str) -> 'Survey':
        """Load a :class:`Survey` from files.

        Parameters
        ----------
        folder : str
            A folder where the survey files can be found.
        name : str
            The name of the survey. The survey is loaded from two files. Both filenames must start with 'name' and one
            must end with '_data.csv' and the other with '_varinfo.csv'.
        """

        data_file = os.path.join(folder, name + "_data.csv")
        var_info_file = os.path.join(folder, name + "_varinfo.csv")
        df = pd.read_csv(data_file, header=0, index_col=0)
        var_info_df = pd.read_csv(var_info_file, index_col=0)
        survey = Survey(df)
        for v_name in var_info_df.index:
            var = survey.variables[v_name]
            var.name = v_name
            var.type = var_info_df.loc[v_name, 'type']
            var.label = var_info_df.loc[v_name, 'label']

            var.valid_values = json.loads(var_info_df.loc[v_name, 'valid_values'])
            var.invalid_values = json.loads(var_info_df.loc[v_name, 'invalid_values'])
            if is_float_dtype(df.dtypes[v_name]):
                var.valid_values = {float(k):v for k,v in var.valid_values.items()}
                var.invalid_values = {float(k):v for k,v in var.invalid_values.items()}
            elif is_integer_dtype(df.dtypes[v_name]):
                var.valid_values = {int(k):v for k,v in var.valid_values.items()}
                var.invalid_values = {int(k): v for k, v in var.invalid_values.items()}
        survey.check_variables(verbose=False)
        return survey



    def __init__(self, data: pd.DataFrame):
        self.data : pd.DataFrame = data.copy()
        self.variables = pd.Series(index=data.columns, data=[SurveyVariable(name=col) for col in data.columns])
        self._convert_non_numeric_columns_to_str()
        self._replace_nan()


    def save(self, folder:str, name: str):
        """Save this survey to a folder:

        Parameters
        ----------
        folder : str
            A folder where the survey files are saved.
        name : str
            The name of the survey. The survey is saved to two files. Both filenames start with 'name' and one ends
            with '_data.csv' and the other with '_varinfo.csv'.
        """

        Path("folder").mkdir(parents=True, exist_ok=True)
        data_file = os.path.join(folder, name+"_data.csv")
        var_info_file = os.path.join(folder, name+"_varinfo.csv")
        self.data.to_csv(data_file, header=True)
        var_info = self.variable_info(compute_value_counts=False)
        var_info.loc[:,'valid_values'] = var_info['valid_values'].apply(json.dumps)
        var_info.loc[:,'invalid_values'] = var_info['invalid_values'].apply(json.dumps)
        var_info.to_csv(var_info_file, header=True)

    def copy(self):
        """Create a copy of this Survey. 
        
        The underlying data and the variables are copied, so changing one of the surveys does not affect the other.
        """

        survey = Survey(self.data)
        survey.variables = self.variables.apply(deepcopy)
        return survey

    @property
    def shape(self):
        """The shape of the data.

        Returns
        -------
        tuple[int, int]
            The number of respondents in the first entry and the number of variables (or questions) in the second entry.
        """

        return self.data.shape

    @property
    def num_questions(self):
        """Number of questions (or variables).

        Returns
        -------
        int
            The number of questions of this survey.
        """

        return self.data.shape[1]

    @property
    def num_respondents(self):
        """Number of respondents.

        Returns
        -------
        int
            The number of respondents of this survey.
        """

        return self.data.shape[0]

    def __getitem__(self, item):
        """Retrieve a data element/slice/subset/...

        Parameters
        ----------
        item: list, slice, subset selection, ...
            A selection of the data. Refer to the documentation of :meth:`pandas.DataFrame.__getitem__`.

        Returns
        -------
        various types
            A part of the data.
        """

        return self.data.__getitem__(item)

    def variable_info(self, column_selection: ColumnSelectionTypes = None,  compute_value_counts: bool = True) -> pd.DataFrame:
        """A pandas dataframe containing information about the variables (or questions).

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        compute_value_counts : bool
            If True, the resulting dataframe contains information about the number of valid answers for each question
            and the number of unique answers for each question.

        Returns
        -------
        :class:`pandas.DataFrame`
            Information about the selected variables.
        """

        column_selection = self.interpret_column_selection(column_selection)
        df = pd.DataFrame(index=column_selection, columns=["id", "name", "type", "label", "valid_values", "invalid_values", "is_usable"],
                          data=[[i]+self.variables[v].to_row() for i,v in enumerate(column_selection)])
        if compute_value_counts:
            num_valid_answers = pd.Series(index = df.index, data=0, dtype=int)
            unique_answers = pd.Series(index = df.index, data=0, dtype=int)
            for v in df.index:
                invalid_values = df.loc[v,'invalid_values']
                valid_answers = self.data.loc[~self.data[v].isin(invalid_values), v]
                num_valid_answers[v] = len(valid_answers)

                nunique = valid_answers.nunique()
                unique_answers[v] = nunique

            df['num_valid_answers'] = num_valid_answers
            df['num_unique_answers'] = unique_answers
        return df

    def set_pyreadstat_meta_data(self, meta, type_of_scale_variables: str = 'numeric', erase_existing: bool = False, suppress_check_var_warning: bool = False):
        """Use metadata returned from the python package pyreadstat to set properties of the variables (or questions).

        See `pyreadstat <https://github.com/Roche/pyreadstat>`_ for more information.

        Parameters
        ----------
        meta : pyreadstat metadata object type
            A pyreadstat metadata object.
        type_of_scale_variables : str
            If a variable is of type 'scale', replace 'scale' by this string.
        erase_existing : bool
            Whether the existing information about the variables should be overwritten.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables. A variable is unusable, if we can not
            automatically create features for the corresponding question.
        """

        if type_of_scale_variables and not SurveyVariable.is_valid_type_name(type_of_scale_variables):
            warn(f"Unknown type: type_of_scale_variables={type_of_scale_variables}")

        did_warn_var_type = False

        for i,c in enumerate(meta.column_names):
            variable = self.variables[c]
            variable.name = meta.column_names[i]
            variable.label = meta.column_labels[i]
            variable.type = meta.variable_measure[c]
            if variable.type == 'unknown':
                if not did_warn_var_type:
                    warn("Meta data does not contain type information. We use pyreadstat's internal type instead. Please change the types later...")
                    did_warn_var_type = True
                variable.type = meta.readstat_variable_types[c]

            if type_of_scale_variables and variable.type == "scale":
                variable.type = type_of_scale_variables

            labels = meta.variable_value_labels.get(c)
            if labels:
                if erase_existing:
                    variable.set_values({}, {})
                missing_ranges = [(r['lo'], r['hi']) for r in meta.missing_ranges.get(c,[])]
                valid, invalid = {}, {}
                for (num,label) in labels.items():
                    if any(num >= r[0] and num <= r[1] for r in missing_ranges):
                        invalid[num] = label
                    else:
                        valid[num] = label
                variable.add_values(valid, invalid)
        self._check_variables_and_warn_if_needed(suppress_check_var_warning)

    def select_questions(self, column_selection: ColumnSelectionTypes, suppress_check_var_warning=False) -> 'Survey':
        """A new survey containing a subset of the columns.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables. 
            A variable is unusable, if we can not automatically create features for the corresponding question.

        Returns
        -------
        :class:`Survey`
            The reduced survey.
        """

        columns = self.interpret_column_selection(column_selection)
        subset = Survey(self.data[columns])
        subset.variables = self.variables[columns].apply(deepcopy)
        subset._check_variables_and_warn_if_needed(suppress_check_var_warning)
        return subset

    def select_respondents(self,  row_selection:Union[list,np.ndarray,range,RowSelector], suppress_check_var_warning=True) -> 'Survey':
        """A new survey containing a subset of the rows.

        Parameters
        ----------
        row_selection : list , np.ndarray, range or RowSelector
            Specifies a subset of the rows (i.e. the respondents).
            A RowSelector is a ``Callable[[pd.Series], bool]`` specifying if the row given by the pandas.Series should
            be contained in the result.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables. 
            A variable is unusable, if we can not automatically create features for the corresponding question.

        Returns
        -------
        :class:`Survey`
            The reduced survey.
        """

        if isinstance(row_selection, Callable):
            rows = [r for r in self.data.index if row_selection(self.data.loc[r,:])]
        else:
            rows = self.data.index[row_selection]
        subset = Survey(self.data.loc[rows,:])
        subset.variables = self.variables.apply(deepcopy)
        subset._check_variables_and_warn_if_needed(suppress_check_var_warning)
        return subset


    def resample_respondents_to_balance_answers_of_question(self, col_name:str, sample_size_per_answer: int = 1000, value_ranges:list = None, suppress_check_var_warning=True):
        var = self.variables[col_name]
        col = self.data[col_name]
        if var.invalid_values:
            col = col[~col.isin(var.invalid_values_as_list())]

        samples = []
        if var.is_numeric_type():
            if value_ranges is None:
                raise ValueError(f"value_ranges have to be a list of Intervals for numeric variables, not {value_ranges}")
            if not isinstance(value_ranges[0], UnionOfIntervals):
                raise ValueError(f"value_ranges have to be a list of Intervals for numeric variables, not {value_ranges}")
            for r in value_ranges:
                sel = col.index[col.apply(lambda v: v in r)]
                if sel.any():
                    samples.extend(np.random.choice(sel, sample_size_per_answer, replace=True))
        else:
            if value_ranges is None:
                value_ranges = [{v} for v in var.valid_values_as_list()]
            for r in value_ranges:
                sel = col.index[col.isin(r)]
                if sel.any():
                    samples.extend(np.random.choice(sel, sample_size_per_answer, replace=True))

        resampled = Survey(self.data.loc[samples,:])
        resampled.variables = self.variables.apply(deepcopy)
        resampled._check_variables_and_warn_if_needed(suppress_check_var_warning)
        return resampled

    def summarize(self, column_selection: ColumnSelectionTypes = None) -> pd.DataFrame:
        """Create a summary of some interesting aspects of this survey.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing information about the variables and their data.
        """

        column_selection = self.interpret_column_selection(column_selection)

        column_names = ['name', 'type', 'label']
        if not all(self.variables[c].is_nominal_type() for c in column_selection):
            column_names.extend(['min','mean','median','max'])
        min_count_col_idx = len(column_names)
        if not all(self.variables[c].is_numeric_type() for c in column_selection):
            max_num_cats = max(len(v.valid_values) for c in column_selection if not (v := self.variables[c]).is_numeric_type())
            column_names.extend([f"count(value {i+1})" for i in range(0,max_num_cats)])
            column_names.append("count(total)")
        sum_count_col_idx = len(column_names)-1

        summary = pd.DataFrame(index=column_selection, columns=column_names,
                               data=[[(v:=self.variables[c]).name, v.type, v.label] + [np.nan]*(len(column_names)-3)
                                     for c in column_selection])
        for c in column_selection:
            if not (var := self.variables[c]).is_nominal_type():
                summary.loc[c,['min','mean','median','max']] = [(d := self.data.loc[~self.data[c].isin(var.invalid_values),c]).min(), d.mean(), d.median(), d.max()]
            if not var.is_numeric_type():
                summary.loc[c,column_names[min_count_col_idx:min_count_col_idx+len(var.valid_values)]] = [(self.data[c] == v).sum() for v in var.valid_values.keys()]
                summary.loc[c,column_names[sum_count_col_idx]] = summary.loc[c,column_names[min_count_col_idx:min_count_col_idx+len(var.valid_values)]].sum()
            else:
                summary.loc[c, column_names[sum_count_col_idx]] = self.data.shape[0] - self.data[c].isin(var.invalid_values).sum()


        return summary


    def replace_variable_names(self, names: Union[list, np.ndarray, pd.Series], column_selection:ColumnSelectionTypes = None):
        """Replace the names of selected variables.
        
        Note that this changes the index of the dataframe columns and variable info data structure of this object.

        Parameters
        ----------
        names : list, np.ndarray or pandas.Series
            New names.
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        """

        column_selection = self.interpret_column_selection(column_selection)
        if len(names) != len(column_selection):
            raise ValueError("list of names has wrong length")

        for v,n in zip(self.variables[column_selection], names):
            v.name = n
        new_index = pd.Series(index=self.variables.index, data=self.variables.index)
        new_index[column_selection] = names
        self.variables = self.variables.set_axis(new_index)
        self.data.columns = new_index


    def set_variable_labels(self, labels: Union[list, np.ndarray, pd.Series], column_selection:ColumnSelectionTypes = None):
        """Set the labels (which often is the question text) of selected variables.

        Parameters
        ----------
        labels : list, np.ndarray, pandas.Series
            New labels.
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        """

        column_selection = self.interpret_column_selection(column_selection)
        if len(labels) != len(column_selection):
            raise ValueError("list of labels has wrong length")

        for v,l in zip(self.variables[column_selection], labels):
            v.label = l

    def set_variable_value_lists(self, column_selection: ColumnSelectionTypes = None, valid_values: Optional[dict] = None, invalid_values: Optional[dict] = None,
                                 suppress_check_var_warning: bool = False):
        """Set the dictionaries of valid and invalid values (and their labels) for selected variables.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        valid_values : dict, optional
            A dictionary containing items ``(value, label)`` for the valid values of the selected variables.
        invalid_values : dict, optional
            A dictionary containing items ``(value, label)`` for the invalid values of the selected variables.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables after the change.
            A variable is unusable, if we can not automatically create features for the corresponding question.
        """

        column_selection = self.interpret_column_selection(column_selection)
        for v in self.variables[column_selection]:
            # TODO: should we check if we miss values in the list?
            v.set_values(valid_values, invalid_values)

        self._check_variables_and_warn_if_needed(suppress_check_var_warning=suppress_check_var_warning)

    def replace_variable_values(self, column_selection: ColumnSelectionTypes = None, mapping: Union[dict,list,None] = None,
                                change_type_to: Optional[str] = None, suppress_check_var_warning: bool = False):
        """Replace the values found in the selected columns by different values. 
        
        The corresponding value labels (in the `variable_info` of this object) are retained.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        mapping : dict, list or None
            An object providing replacement values.
            
            - If `mapping` is None, values are replaced by integers in order of the natural sorting order of the old values.
            - If `mapping` is a list, the values are mapped to their indices in this list.
            - If `mapping` is a dictionary, the values are mapped to the new values given in the dictionary.

        change_type_to : str, optional
            Optionally change the type of the changed columns.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables.
            A variable is unusable, if we can not automatically create features for the corresponding question.
        """

        target_type_is_number = True
        if isinstance(mapping, dict) and len(mapping)>0:
            target_type_is_number = isinstance(next(iter(mapping.values())), numbers.Number)

        column_selection = self.interpret_column_selection(column_selection)
        for c in column_selection:
            var = self.variables[c]
            unique_values = self.data[c].unique()

            if len(additional_values := set(unique_values) - (set(var.valid_values.keys()) | set(var.invalid_values.keys())))>0:
                warn("Variable takes values not in the lists: We treat them as valid values...")
                var.add_values(valid_values=zip(additional_values, [str(v) for v in additional_values]))

            if mapping is None:
                mapping = dict(zip(var.valid_values_as_list(), range(1,1+len(var.valid_values_as_list()))))
                mapping |= dict(zip(var.invalid_values_as_list(), range(-len(var.invalid_values_as_list()),0)))
            elif isinstance(mapping, list):
                mapping = dict(zip(mapping, range(1,len(mapping)+1)))
            elif not isinstance(mapping, dict):
                raise ValueError("mapping must be a list or a dict")

            unmapped_values = [v for v in var.all_values_as_list() if v not in mapping]
            if len(unmapped_values)>0:
                mapping = mapping.copy()
                min_target_value = min((min(mapping.values(), default=0) if target_type_is_number else min((int(v) for v in mapping.values()), default=0)),0) - 1
                max_target_value = max((max(mapping.values(), default=0) if target_type_is_number else max((int(v) for v in mapping.values()), default=0)),0) + 1
                for v in unmapped_values:
                    if v in var.valid_values:
                        mapping[v] = max_target_value if target_type_is_number else str(max_target_value)
                        max_target_value += 1
                    elif v in var.invalid_values:
                        mapping[v] = min_target_value if target_type_is_number else str(min_target_value)
                        min_target_value -= 1

            var.replace_values(mapping)
            self.data[c].replace(mapping, inplace=True)
            # this changes the dtype of the column in the dataframe - this is what we want if the funciton is used to encode a variable numerically -
            # it is not sure that this will happen in newer versions of pandas -> just to keep in mind...

            if change_type_to is not None:
                var.type = change_type_to

        self._check_variables_and_warn_if_needed(suppress_check_var_warning=suppress_check_var_warning)

    def replace_variable_value_labels(self, mapping: dict, column_selection: ColumnSelectionTypes = None):
        """Replace variable value labels according to the dictionary `mapping`.

        Parameter
        ---------
        mapping : dict
            A dictionary containing items ``(value, new_label)``.
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        """

        column_selection = self.interpret_column_selection(column_selection)
        for v in self.variables[column_selection]:
            v.set_value_labels(mapping)

    def complete_rows(self) -> np.ndarray:
        """Find out which rows are complete.
        
        A row is complete if the corresponding respondent gave a valid answer to all questions.

        Returns
        -------
        np.ndarray:
            A boolean array of shape (:attr:`num_respondents`, ) containing the entry True if and only if the 
            corresponding row does not contain invalid values.
        """

        complete = np.ones(self.data.shape[0], dtype=bool)
        for c in self.data.columns:
            complete &= ~self.data[c].isin(self.variables[c].invalid_values.keys())
        return complete

    def check_variables(self, verbose: bool = False) -> dict:
        """Check if we can conveniently use the information in this survey for a tangle analysis.

        Parameters
        ----------
        verbose : bool
            Whether a description should be printed to the console if an error is found.

        Returns
        -------
        dict
            A dictionary containing information about possible problems with the variables' metadata.
        """

        problem_info = {}
        for v in self.variables:
            if not v.is_valid_type():
                problem_info[v.name] = f"invalid type:  {v.type}"
            elif v.is_numeric_type():
                if not is_numeric_dtype(self.data[v.name]):
                    problem_info[v.name] = f"numeric variable takes non-numeric values"
                elif len(np.setdiff1d(self.data[v.name].unique(), v.invalid_values_as_list())) < 2:
                    problem_info[v.name] = "not enough valid values"
            else:
                if v.is_ordinal_type() and not is_numeric_dtype(self.data[v.name]):
                    problem_info[v.name] = f"ordinal variable takes non-numeric values"
                elif not v.valid_values:
                    problem_info[v.name] = f"list of valid values is missing"
                else:
                    unique_values = set(self.data[v.name].unique())
                    valid_values = set(v.valid_values.keys())
                    unique_values.difference_update(v.invalid_values.keys())
                    if len(unique_values) < 2:
                        problem_info[v.name] = "not enough valid values"
                    elif len(unique_values - valid_values)>0:
                        problem_info[v.name] = "variable takes unknown values"
            v.is_usable = v.name not in problem_info
            if verbose and not v.is_usable:
                print(f"variable '{v.name}' [{v.type}] -> PROBLEM: {problem_info[v.name]}")
        return problem_info

    def _check_variables_and_warn_if_needed(self, suppress_check_var_warning=False):
        if len(v := self.check_variables()):
            if not suppress_check_var_warning:
                warn(f"There might by unusable variables: {v}")

    def guess_missing_variable_types(self, column_selection: ColumnSelectionTypes = None, all_integer_variables_are_ordinal=False, suppress_check_var_warning=True):
        """Guess missing variable types from data.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        all_integer_variables_are_ordinal : bool
            If True, we assume all integer columns should correspond to an ordinal variable.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables. 
            A variable is unusable, if we can not automatically create features for the corresponding question.
        """

        warn("unfinished functionality... TODO: guess_missing_variable_types")
        columns = self.interpret_column_selection(column_selection)
        for i,c in enumerate(columns):
            variable = self.variables[c]
            if SurveyVariable.is_valid_type(variable.name):
                continue
            column = self.data[c]
            dtype_kind = column.dtype.kind
            if dtype_kind in 'b':   # binary
                variable.type = 'nominal'
            elif dtype_kind in 'iu' or (dtype_kind=='f' and ((column[~column.isna()] % 1) == 0).all()):
                # integer values: unfortunately, we cannot infer if the variable is ordinal, nominal or numeric
                variable.type = 'ordinal' if all_integer_variables_are_ordinal else 'nominal'
            elif dtype_kind == 'f':
                variable.type = 'numeric'
            elif dtype_kind in 'mM':    # date-time stuff is ordinal
                variable.type = 'ordinal'
            else:   # all others are nominal
                variable.type = 'nominal'
        self._check_variables_and_warn_if_needed(suppress_check_var_warning)

    def guess_missing_variable_ranges(self, column_selection: ColumnSelectionTypes = None,
                                      invalid_value_range: Optional[UnionOfIntervals] = UnionOfIntervals.negative_axis(),
                                      mark_added_labels: bool = True,
                                      overwrite_existing_information: bool = False,
                                      keep_existing_labels = True,
                                      suppress_check_var_warning: bool = True):
        """Guess missing variable value lists from data.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        invalid_value_range : :class:`SurveyVariable.UnionOfIntervals`
            Values inside this interval are considered to be missing values. 
            Often this is ``UnionOfIntervals(-np.inf, 0, False, True)``.
        mark_added_labels : bool
            If True, all labels inserted automatically are marked by appending the string '(*)'.
        overwrite_existing_information : bool
            Overwrite value range information that is already present.
        keep_existing_labels : bool
            Keep known labels, even if the value range information is overwritten.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables. 
            A variable is unusable, if we can not automatically create features for the corresponding question.
        """

        column_selection = self.interpret_column_selection(column_selection)
        for i,c in enumerate(column_selection):
            variable = self.variables[c]
            unique_values = self.data[c].unique()
            existing_labels = (variable.invalid_values | variable.valid_values) if keep_existing_labels else dict()
            if invalid_value_range is None:
                found_invalid_values = []
            else:
                found_invalid_values = list(unique_values[[v in invalid_value_range for v in unique_values]])

            invalid_labels = [existing_labels.get(v, None) or f"{v}" + ("(*)" if mark_added_labels else "") for v in found_invalid_values]
            if variable.is_numeric_type():
                valid_labels = found_valid_values = []
            else:
                found_valid_values = np.setdiff1d(unique_values, found_invalid_values + variable.invalid_values_as_list())
                valid_labels = [existing_labels.get(v, None) or f"{v}" + ("(*)" if mark_added_labels else "") for v in found_valid_values]

            if overwrite_existing_information:
                variable.set_values(valid_values=dict(zip(found_valid_values, valid_labels)),
                                    invalid_values=dict(zip(found_invalid_values, invalid_labels)))
            else:
                variable.add_values(valid_values=dict(zip(found_valid_values, valid_labels)),
                                    invalid_values=dict(zip(found_invalid_values, invalid_labels)))

        self._check_variables_and_warn_if_needed(suppress_check_var_warning)

    def set_variable_types(self, types: Union[str, list, np.ndarray], column_selection: ColumnSelectionTypes = None, suppress_check_var_warning=True):
        """Guess missing variable value lists from data.

        Parameters
        ----------
        types: str, list or np.ndarray
            If `type` is a string, a single variable type name. Otherwise a list of variable type names.
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        suppress_check_var_warning : bool
            If True, don't show a warning if there are 'unusable' variables. 
            A variable is unusable, if we can not automatically create features for the corresponding question.
        """

        column_selection = self.interpret_column_selection(column_selection)

        if isinstance(types, str):
            types = [types]*len(column_selection)

        if not suppress_check_var_warning and not all(SurveyVariable.is_valid_type_name(t) for t in types):
            warn("Apparently, you want to use an invalid type. Please try again or cope with the consequences...")

        for i,c in enumerate(column_selection):
            self.variables[c].type = types[i]

        self._check_variables_and_warn_if_needed(suppress_check_var_warning)


    def count_number_of_unique_answers(self, column_selection: ColumnSelectionTypes = None) -> pd.Series:
        """Count the number of unique answers for selected columns.

        Parameters
        ----------
        column_selection : str, int, list, np.ndarray, pd.Series, range, QuestionSelector or None
            If None, all columns are taken into account.
            Otherwise, a subset will be considered. The parameter is interpreted as described in :meth:`Survey.interpret_column_selection`.
        """
        col_sel = self.interpret_column_selection(column_selection)
        return self.data[col_sel].nunique() if len(col_sel)>1 else self.data[col_sel[0]].nunique()





    def count_valid_answers_per_respondent(self) -> np.ndarray:
        """Count the number of valid answers for each respondent.

        Returns
        -------
        np.ndarray
            An array of shape (:attr:`num_respondents`, ), where each row contains the number of questions answered
            validly by the respondent corresponding to that row.
        """

        data_copy = self.data.copy()
        for c in data_copy.columns:
            inv = list(self.variables[c].invalid_values.keys())
            data_copy[c].replace(inv, [np.nan]*len(inv), inplace=True)
        return data_copy.shape[1] - data_copy.isna().sum(axis=1)


    def interpret_column_selection(self, column_selection: ColumnSelectionTypes):
        """ Interpret different ways to select a subset of columns (or variables).

        Parameters
        ----------
        column_selection: str, int, list, np.ndarray, pandas.Series, range, QuestionSelector or None
            A specification of a selection of columns.
            The result is obtained in the following way:

            - if `column_selection` is None, the function chooses all columns of the survey.
            - if `column_selection` is a single integer, the function chooses the single column with this index.
            - if `column_selection` is a single string, the function chooses the single column with this name.
            - if `column_selection` is something arraylike, the function does the same as for single values, but for every value in the list.
            - if `column_selection` is a QuestionSelector (a ``Callable[[SurveyVariable], bool]``), the 
              QuestionSelector is called for every :class:`SurveyVariable` object and the :class:`SurveyVariable`
              is selected if QuestionSelector returns True.

        Returns
        -------
        list
            A selection of column names (i.e. variable names).
        """

        if column_selection is None:
            column_selection = self.data.columns
        elif isinstance(column_selection, int):
            column_selection = [self.data.columns[column_selection]]
        elif isinstance(column_selection, str):
            column_selection = [column_selection]
        elif isinstance(column_selection, Callable):
            column_selection = [c.name for c in self.variables if column_selection(c)]
        elif isinstance(column_selection, (list, np.ndarray, pd.Series, range)):
            if isinstance(column_selection[0], (int, bool, np.bool_)):
                column_selection = self.data.columns[column_selection]
        else:
            raise ValueError("sorry, don't understand your columns...")
        return column_selection

######################################################################
# private

    def _replace_values_by_labels(self, df, inplace=True):
        if not inplace:
            df = df.copy()
        for col in df.columns:
            var_info = self.variables[col]
            df.loc[:, col] = df[col].apply(lambda x: str(f"'{var_info.valid_values.get(v, None)}'" or v for v in x)
            if isinstance(x, tuple) else var_info.valid_values.get(x, None) or x)
        df.rename(columns={v.name:(v.label or v.name) for v in self.variables}, inplace=True)
        if not inplace:
            return df


    def _convert_non_numeric_columns_to_str(self):
        for v in self.variables:
            if not is_numeric_dtype(self.data.dtypes[v.name]):
                str_col = self.data[v.name].apply(str)
                self.data[v.name] = str_col.astype(object)

    def _replace_nan(self):
        for v in self.variables:
            col_data = self.data[v.name]
            if (isna := (col_data.isna() | (col_data == 'nan'))).any():
                if is_numeric_dtype(self.data.dtypes[v.name]):
                    na = min(col_data.min(), min(v.invalid_values.keys(), default=0)) - 1
                else:
                    na, trial = "NaN", 0
                    while (col_data == na).any() or na in v.invalid_values:
                        trial += 1
                        na = "NaN_"+ str(trial)
                self.data.loc[isna,v.name] = na
                v.add_values(valid_values=None, invalid_values={na: "NaN"})

