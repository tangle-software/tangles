from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import numpy as np
from collections.abc import Iterable
import numbers


class UnionOfIntervals:

    @staticmethod
    def all():
        return UnionOfIntervals(-np.inf, np.inf, False, False)

    @staticmethod
    def nothing():
        return UnionOfIntervals(0, 0, False, False)

    @staticmethod
    def positive_axis(include_zero: bool = False):
        return UnionOfIntervals(0, np.inf, include_zero, False)

    @staticmethod
    def negative_axis(include_zero: bool = False):
        return UnionOfIntervals(-np.inf, 0, False, include_zero)

    @staticmethod
    def create_with_isolated_points(iterable:Iterable):
        iterator = iter(iterable)
        v = next(iterator, None)
        if v is None:
            return UnionOfIntervals.nothing()
        res_ptr = res = UnionOfIntervals(v, v, True, True)
        while (v := next(iterator, None)) is not None:
            res_ptr.next = UnionOfIntervals(v, v, True, True, prev=res_ptr)
            res_ptr = res_ptr.next
        return res

    @staticmethod
    def create_with_tuples(tuples: list): # it's a union :-)
        res = UnionOfIntervals(*tuples[0])
        for t in tuples[1:]:
            res = res | UnionOfIntervals(*t)
        return res

    def __init__(self, min=0, max=0, min_incl=False, max_incl=False, prev=None, next=None):
        self.min, self.max, self.min_incl, self.max_incl = min, max, min_incl, max_incl
        self.prev, self.next = prev, next

    def __hash__(self):
        if self.is_empty():
            return hash("empty")
        if self.is_all():
            return hash("all")
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, UnionOfIntervals):
            return False
        if self.is_empty() and other.is_empty():
            return True
        if self.is_all() and other.is_all():
            return True

        return (self.min==other.min and self.max==other.max and self.min_incl==other.min_incl and self.max_incl==other.max_incl and
                self.next == other.next) # we ignore prev, to be able to check tails (does anyone need this...? ;-) )

    def __contains__(self,p):
        if not isinstance(p, numbers.Number):
            return False
        if self.min < p < self.max:
            return True
        if self.min_incl and p == self.min:
            return True
        if self.max_incl and p == self.max:
            return True
        if self.next:
            return self.next.__contains__(p)
        return False


    def __str__(self):
        a, s = self, ""
        while True:
            s += f"{'[' if a.min_incl else '('}{a.min},{a.max}{']' if a.max_incl else ')'}"
            if (a := a.next) is None: break
            s += ' u '
        return s

    def __iter__(self):
        a = self
        while a:
            yield a
            a = a.next

    def __len__(self):
        a, l = self, 1
        while (a := a.next): l += 1
        return l

    def points(self):
        ptr = self
        while ptr:
            yield ptr.min, ptr.min_incl
            yield ptr.max, ptr.max_incl
            ptr = ptr.next


    def is_empty(self):
        return self.min > self.max or self.min == self.max and not (self.min_incl and self.max_incl)

    def is_all(self):
        return self.min == -np.inf and self.max == np.inf

    def __invert__(self):
        if self.is_empty():
            return UnionOfIntervals.all()
        if self.is_all():
            return UnionOfIntervals.nothing()

        plist = list(self.points())
        if plist[0][0] == -np.inf:
            plist = plist[1:]
        else:
            plist.insert(0,(-np.inf, True))
        if plist[-1][0] == np.inf:
            plist = plist[:-1]
        else:
            plist.append((np.inf, True))
        res = res_ptr = UnionOfIntervals(plist[0][0], plist[1][0], not plist[0][1], not plist[1][1])
        for i in range(2,len(plist), 2):
            res_ptr.next = UnionOfIntervals(plist[i][0], plist[i + 1][0], not plist[i][1], not plist[i + 1][1], prev = res_ptr)
            res_ptr = res_ptr.next
        return res

    def __and__(self, other: 'UnionOfIntervals'):
        res, res_it, i1, i2 = None, None, self, other
        while i1 and i2:
            lo, lo_incl = (i1.min, i1.min_incl) if i1.min > i2.min else (i2.min, i2.min_incl) if i1.min < i2.min else (i1.min, i1.min_incl and i2.min_incl)
            hi, hi_incl = (i1.max, i1.max_incl) if i1.max < i2.max else (i2.max, i2.max_incl) if i1.max > i2.max else (i1.max, i1.max_incl and i2.max_incl)
            if lo < hi or lo == hi and (lo_incl or hi_incl):
                if res_it:
                    res_it.next = UnionOfIntervals(lo, hi, lo_incl, hi_incl, prev=res_it)
                    res_it = res_it.next
                else:
                    res = UnionOfIntervals(lo, hi, lo_incl, hi_incl)
                    res_it = res

            if i1.max < i2.max:
                i1 = i1.next
            else:
                i2 = i2.next
        return res if res else UnionOfIntervals.nothing()

    def __or__(self, other: 'UnionOfIntervals'):
        return ~((~self) & (~other))

    def __sub__(self, other: 'UnionOfIntervals'):
        return self & (~other)

    def __xor__(self, other: 'UnionOfIntervals'):
        return (self-other) | (other-self)






class SurveyVariable(ABC):
    """A variable (column) of a :class:`Survey`.

    Parameters
    ----------
    name : str
        The name of the variable.
    type : str
        The type of the variable.
    label : str
        The label of the variable (often the question text or some brief version of it).
    valid_values : dict
        A dictionary containing the possible 'raw' valid values as keys and (possibly meaningful) labels as values.
    invalid_values : dict
        A dictionary containing the possible 'raw' invalid values as keys and (possibly meaningful) labels as values.
    is_usable : bool
        A flag that is True, if we are sure that we can automatically create nice features for this question.
    """

    numeric_types = ['scale', 'continuous', 'numeric', 'numerical', 'conti', 'contin', 'cont', 'num', 'numi']
    """possible names of numeric variables
    """

    ordinal_types = ['ordinal', 'ord', 'ordi','ordin']
    """possible names of ordinal variables
    """

    nominal_types = ['nominal', 'categorial', 'categorical', 'nom', 'nomi', 'nomin', 'cat', 'binary', 'bin']
    """possible names of nominal variables
    """

    @staticmethod
    def is_valid_type_name(type_name: str) -> bool:
        """A static function that checks if a string is a valid variable typename.

        Parameters
        ----------
        type_name : str
            A string in question.

        Returns
        -------
        bool
            Whether the string is a valid.
        """

        return (type_name in SurveyVariable.numeric_types or
                type_name in SurveyVariable.ordinal_types or
                type_name in SurveyVariable.nominal_types)

    def __init__(self, name: str, type: str = 'unknown', label: Optional[str] = None, valid_values: Optional[dict] = None, invalid_values:Optional[dict] = None, is_usable: bool = False):
        self.name = name
        self.type = type
        self.label = label
        self.valid_values = dict(sorted(valid_values.items())) if valid_values else {}
        self.invalid_values = dict(sorted(invalid_values.items())) if invalid_values else {}
        self.is_usable = is_usable

    def set_values(self, valid_values:Optional[dict] = None, invalid_values: Optional[dict] = None):
        """Set this survey variable's valid and invalid values.

        Parameters
        ----------
        valid_values : dict
            The valid values to add, in a dict with items ``(raw value, label)``.
        invalid_values :  dict
            The invalid values to add, in a dict with items ``(raw value, label)``.
        """

        if valid_values is not None:
            self.valid_values = dict(sorted(valid_values.items()))
        if invalid_values is not None:
            self.invalid_values = dict(sorted(invalid_values.items()))

    def add_values(self, valid_values: Optional[dict] = None, invalid_values: Optional[dict] = None):
        """Add `valid_values` and `invalid_values` to this survey variable's valid and invalid values, respectively.

        Parameters
        ----------
        valid_values : dict
            The valid values to add, in a dict with items ``(raw value, label)``. Existing valid values are not overwritten.
        invalid_values :  dict
            The invalid values to add, in a dict with items ``(raw value, label)``. Existing invalid values are not overwritten.
        """

        if valid_values is not None:
            self.valid_values = dict(sorted((valid_values | self.valid_values).items()))
        if invalid_values is not None:
            self.invalid_values = dict(sorted((invalid_values | self.invalid_values).items()))

    def replace_values(self, mapping: dict):
        """Replace values by other values. The labels are retained.

        Parameters
        ----------
        mapping : dict
            The mapping.
        """

        self.valid_values = dict(sorted((mapping.get(v) or v,l) for v,l in self.valid_values.items()))
        self.invalid_values = dict(sorted((mapping.get(v) or v,l) for v, l in self.invalid_values.items()))

    def set_value_labels(self, mapping: dict):
        """Replace value labels by other labels, if the corresponding values are in the mapping.

        Parameters
        ----------
        mapping : dict
            The mapping.
        """

        self.valid_values = dict(sorted((v, mapping.get(v) or l) for v, l in self.valid_values.items()))
        self.invalid_values = dict(sorted((v, mapping.get(v) or l) for v, l in self.invalid_values.items()))

    def is_valid_type(self) -> bool:
        """Check the validity of this variable's type string.

        Returns
        -------
        bool
            Whether the type of this variable is valid.
        """

        return SurveyVariable.is_valid_type_name(self.type)


    def is_nominal_type(self):
        """Check if this variable is nominal.

        Returns
        -------
        bool
            Whether the type of this variable is nominal.
        """

        return self.type in SurveyVariable.nominal_types

    def is_ordinal_type(self):
        """Check if this variable is ordinal.

        Returns
        -------
        bool
            Whether the type of this variable is ordinal.
        """

        return self.type in SurveyVariable.ordinal_types

    def is_numeric_type(self):
        """Check if this variable is numeric.

        Returns
        -------
        bool
            Whether the type of this variable is numeric.
        """

        return self.type in SurveyVariable.numeric_types

    def value_class(self):
        """
        Return the :class:`SurveyVariableValues` subclass that this variable is an instance of.
        """

        if self.is_nominal_type():
            return NominalVariableValues
        elif self.is_ordinal_type():
            return OrdinalVariableValues
        elif self.is_numeric_type():
            return NumericalVariableValues
        else:
            return None

    def is_allowed_operation(self, op):
        if (cl := self.value_class()) == None:
            return False
        return op in cl.allowed_operations


    def create_values(self) -> 'SurveyVariableValues':
        """Create a :class:`SurveyVariableValues` object for this variable.

        Returns
        -------
        :class:`SurveyVariableValues`
            A :class:`SurveyVariableValues` object.
        """

        if (cl := self.value_class()) == None:
            raise ValueError(f"unknown variable type: {self.type}")
        return cl(self)

    def valid_values_as_list(self) -> list:
        """Return the (raw) valid values as a list.

        Returns
        -------
        list
            A list of valid values.
        """

        return list(self.valid_values.keys())

    def invalid_values_as_list(self) -> list:
        """Return the (raw) invalid values as a list.

        Returns
        -------
        list
            A list of invalid values.
        """

        return list(self.invalid_values.keys())

    def all_values_as_list(self) -> list:
        """Return all (raw) values as a list.

        Returns
        -------
        list
            A list of all valid and invalid values.
        """

        return self.valid_values_as_list() + self.invalid_values_as_list()

    def to_row(self) -> list:
        """Return the information contained in this object as a row.
        
        The returned row can be inserted in a :class:`pandas.DataFrame`.

        Returns
        -------
        list
            A list of length 6 containing the name, the type, the label, the valid values, the invalid values 
            and whether this variable is usable.
        """

        return [self.name, self.type, self.label, self.valid_values, self.invalid_values, self.is_usable]

    def __str__(self) -> str:
        """Convert this SurveyVariable's information to a string.
        
        The information contains the name, the type, the label, the valid values, the invalid values 
        and whether this variable is usable.

        Returns
        -------
        str
            A string representation of this object's information.
        """

        return str(self.to_row())

    def __copy__(self) -> 'SurveyVariable':
        """Copy this object.

        Returns
        -------
        :class:`SurveyVariable`:
            A copy of this object.
        """

        return SurveyVariable(self.name, self.type, self.label, self.valid_values.copy(), self.invalid_values.copy(), self.is_usable)

class SurveyVariableValues:
    """ This class manages the interaction of variables (and their lists of values) and features (or separations). 

    Objects of this class represent the values a survey variable can take after all or a part of the corresponding
    features (or separations) are specified (or oriented) by a tangle.
    
    For example: We assume we have a tangle that contains some specified features corresponding to this variable. 
    If an ordinal variable can take values in :math:`\{1, \dots, 10\}` and our tangle contains two specified features,
    one containing points with values greater than :math:`4` and one containing points with values smaller than 
    :math:`7`, then the remaining possible values of this variable in our tangle are :math:`\{5,6\}`.

    Parameters
    ----------
    var : class:`SurveyVariable`
        The corresponding survey variable.
    """

    def __init__(self, var: SurveyVariable):
        self.var = var

    def __str__(self) -> str:
        """ String representation.

        Returns
        -------
        str
            A string representation of this variable's possible values.
        """
        
        return f"{self.var.name} [{self.var.type}]: values in {self.possible_values_representation}"

    def update_values_for_specification(self, sep_orientation: int, sep_metadata: Tuple, metadata_orientation: int):
        """ Update the list of possible values a variable can take.
        
        This function is called after a corresponding feature has been specified.

        Parameters
        ----------
        sep_orientation : int
            Either -1 or 1, as an indication of the specification of a separation.
        sep_metadata : tuple
            The metadata corresponding to a separation (or feature).
        metadata_orientation : int
            The specification (or orientation) associated with the metadata.
        """

        if sep_orientation == 0:
            return
        if sep_orientation != metadata_orientation:
            sep_metadata = (sep_metadata[0], SurveyVariableValues.invert_op(sep_metadata[1]), sep_metadata[2])
        self._update_possible_values_impl(sep_metadata)

    @abstractmethod
    def clear_possible_values(self):
        pass

    @abstractmethod
    def possible_values_representation(self, insert_labels:  bool = False):
        """ Return a representation of the possible values the corresponding variable can take (according to specifications seen so far).

        Abstract function, to be implemented by specialised subclasses.

        Parameters
        ----------
        insert_labels : bool
            If True, the variable name and the values are replaced by the corresponding labels.
        """

        pass

    @abstractmethod
    def _update_possible_values_impl(self, sep_metadata: Tuple):
        pass

    @staticmethod
    def invert_op(op: str):
        """Invert an operation contained in a metadata tuple.

        Parameter
        ---------
        op : str
            A string representing an operation.

        Returns
        -------
        str
            The inverted operation.
        """

        if op == '==': return '!='
        elif op == '!=': return '=='
        elif op == '<': return  '>='
        elif op == '>': return '<='
        elif op == '<=': return '>'
        elif op == '>=': return '<'
        elif op == 'in': return 'not in'
        elif op == 'not in': return 'in'
        else:
            raise ValueError(f"invert_op: don't understand operation: {op}")

class NominalVariableValues(SurveyVariableValues):
    """ A nominal :class:`SurveyVariableValues` class.

    The values a nominal variable can take is represented as set of possible values. 
    This set is taking specifications of features corresponding to this variable into account, making it a subset of
    all possible values of this variable. 

    Parameters
    ----------
    var : :class:`SurveyVariable`
        Corresponding survey variable.
    """

    allowed_operations = {'==','!=','in', 'not in'}
    """operations allowed for a nominal variable's values list"""

    def __init__(self, var: SurveyVariable):
        super().__init__(var)
        self.possible_values = set(var.valid_values)

    def clear_possible_values(self):
        self.possible_values = set()

    def possible_values_representation(self, insert_labels:  bool = False):
        sorted_values = sorted(self.possible_values)
        if insert_labels:
            if len(sorted_values) == 0: # happens in "black hole" tangles
                labels = "[]"
            elif len(sorted_values) == 1:
                labels = f"{self.var.valid_values.get(sorted_values[0], None) or sorted_values[0]}"
            else:
                labels = f"[{self.var.valid_values.get(sorted_values[0], None) or sorted_values[0]}"
                for v in sorted_values[1:-1]:
                    labels += f"; {self.var.valid_values.get(v,None) or v}"
                labels += f"; {self.var.valid_values.get(sorted_values[-1], None) or sorted_values[-1]}]"
            return labels
        else:
            return tuple(sorted_values)

    def _update_possible_values_impl(self, sep_metadata: Tuple):
        assert(sep_metadata[0] == self.var.name)
        if (op := sep_metadata[1]) == '==':
            self.possible_values = {sep_metadata[2]}
        elif op == '!=':
            self.possible_values.difference_update({sep_metadata[2]})
        elif op == 'in':
            self.possible_values.intersection_update(sep_metadata[2])
        elif op == 'not in':
            self.possible_values.difference_update(sep_metadata[2])
        else:
            raise ValueError(f"operation '{op}' not allowed for nominal variable")


class OrdinalVariableValues(NominalVariableValues):
    """ An ordinal :class:`SurveyVariableValues` class.

    The values a nominal variable can take is represented as set of possible values. 
    This set is taking specifications of features corresponding to this variable into account, making it a subset of
    all possible values of this variable. 
    
    Parameters
    ----------
    var : :class:`SurveyVariable`
        Corresponding survey variable.
    """

    allowed_operations = NominalVariableValues.allowed_operations | {'<', '<=', '>', '>='}
    """operations allowed for a ordinal variable's values list"""

    def __init__(self, var: SurveyVariable):
        super().__init__(var)

    def possible_values_representation(self, insert_labels: bool = False):
        if insert_labels:
            num_poss_answers = len(self.possible_values)
            if num_poss_answers == 0:
                return "[]"
            all_pos_v_sorted = sorted(self.var.valid_values.keys())
            min_answer, max_answer = min(self.possible_values), max(self.possible_values)
            if num_poss_answers>2 and all_pos_v_sorted.index(max_answer)-all_pos_v_sorted.index(min_answer) == num_poss_answers-1:
                return f"[{self.var.valid_values[min_answer]}; ... ; {self.var.valid_values[max_answer]}]"
            else:
                return super().possible_values_representation(True)
        else:
            return super().possible_values_representation(False)

    def _update_possible_values_impl(self, sep_metadata: Tuple):
        if sep_metadata[1] in NominalVariableValues.allowed_operations:
            return super()._update_possible_values_impl(sep_metadata)

        if (op := sep_metadata[1]) == '<':
            self.possible_values = {v for v in self.possible_values if v < sep_metadata[2]}
        elif op == '<=':
            self.possible_values = {v for v in self.possible_values if v <= sep_metadata[2]}
        elif op == '>':
            self.possible_values = {v for v in self.possible_values if v > sep_metadata[2]}
        elif op == '>=':
            self.possible_values = {v for v in self.possible_values if v >= sep_metadata[2]}
        else:
            raise ValueError(f"operation '{op}' not allowed for ordinal variable")


class NumericalVariableValues(SurveyVariableValues):
    """ A numeric :class:`SurveyVariableValues` class.
    
    The values a nominal variable can take is represented as set of possible values. 
    This set is taking specifications of features corresponding to this variable into account, making it a subset of
    all possible values of this variable. 

    Parameters
    ----------
    var : :class:`SurveyVariable`
        Corresponding survey variable.
    """

    allowed_operations = {'<', '<=', '>', '>=', '==', '!=', 'in', 'not in'}  # is it enough?
    """operations allowed for a numerical variable's values range"""

    def __init__(self, var: SurveyVariable):
        super().__init__(var)
        self.intervals = UnionOfIntervals(-np.inf, np.inf)

    def __str__(self):
        return f"{self.var.name} [{self.var.type}]: values in {self.possible_values_representation()}"

    def clear_possible_values(self):
        self.intervals = UnionOfIntervals.nothing()

    def possible_values_representation(self, insert_labels: bool = False):
        if insert_labels:
            return str(self.intervals)
        else:
            return self.intervals

    def _update_possible_values_impl(self, sep_metadata: Tuple):
        assert(sep_metadata[0] == self.var.name)
        op = sep_metadata[1]
        if op == '==':
            self.intervals = UnionOfIntervals(sep_metadata[2], sep_metadata[2], True, True)
        elif op == '!=':
            self.intervals -= UnionOfIntervals(sep_metadata[2], sep_metadata[2], True, True)
        elif op[0] == '<':
            self.intervals &=  UnionOfIntervals(-np.inf, sep_metadata[2], False, len(op) > 1 and op[1] == '=')
        elif op[0] == '>':
            self.intervals &= UnionOfIntervals(sep_metadata[2], np.inf, len(op) > 1 and op[1] == '=', False)
        elif op == 'in':
            sep_metadata_value = sep_metadata[2] if isinstance(sep_metadata[2], UnionOfIntervals) else UnionOfIntervals.create_with_isolated_points(sep_metadata[2])
            self.intervals &= sep_metadata_value
        elif op == 'not in':
            sep_metadata_value = sep_metadata[2] if isinstance(sep_metadata[2], UnionOfIntervals) else UnionOfIntervals.create_with_isolated_points(sep_metadata[2])
            self.intervals -= sep_metadata_value
        else:
            raise ValueError(f"operation '{op}' not allowed for scale variable {self.var.name}")

