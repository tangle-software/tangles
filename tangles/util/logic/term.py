"""
A logcial term made up of "conjunctions of clauses which themselves are disjunctions of literals" is said to be in

    Conjuctive Normal Form (CNF),

i.e. (A or not B or C) and (B or not C).

On the other hand a disjunction of conjunctions is said to be in

    Disjunctive Normal Form (DNF),

i.e. (A and B) or C or (not A).

We encode both of these forms in matrices. Each row represents one of the variables, each column one of the clauses.
Thus (A or not B or C) and (B or not C) would be encoded as
::

    ( 1  0)
    (-1  1)
    ( 1 -1)

On the other hand (A and B) or C or (not A) would be encoded as
::

    (1 0 -1)
    (1 0  0)
    (0 1  0)

As you can see, it is not possible from the matrix to tell which of these two normal forms the matrices are in.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from tangles.util.logic.np_term_operations import simplify, append, distribute


class LogicTerm(ABC):
    @abstractmethod
    def and_(self, other_term: "LogicTerm") -> "LogicTerm":
        pass

    @abstractmethod
    def or_(self, other_term: "LogicTerm") -> "LogicTerm":
        pass

    @abstractmethod
    def not_(self) -> "LogicTerm":
        pass


class DisjunctiveNormalForm(LogicTerm):
    def __init__(self, matrix: np.ndarray, row_labels: list[str]):
        self._matrix = matrix
        self._labels = row_labels

    def _and(self, other_term: "DisjunctiveNormalForm") -> "DisjunctiveNormalForm":
        return DisjunctiveNormalForm(
            simplify(distribute(self._matrix, other_term._matrix)), self._labels
        )

    def _or(self, other_term: "DisjunctiveNormalForm") -> "DisjunctiveNormalForm":
        return DisjunctiveNormalForm(
            simplify(append(self._matrix, other_term._matrix)), self._labels
        )

    def _not(self) -> "ConjunctiveNormalForm":
        return ConjunctiveNormalForm(-self._matrix, self._labels)

    def __repr__(self) -> str:
        intersection_terms = []
        for column in self._matrix.T:
            intersection_term = dnf_column_to_str(column, self._labels)
            if "and" in intersection_term:
                intersection_term = f"({intersection_term})"
            intersection_terms.append(intersection_term)
        return " or ".join(intersection_terms)


def dnf_column_to_str(column, row_labels):
    labels = []
    for i, specification in enumerate(column):
        if specification != 0:
            negation = "¬" if specification == -1 else ""
            labels.append(f"{negation}{row_labels[i]}")
    return " and ".join(labels)


class ConjunctiveNormalForm(LogicTerm):
    def __init__(self, matrix: np.ndarray, row_labels: list[str]):
        self._matrix = matrix
        self._labels = row_labels

    def _and(self, other_term: "ConjunctiveNormalForm") -> "ConjunctiveNormalForm":
        return ConjunctiveNormalForm(
            simplify(append(self._matrix, other_term._matrix)), self._labels
        )

    def _or(self, other_term: "ConjunctiveNormalForm") -> "ConjunctiveNormalForm":
        return ConjunctiveNormalForm(
            simplify(distribute(self._matrix, other_term._matrix)), self._labels
        )

    def _not(self) -> "DisjunctiveNormalForm":
        return DisjunctiveNormalForm(-self._matrix, self._labels)

    def __repr__(self) -> str:
        intersection_terms = []
        for column in self._matrix.T:
            intersection_term = cnf_column_to_str(column, self._labels)
            if "or" in intersection_term:
                intersection_term = f"({intersection_term})"
            intersection_terms.append(intersection_term)
        return " and ".join(intersection_terms)


def cnf_column_to_str(column, row_labels):
    labels = []
    for i, specification in enumerate(column):
        if specification != 0:
            negation = "¬" if specification == -1 else ""
            labels.append(f"{negation}{row_labels[i]}")
    return " or ".join(labels)


class TextTerm(LogicTerm):
    def __init__(self, text: str, _outer_operation: Optional[str] = None):
        self._text = text
        self._outer_operation = "" if not _outer_operation else _outer_operation

    def and_(self, other_term: "TextTerm") -> "TextTerm":
        if self._text == "true":
            return other_term
        if other_term._text == "true":
            return self

        if self._text == "false":
            return self
        if other_term._text == "false":
            return other_term

        first_term_text = (
            self._text if self._outer_operation != "or" else f"({self._text})"
        )
        second_term_text = (
            other_term._text
            if other_term._outer_operation != "or"
            else f"({other_term._text})"
        )
        return TextTerm(
            text=f"{first_term_text} and {second_term_text}",
            _outer_operation="and",
        )

    def or_(self, other_term: "TextTerm") -> "TextTerm":
        if self._text == "false":
            return other_term
        if other_term._text == "false":
            return self

        if self._text == "true":
            return self
        if other_term._text == "true":
            return other_term

        first_term_text = (
            self._text if self._outer_operation != "and" else f"({self._text})"
        )
        second_term_text = (
            other_term._text
            if other_term._outer_operation != "and"
            else f"({other_term._text})"
        )
        return TextTerm(
            text=f"{first_term_text} or {second_term_text}",
            _outer_operation="or",
        )

    def not_(self) -> "TextTerm":
        if self._text[0] == "¬":
            return TextTerm(self._text[1:], _outer_operation=self._outer_operation)
        if self._outer_operation == "":
            return TextTerm(f"¬{self._text}")
        return TextTerm(f"¬({self._text})", _outer_operation=self._outer_operation)

    def __repr__(self) -> str:
        return self._text


class SemanticTextTerm(LogicTerm):
    def __init__(self, text: TextTerm, array: np.ndarray):
        self.text = text
        self.array = array

    def and_(self, other_term: "SemanticTextTerm") -> "SemanticTextTerm":
        return SemanticTextTerm(
            text=self.text.and_(other_term.text),
            array=np.minimum(self.array, other_term.array),
        )

    def or_(self, other_term: "SemanticTextTerm") -> "SemanticTextTerm":
        return SemanticTextTerm(
            text=self.text.or_(other_term.text),
            array=np.maximum(self.array, other_term.array),
        )

    def not_(self) -> "SemanticTextTerm":
        return SemanticTextTerm(text=self.text.not_(), array=-self.array)
