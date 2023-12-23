import numpy as np

from tangles.util.logic.term import TextTerm, SemanticTextTerm
from tangles._typing import OrientedSep


def to_term(sep_sys, sep: OrientedSep, base_seps: list[int]):
    pass


def array_to_term(
    sep: np.ndarray, features: np.ndarray, feature_labels: list[str]
) -> TextTerm:
    rec_log = RecursionLogic(features, feature_labels, sep)
    starting_approximation = np.ones(sep.shape, dtype=np.int8)
    return array_to_term_recursive(
        sep=sep,
        approximation=starting_approximation,
        next_term=SemanticTextTerm(TextTerm("true"), starting_approximation),
        rec_log=rec_log,
    ).text


def array_to_term_recursive(
    sep: np.ndarray,
    approximation: np.ndarray,
    next_term: SemanticTextTerm,
    rec_log: "RecursionLogic",
) -> SemanticTextTerm:
    next_sep = np.minimum(sep, next_term.array)
    next_approx = np.minimum(approximation, next_term.array)

    if np.all(next_approx == next_sep):
        return next_term
    if np.all(next_sep == -1):
        return SemanticTextTerm(TextTerm("false"), -np.ones(sep.shape, dtype=np.int8))

    new_term = rec_log.find_best_term_extension(sep=next_sep, term=next_approx)

    first_term = array_to_term_recursive(
        sep=next_sep,
        approximation=next_approx,
        next_term=new_term,
        rec_log=rec_log,
    )
    second_term = array_to_term_recursive(
        sep=next_sep,
        approximation=next_approx,
        next_term=new_term.not_(),
        rec_log=rec_log,
    )
    or_term = rec_log.or_term(first_term, second_term, next_term)
    and_term = rec_log.and_term(next_term, or_term, approximation)
    return and_term


class RecursionLogic:
    def __init__(
        self, features: np.ndarray, feature_labels: list[str], og_sep: np.ndarray
    ):
        self._features = features
        self._og_sep = og_sep
        self._terms = [
            SemanticTextTerm(TextTerm(feature_labels[i]), features[:, i])
            for i in range(features.shape[1])
        ]

    def find_best_term_extension(
        self, sep: np.ndarray, term: np.ndarray
    ) -> SemanticTextTerm:
        mask_ab = np.minimum(term, -sep) == 1
        mask_cd = np.minimum(term, sep) == 1
        a_ar = np.sum(self._features[mask_ab] == 1, axis=0)
        b_ar = np.sum(-self._features[mask_ab] == 1, axis=0)
        c_ar = np.sum(self._features[mask_cd] == 1, axis=0)
        d_ar = np.sum(-self._features[mask_cd] == 1, axis=0)

        nested_bias = np.maximum(a_ar * (c_ar == 0), b_ar * (d_ar == 0))
        if np.any(nested_bias) > 0:
            return self._terms[np.argmax(nested_bias)]
        scores_1 = np.zeros(self._features.shape[1], dtype=np.float_)
        scores_1[c_ar != 0] = a_ar[c_ar != 0] / c_ar[c_ar != 0]
        scores_2 = np.zeros(self._features.shape[1], dtype=np.float_)
        scores_2[d_ar != 0] = b_ar[d_ar != 0] / d_ar[d_ar != 0]
        return self._terms[np.argmax(np.maximum(scores_1, scores_2))]

    def or_term(
        self,
        first_term: SemanticTextTerm,
        second_term: SemanticTextTerm,
        text_term: SemanticTextTerm,
    ) -> SemanticTextTerm:
        if np.all(
            np.minimum(first_term.array, text_term.array)
            <= np.minimum(second_term.array, text_term.array)
        ):
            return second_term
        elif np.all(
            np.minimum(second_term.array, text_term.array)
            <= np.minimum(first_term.array, text_term.array)
        ):
            return first_term
        elif second_term.array.sum() <= first_term.array.sum():
            return first_term.or_(second_term)
        return second_term.or_(first_term)

    def and_term(
        self,
        text_term: SemanticTextTerm,
        or_term: SemanticTextTerm,
        approximation: np.ndarray,
    ) -> SemanticTextTerm:
        if np.all(np.minimum(or_term.array, approximation) <= self._og_sep):
            return or_term
        return text_term.and_(or_term)
