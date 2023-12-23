from .term import (
    LogicTerm,
    ConjunctiveNormalForm,
    DisjunctiveNormalForm,
    TextTerm,
)
from .np_term_operations import simplify, distribute, append
from .sep_to_term import to_term, array_to_term

__all__ = [
    "LogicTerm",
    "ConjunctiveNormalForm",
    "DisjunctiveNormalForm",
    "TextTerm",
    "simplify",
    "distribute",
    "append",
    "to_term",
    "array_to_term",
]
