import pytest
import numpy as np
import pandas as pd
from tangles.convenience import (
    Survey,
    SurveyVariable,
    UnionOfIntervals,
    OrdinalVariableValues,
    NominalVariableValues,
    NumericalVariableValues,
)


def test_Survey_replace_values():
    num_var = 10
    data = np.random.randint(12, size=(100, num_var))
    data_df = pd.DataFrame(
        data=data.copy(), columns=[f"var_{i}" for i in range(1, num_var + 1)]
    )
    survey = Survey(data_df)
    survey.guess_variable_types(all_integer_variables_are_ordinal=True)
    survey.guess_variable_value_lists(
        invalid_value_range=~UnionOfIntervals(1, 10, True, True)
    )
    survey.replace_variable_values(mapping={11: -1})
    survey.replace_variable_value_labels(mapping={0: "unknown", -1: "missing"})
    survey.replace_variable_value_labels(
        mapping=dict(zip(range(1, 11), [f"value {i}" for i in range(1, 11)]))
    )
    info = survey.variable_info()
    for i, v in enumerate(info.index):
        var = info.loc[v, :]
        for value, label in var.valid_values.items():
            assert label == f"value {value}"
        assert var.invalid_values == {0: "unknown", -1: "missing"}

        col = survey.data[v]
        assert set(col.unique()) == set(range(-1, 11))
        for value in range(11):
            assert ((col == value) == (data[:, i] == value)).all()
        assert ((col == -1) == (data[:, i] == 11)).all()

    survey.replace_variable_values(mapping=dict(zip(range(2, 11, 2), range(1, 11, 2))))
    info = survey.variable_info()
    for i, v in enumerate(info.index):
        var = info.loc[v, :]
        col = survey.data[v]
        for v, l in var.valid_values.items():
            assert ((col == v) == ((data[:, i] == v) | (data[:, i] == v + 1))).all()
            assert l == f"value {v} | value {v+1}"

    survey.replace_variable_values(mapping=[1, 3, 5, 7, 9])
    info = survey.variable_info()
    for i, v in enumerate(info.index):
        var = info.loc[v, :]
        col = survey.data[v]
        for v, l in var.valid_values.items():
            assert (
                (col == v) == ((data[:, i] == v * 2 - 1) | (data[:, i] == v * 2))
            ).all()
            assert l == f"value {v*2-1} | value {v*2}"

    wanted_order = [5, 1, 3, 4, 2]
    survey.replace_variable_values(mapping=wanted_order)
    info = survey.variable_info()
    for i, v in enumerate(info.index):
        var = info.loc[v, :]
        valid_values = var.valid_values
        col = survey.data[v]
        for index, wanted_value in enumerate(wanted_order):
            assert (
                (col == index + 1)
                == (
                    (data[:, i] == wanted_value * 2 - 1)
                    | (data[:, i] == wanted_value * 2)
                )
            ).all()
            assert (
                valid_values[index + 1]
                == f"value {wanted_value*2-1} | value {wanted_value*2}"
            )


def test_scalevar_all_ops():
    var = SurveyVariable(
        "scale",
        var_type="scale",
        label="SCALE",
        valid_values=dict(zip(range(10), range(10))),
        invalid_values={-1: -1, -2: -2},
    )
    (v := var.create_values()).update_values_for_specification(1, ("scale", "<", 5), 1)
    assert v.intervals == UnionOfIntervals(-np.inf, 5, False, False)
    (v := var.create_values()).update_values_for_specification(1, ("scale", "<=", 5), 1)
    assert v.intervals == UnionOfIntervals(-np.inf, 5, False, True)
    (v := var.create_values()).update_values_for_specification(1, ("scale", ">", 5), 1)
    assert v.intervals == UnionOfIntervals(5, np.inf, False, False)
    (v := var.create_values()).update_values_for_specification(1, ("scale", ">=", 5), 1)
    assert v.intervals == UnionOfIntervals(5, np.inf, True, False)
    (v := var.create_values()).update_values_for_specification(1, ("scale", "==", 5), 1)
    assert v.intervals == UnionOfIntervals(5, 5, True, True)
    (v := var.create_values()).update_values_for_specification(1, ("scale", "!=", 5), 1)
    assert v.intervals == UnionOfIntervals.create_with_tuples(
        [(-np.inf, 5, False, False), (5, np.inf, False, False)]
    )
    (v := var.create_values()).update_values_for_specification(
        1, ("scale", "in", UnionOfIntervals(1, 4)), 1
    )
    assert v.intervals == UnionOfIntervals(1, 4)
    (v := var.create_values()).update_values_for_specification(
        1, ("scale", "not in", UnionOfIntervals(1, 4)), 1
    )
    assert v.intervals == UnionOfIntervals.create_with_tuples(
        [(-np.inf, 1, False, True), (4, np.inf, True, False)]
    )


def test_update_possible_values_nom_ord():
    num_nom_values = 5
    v_nom = SurveyVariable(
        "nom",
        "nominal",
        valid_values=dict(zip(range(num_nom_values), range(num_nom_values))),
        invalid_values=dict(zip([-3, -2], ["nan", "slip"])),
    )
    meta = [(v_nom.name, "==", i) for i in range(num_nom_values)]
    values = v_nom.create_values()
    assert values.possible_values == set(range(num_nom_values))
    values.update_values_for_specification(-1, meta[2], 1)
    assert values.possible_values == set(range(num_nom_values)) - {2}
    values.update_values_for_specification(1, meta[4], 1)
    assert values.possible_values == {4}

    num_ord_values = 10
    v_ord = SurveyVariable(
        "ord",
        "ordinal",
        valid_values=dict(zip(range(num_ord_values), range(num_ord_values))),
        invalid_values=dict(zip([-3, -2], ["nan", "slip"])),
    )
    meta = [(v_ord.name, ">", i) for i in range(num_ord_values - 1)]
    values = v_ord.create_values()
    assert values.possible_values == set(range(10))
    values.update_values_for_specification(1, meta[2], 1)
    assert values.possible_values == set(range(3, 10))
    values.update_values_for_specification(-1, meta[8], 1)
    assert values.possible_values == set(range(3, 9))
    values.update_values_for_specification(1, meta[6], -1)
    assert values.possible_values == set(range(3, 7))


def test_interval_complements_basic():
    assert ~UnionOfIntervals(-np.inf, 2) == UnionOfIntervals(2, np.inf, True, False)
    assert ~UnionOfIntervals(1, np.inf) == UnionOfIntervals(-np.inf, 1, False, True)
    assert UnionOfIntervals(-np.inf, 0) & UnionOfIntervals(
        -np.inf, 2
    ) == UnionOfIntervals(-np.inf, 0)
    assert UnionOfIntervals(0, np.inf) & UnionOfIntervals(
        -2, np.inf
    ) == UnionOfIntervals(0, np.inf)
    assert UnionOfIntervals(-np.inf, 0) & UnionOfIntervals(-2, -1) == UnionOfIntervals(
        -2, -1
    )

    i = ~UnionOfIntervals(1, 2)
    assert (i.min, i.min_incl, i.max, i.max_incl) == (-np.inf, False, 1, True)
    assert (i.next.min, i.next.min_incl, i.next.max, i.next.max_incl) == (
        2,
        True,
        np.inf,
        False,
    )
    assert i.next.next is None
    assert ~i == UnionOfIntervals(1, 2)

    j = UnionOfIntervals(1, 2)
    j.next = UnionOfIntervals(3, 4)
    i = ~j
    assert (i.min, i.min_incl, i.max, i.max_incl) == (-np.inf, False, 1, True)
    assert i.next is not None
    assert (i.next.min, i.next.min_incl, i.next.max, i.next.max_incl) == (
        2,
        True,
        3,
        True,
    )
    assert i.next.next is not None
    assert (
        i.next.next.min,
        i.next.next.min_incl,
        i.next.next.max,
        i.next.next.max_incl,
    ) == (4, True, np.inf, False)
    assert i.next.next.next is None

    j = UnionOfIntervals(-np.inf, 1)
    j.next = UnionOfIntervals(2, 3)
    i = ~j
    assert (i.min, i.min_incl, i.max, i.max_incl) == (1, True, 2, True)
    assert i.next is not None
    assert (i.next.min, i.next.min_incl, i.next.max, i.next.max_incl) == (
        3,
        True,
        np.inf,
        False,
    )
    assert i.next.next is None

    j = UnionOfIntervals(1, 2)
    j.next = UnionOfIntervals(3, np.inf)
    i = ~j
    assert (i.min, i.min_incl, i.max, i.max_incl) == (-np.inf, False, 1, True)
    assert i.next is not None
    assert (i.next.min, i.next.min_incl, i.next.max, i.next.max_incl) == (
        2,
        True,
        3,
        True,
    )
    assert i.next.next is None

    # does not check if the open-closed inversion works everywhere, but I think we should cope with this ;-)


def test_interval_intersection_basic():
    assert (
        UnionOfIntervals(-np.inf, -1) & UnionOfIntervals(1, np.inf)
        == UnionOfIntervals.nothing()
    )
    assert UnionOfIntervals(-np.inf, 1) & UnionOfIntervals(
        0, np.inf
    ) == UnionOfIntervals(0, 1, False, False)
    assert UnionOfIntervals(1, 3) & UnionOfIntervals(2, 4) == UnionOfIntervals(
        2, 3, False, False
    )
    assert UnionOfIntervals(-3, 3) & UnionOfIntervals(-1, 1) == UnionOfIntervals(
        -1, 1, False, False
    )

    i = UnionOfIntervals(-np.inf, 1)
    i.next = UnionOfIntervals(2, np.inf, prev=i)
    j = i & UnionOfIntervals(0, 3)
    assert (j.min, j.min_incl, j.max, j.max_incl) == (0, False, 1, False)
    assert j.next is not None
    assert (j.next.min, j.next.min_incl, j.next.max, j.next.max_incl) == (
        2,
        False,
        3,
        False,
    )


def test_interval_union_basic():
    assert (
        UnionOfIntervals(-np.inf, 1) | UnionOfIntervals(-1, np.inf)
        == UnionOfIntervals.all()
    )
    assert (
        UnionOfIntervals(-np.inf, 10, False, True)
        | UnionOfIntervals(10, np.inf, False, False)
        == UnionOfIntervals.all()
    )
    i = UnionOfIntervals(1, 2, False, True) | UnionOfIntervals(3, 4, False, False)
    assert (i.min, i.min_incl, i.max, i.max_incl) == (1, False, 2, True)
    assert i.next is not None
    assert (i.next.min, i.next.min_incl, i.next.max, i.next.max_incl) == (
        3,
        False,
        4,
        False,
    )


def test_interval_from_tuples():
    i = UnionOfIntervals.create_with_tuples([(2, 3), (4, 5), (6, 7)])
    assert [x[0] for x in i.points()] == [2, 3, 4, 5, 6, 7]
    assert all(x[1] == False for x in i.points())

    i = UnionOfIntervals.create_with_tuples(
        [(2, 3, False, True), (4, 5, False, True), (6, 7, False, True)]
    )
    assert [x[0] for x in i.points()] == [2, 3, 4, 5, 6, 7]
    assert all(x[1] == (i % 2 == 1) for i, x in enumerate(i.points()))


def test_de_morgan():
    i = UnionOfIntervals.create_with_tuples([(2, 3), (4, 5), (6, 7)])
    j = UnionOfIntervals.create_with_tuples(
        [(-np.inf, 3), (3.5, 4.5), (5, 8), (10, np.inf)]
    )

    assert ~(i & j) == ~i | ~j
    assert ~(i | j) == ~i & ~j


if __name__ == "__main__":
    test_Survey_replace_values()
