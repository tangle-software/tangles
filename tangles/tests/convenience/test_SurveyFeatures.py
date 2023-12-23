import pytest
import numpy as np
import pandas as pd
from tangles.convenience import Survey, SurveyVariable, SurveyVariableValues, SimpleSurveyFeatureFactory, UnionOfIntervals
from tangles.convenience.SurveyFeatureFactory import *



def test_survey_separations_simple():
    data = np.empty((1000,4))
    min_nom, max_nom = 1,7
    data[:,0] = np.random.randint(min_nom,max_nom+1,size=data.shape[0])
    min_ord, max_ord = 0, 10
    data[:,1] = np.random.randint(min_ord, max_ord, size=data.shape[0])
    min_scale, max_scale = -12.1, 42.9
    data[:,2] = min_scale + (max_scale-min_scale)*np.random.random(data.shape[0])
    bin_values = (5,60)
    data[:,3] = np.random.choice(bin_values,size=data.shape[0], replace=True)
    pd_data = pd.DataFrame(data=data, columns=['nom', 'ord', 'scale', 'bin'])

    survey = Survey(pd_data)
    survey.set_variable_types(['nominal','ordinal','scale','binary'])

    factory = SimpleSurveyFeatureFactory(survey)
    seps, meta = factory.create_features()

    num_nom_seps = pd_data['nom'].nunique()
    num_ord_seps = pd_data['ord'].nunique()-1
    num_scale_seps = SurveyFeatureFactory.default_numeric_var_num_bins - 1
    num_bin_seps = 1
    assert seps.shape == (data.shape[0], sum((num_nom_seps, num_ord_seps, num_scale_seps, num_bin_seps)))
    assert meta.shape == (seps.shape[1],)

    assert all(m[0] == 'nom' for m in meta[:num_nom_seps])
    assert all(m[1] == '==' for m in meta[:num_nom_seps])
    assert set([m[2] for m in meta[:num_nom_seps]]) == set(range(min_nom, max_nom+1))

    assert all(m[0] == 'ord' for m in meta[num_nom_seps:num_nom_seps+num_ord_seps])
    assert all(m[1] == '>=' for m in meta[num_nom_seps:num_nom_seps+num_ord_seps])
    assert set([m[2] for m in meta[num_nom_seps:num_nom_seps+num_ord_seps]]) == set(pd_data['ord'].unique()) - {pd_data['ord'].min()}

    assert all(m[0] == 'scale' for m in meta[num_nom_seps+num_ord_seps:num_nom_seps+num_ord_seps+num_scale_seps])
    assert all(m[1] == '>=' for m in meta[num_nom_seps+num_ord_seps:num_nom_seps+num_ord_seps+num_scale_seps])
    min_v, max_v = pd_data['scale'].min(), pd_data['scale'].max()
    assert np.all(np.abs(np.array([m[2] for m in meta[num_nom_seps+num_ord_seps:num_nom_seps+num_ord_seps+num_scale_seps]])
                         -(min_v + np.arange(1,1+num_scale_seps)*(max_v - min_v)/(num_scale_seps+1))) < 1e-6)

    assert meta[-1][0] == 'bin'
    assert meta[-1][1] == '=='
    assert meta[-1][2] in bin_values

    assert (seps[:,:num_nom_seps].sum(axis=1) == -(num_nom_seps-2)).all()

    assert (np.maximum(-seps[:,num_nom_seps], np.maximum.reduce(seps[:,num_nom_seps:num_nom_seps+num_ord_seps], axis=1)) == 1).all()
    assert all((seps[:,i] >= seps[:,i+1]).all() for i in range(num_nom_seps, num_nom_seps+num_ord_seps-1))

    assert (np.maximum(-seps[:, num_nom_seps+num_ord_seps], np.maximum.reduce(seps[:, num_nom_seps+num_ord_seps:num_nom_seps+num_ord_seps+num_scale_seps], axis=1)) == 1).all()
    assert all((seps[:, i] >= seps[:, i + 1]).all() for i in range(num_nom_seps+num_ord_seps, num_nom_seps+num_ord_seps+num_scale_seps - 1))



@pytest.mark.parametrize("feature_func", [
numericvar_features_split_regular_ge,
numericvar_features_inside_regular_bins,
extensive_numericvar_features,
ordinalvar_features_ge_all_splits,
nominalvar_features_all_cats
])
def test_feature_meta_data(feature_func):
    # make sure the sep is doing what the metadata says...
    for invalid in [[], [-100, -200]]:
        for num_valid_values in [2,5,10,100]:
            data = np.random.randint(1,num_valid_values+1,size=300)
            col = pd.Series(data=np.append(data, invalid))
            seps, meta = feature_func(single_col_data=col, invalid_values=invalid)
            assert seps.shape[1] == meta.shape[0]
            for i,m in enumerate(meta):
                assert all(2 * eval(f"p {m[0]} m[1]") - 1 == seps[j, i] for j, p in enumerate(col) if p not in invalid)
                assert all(2 * eval(f"p {SurveyVariableValues.invert_op(m[0])} m[1]") - 1 == -seps[j, i] for j, p in enumerate(col) if p not in invalid)


@pytest.mark.parametrize("op", ['==','!=','<','>','<=','>='])
def test_binary_unique_value_features(op):
    invalid = [-100,-200]
    data = 13 + (np.random.random(100)>0.5)*8
    col = pd.Series(data=np.append(data, invalid))
    seps, meta = binary_unique_value_features(col, np.unique(data), op=op)
    assert seps.shape == (len(col), 1)
    assert meta.shape == (1,)
    assert meta[0][0] == op
    assert all(2*eval(f"p {op} {meta[0][1]}")-1 == seps[j,0] for j, p in enumerate(col))


def test_simple_unique_value_features():
    invalid = [-100, -200]
    data = np.random.randint(1, 5, size=300)
    col = pd.Series(data=np.append(data, invalid))
    unique_v = np.unique(data)
    seps, meta = simple_unique_value_features(single_col_data=col, unique_values=unique_v)
    assert seps.shape[1] == meta.shape[0]
    assert seps.shape[1] == len(unique_v)
    for i, m in enumerate(meta):
        assert all(2 * eval(f"p {m[0]} m[1]") - 1 == seps[j, i] for j, p in enumerate(col))


def test_numericvar_features_split_regular_ge():
    invalid = [-100, -200]
    data = -10 + np.random.random(100) * 20
    col = pd.Series(data=np.append(data, invalid))
    min, max = data.min(), data.max()
    for num_bins in range(2, 7):
        seps, meta = numericvar_features_split_regular_ge(col, num_bins=num_bins, invalid_values=invalid, max_num_values_for_extensive_seps=None)
        assert seps.shape == (len(col), num_bins-1)
        assert meta.shape == (num_bins-1,)
        d = (max - min) / num_bins
        for i in range(seps.shape[1]):
            assert meta[i][0] == '>='
            assert np.abs(meta[i][1] - (min + (i+1) * d)) < 1e-8
            assert all(2 * (p>=meta[i][1]) - 1 == seps[j, i] for j, p in enumerate(col))


def test_numericvar_features_inside_regular_bins():
    invalid = [-100,-200]
    data = -10 + np.random.random(100) * 20
    col = pd.Series(data=np.append(data, invalid))
    min, max = data.min(), data.max()
    for num_bins in range(2,7):
        seps, meta = numericvar_features_inside_regular_bins(col, num_bins=num_bins, invalid_values = invalid)
        assert seps.shape == (len(col), num_bins)
        assert meta.shape == (num_bins,)
        d = (max-min)/num_bins
        for i in range(num_bins):
            lo, hi = min+i*d, min+(i+1)*d,
            assert meta[i][0] == 'in'
            assert meta[i][1] == UnionOfIntervals(lo, hi, True, i == num_bins - 1)
            assert all(2* (p in meta[i][1])-1 == seps[j,i] for j, p in enumerate(col))




def test_ordinalvar_features_ge_all_splits():
    invalid = [-100, -200]
    data = np.random.randint(1, 10, size=300)
    unique_v = np.unique(data)
    col = pd.Series(data=np.append(data, invalid))
    seps, meta = ordinalvar_features_ge_all_splits(single_col_data=col, invalid_values=invalid)
    assert seps.shape[1] == meta.shape[0]
    assert seps.shape[1] == len(unique_v)-1
    unique_v.sort()
    for i, m in enumerate(meta):
        assert all(2 * (p>=unique_v[i+1]) - 1 == seps[j, i] for j, p in enumerate(col))


def test_nominalvar_features_all_cats():
    invalid = [-100, -200]
    data = np.random.randint(1, 10, size=300)
    unique_v = np.unique(data)
    col = pd.Series(data=np.append(data, invalid))
    seps, meta = nominalvar_features_all_cats(single_col_data=col, invalid_values=invalid)
    assert seps.shape[1] == meta.shape[0]
    assert seps.shape[1] == len(unique_v)
    for i, m in enumerate(meta):
        assert all(2 * (p==unique_v[i]) - 1 == seps[j, i] for j, p in enumerate(col))


