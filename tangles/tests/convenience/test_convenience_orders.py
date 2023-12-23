import pytest
import numpy as np
import pandas as pd
from tangles.convenience import create_order_function


def test_order_function_01():
    features = 2*(np.random.random((20,10))>0.5).astype(float)-1
    order_func  = create_order_function('O1',features)
    orders = order_func(features)

    for f_idx in range(features.shape[1]):
        A = features[:,f_idx]>0
        B = features[:,f_idx]<0
        o = 0
        for f in range(features.shape[1]):
            o += (A & (features[:,f]>0)).sum() * (B & (features[:,f]>0)).sum() + (A & (features[:,f]<0)).sum() * (B & (features[:,f]<0)).sum()
        assert o == orders[f_idx]


def test_order_function_01biased():
    features = 2*(np.random.random((20,10))>0.5).astype(float)-1
    order_func_biased  = create_order_function('O1-biased', features)
    orders_biased = order_func_biased(features)

    for f_idx in range(features.shape[1]):
        A = features[:,f_idx]>0
        B = features[:,f_idx]<0
        o_biased = 0
        for f in range(features.shape[1]):
            o_biased += (A & (features[:,f]>0)).sum() * (B & (features[:,f]>0)).sum()
        assert o_biased == orders_biased[f_idx]


def test_order_function_02():
    features = 2*(np.random.random((20,10))>0.5).astype(float)-1
    order_func = create_order_function('O2',features)
    orders = order_func(features)

    for f_idx in range(features.shape[1]):
        A = features[:,f_idx]>0
        B = features[:,f_idx]<0
        o = 0
        for f in range(features.shape[1]):
            o += min((A & (features[:,f]>0)).sum(), (B & (features[:,f]>0)).sum())
            o += min((A & (features[:,f]<0)).sum(), (B & (features[:,f]<0)).sum())
        assert o == orders[f_idx]

def test_order_function_03():
    features = 2*(np.random.random((20,10))>0.5).astype(float)-1
    order_func = create_order_function('O3',features)
    orders = order_func(features)

    for f_idx in range(features.shape[1]):
        A = features[:,f_idx]>0
        B = features[:,f_idx]<0
        o = 0
        for f in range(features.shape[1]):
            o += min((A & (features[:,f]>0)).sum(), (B & (features[:,f]>0)).sum())
        assert o == orders[f_idx]


