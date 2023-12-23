import pytest
from tangles.tests.convenience.ConvenienceTestCase import ConvenienceTestCase
from tangles.convenience.convenience_functions import _create_sep_sys
from tangles.separations import FeatureSystem, SetSeparationSystem

import numpy as np

@pytest.fixture
def hamburg() -> ConvenienceTestCase:
    return ConvenienceTestCase.load_test_case_covariance('hamburg_osm_spectral')

@pytest.fixture
def mona() -> ConvenienceTestCase:
    return ConvenienceTestCase.load_test_case_covariance('mona_lisa_spectral', shift=10)

@pytest.fixture
def moby() -> ConvenienceTestCase:
    return ConvenienceTestCase.load_test_case('moby_dick')

def test_hamburg_tst(hamburg: ConvenienceTestCase):
    hamburg.test_tst(100)

def test_hamburg_tot(hamburg: ConvenienceTestCase):
    hamburg.test_tot(100)

def test_mona_tst(mona: ConvenienceTestCase):
    mona.test_tst(100)

def test_mona_tot(mona: ConvenienceTestCase):
    mona.test_tot(900)

@pytest.mark.long
@pytest.mark.skip(reason="Skipping long tests by default.")
def test_moby_tst(moby: ConvenienceTestCase):
    moby.test_tst(1)

@pytest.mark.long
@pytest.mark.skip(reason="Skipping long tests by default.")
def test_moby_tot(moby: ConvenienceTestCase):
    moby.test_tot(1)


def test_convenience_sep_sys_creation():
    np.random.seed(101010)

    bips = np.ones((100, 12))
    bips[np.random.random(bips.shape) > 0.5] = -1
    assert isinstance(_create_sep_sys(bips), FeatureSystem)

    bips[bips > 0] = np.random.random((bips > 0).sum())
    bips[bips < 0] = -np.random.random((bips < 0).sum())
    assert isinstance(_create_sep_sys(bips), FeatureSystem)

    seps = np.zeros((100, 12))
    seps[np.random.random(seps.shape) < 0.7] = 1
    seps[np.random.random(seps.shape) < 0.7] = -1
    assert isinstance(_create_sep_sys(seps), SetSeparationSystem)




