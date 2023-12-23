import pytest
from tangles.tests.search.SearchTestCase import SimpleTSTestCase

_mona_agreements = [200, 500, 1000, 3000]
_hamburg_agreements = [50, 100, 200, 300]

@pytest.fixture
def mona() -> SimpleTSTestCase:
  return SimpleTSTestCase.load("mona_lisa_spectral")

@pytest.fixture
def hamburg() -> SimpleTSTestCase:
  return SimpleTSTestCase.load("hamburg_osm_spectral")

@pytest.mark.parametrize("agreement", _hamburg_agreements)
def test_hamburg_append(hamburg: SimpleTSTestCase, agreement: int) -> None:
  passes = hamburg.run_append(agreement)
  assert passes

@pytest.mark.parametrize("agreement", _hamburg_agreements)
def test_hamburg_insert_half(hamburg: SimpleTSTestCase, agreement: int) -> None:
  passes = hamburg.run_insert_half(agreement)
  assert passes

@pytest.mark.parametrize("agreement", _hamburg_agreements)
def test_hamburg_insert_interleaved(hamburg: SimpleTSTestCase, agreement: int) -> None:
  passes = hamburg.run_insert_interleaved(agreement)
  assert passes

@pytest.mark.parametrize("agreement", _hamburg_agreements)
def test_hamburg_agreement_interaction(hamburg: SimpleTSTestCase, agreement: int) -> None:
  passes = hamburg.run_agreement_interaction(agreement)
  assert passes

def test_sweep_hamburg(hamburg: SimpleTSTestCase):
  hamburg.run_sweep()

@pytest.mark.parametrize("agreement", _mona_agreements)
def test_mona_append(mona: SimpleTSTestCase, agreement: int) -> None:
  passes = mona.run_append(agreement)
  assert passes

@pytest.mark.parametrize("agreement", _mona_agreements)
def test_mona_insert_half(mona: SimpleTSTestCase, agreement: int) -> None:
  passes = mona.run_insert_half(agreement)
  assert passes

@pytest.mark.parametrize("agreement", _mona_agreements)
def test_mona_insert_interleaved(mona: SimpleTSTestCase, agreement: int) -> None:
  passes = mona.run_insert_interleaved(agreement)
  assert passes

@pytest.mark.parametrize("agreement", _mona_agreements)
def test_mona_agreement_interaction(mona: SimpleTSTestCase, agreement: int) -> None:
  passes = mona.run_agreement_interaction(agreement)
  assert passes

def test_sweep_mona(mona: SimpleTSTestCase) -> None:
  mona.run_sweep()