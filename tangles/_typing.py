from typing import Callable
import numpy as np

AgreementFunc = Callable[[np.ndarray, np.ndarray], int]
LessOrEqFunc = Callable[[int, int, int, int], bool]
SetSeparationOrderFunction = Callable[[np.ndarray], np.ndarray]

SepId = int
SepOrientation = int # Union[1, -1] does not work on older versions. maybe this should be an enum?
Separations = np.ndarray
OrientedSep = tuple[SepId, SepOrientation]
Feature = OrientedSep
Features = Separations
