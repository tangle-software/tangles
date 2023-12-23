# type: ignore
import numpy as np
import scipy as sp

from tangles.separations.finding._min_s_t_cut import _augmented_graph_S_T
from tangles.separations.finding import min_S_T_cut
from tangles.util.graph import greedy_neighborhood

def test_augment_matrix_S_T():
  for i in range(100):
    A = np.zeros((100, 100))
    num_edges = np.random.randint(100,1000)
    for e in range(num_edges):
      e = np.random.choice(A.shape[0], size=2, replace=False)
      A[e[0],e[1]] = A[e[1],e[0]] = 1
    A_sparse = sp.sparse.csr_array(A)
    S = np.random.choice(A.shape[0], size=10, replace=False)
    T = np.random.choice(A.shape[0], size=10, replace=False)
    B_sparse = _augmented_graph_S_T(A_sparse, S, T)
    B = B_sparse.toarray()

    assert (A == B[:A.shape[0],:A.shape[1]]).all()
    for s in S:
      assert B[s,A.shape[0]] > 100
      assert B[A.shape[0],s] > 100
    for t in T:
      assert B[t, A.shape[0] + 1] > 100
      assert B[A.shape[0] + 1, t] > 100
