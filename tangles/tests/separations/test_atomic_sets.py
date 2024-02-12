# type: ignore
import pytest

from tangles.tests.path import get_test_data_path
from tangles._debug.tst_validation import is_tst_valid
from tangles.separations import SetSeparationSystem
from tangles import TangleSweep, agreement_func
from tangles.util.graph.minor import contract_graph
from tangles.separations.atomic_sets import atomic_to_seps, seps_to_atomic
from tangles.util.graph import laplacian, normalized_laplacian
from tangles.separations.finding import spectral_features
import networkx as nx
import numpy as np
from scipy import sparse

def _load_mona_graph():
  return sparse.load_npz(get_test_data_path("graphs/mona.npz"))


def _run_atomic_graph_barabasi_test():
  # G = nx.barabasi_albert_graph(500, 2, seed=86)
  # A = nx.adjacency_matrix(G)

  # L = nx.laplacian_matrix(G)
  # l, U = np.linalg.eigh(L)
  # S = -np.ones((U.shape[0], 10), dtype=np.int8)
  # S[U[:, 1:11] > 0] = 1

  # A_reduced, atoms = contract_graph(A, S)

  # # check neighbourhoods in reduced graph:
  # for u in range(A_reduced.shape[0]):
  #   V_u = atoms[u]
  #   V_neighbors = [(neighbor, atoms[neighbor]) for neighbor in np.nonzero(A_reduced[u, :])[1]]

  #   # all neighbors of V_u must be in the 'hull' of V_u here!
  #   neighborsInG = A[V_u, :].nonzero()[1]
  #   if not set(neighborsInG).issubset(set(V_u).union(*[v[1] for v in V_neighbors])):
  #     print("atomic_graph_barabasi_test: vertex in atom has neighbor outside of neighboring atoms -> FAILURE")
  #     return False

  #   # there must be an edge to V_u from every V_neighbor
  #   for V_v in V_neighbors:
  #     edgesV_uToV_v = A[V_u, :][:, V_v[1]]
  #     if edgesV_uToV_v.data.shape[0] == 0:
  #       print("atomic_graph_barabasi_test: nonadjacent atoms have an edge in reduced graph -> FAILURE")
  #       return False

  #     weight = edgesV_uToV_v.sum()
  #     if np.abs(weight - A_reduced[u, V_v[0]]) > 1e-10:
  #       print("atomic_graph_barabasi_test: weight does not match")
  #       return False

  # # check mapping sep <-> sep reduced
  # S_a = seps_to_atomic(S, atoms)
  # S_b = atomic_to_seps(S_a, atoms)

  # if np.any(S_b != S):
  #   print("atomic_graph_barabasi_test: atomSepsToSeps(atomSepsToSeps(...))  -> FAILURE")
  #   return False

  # return True
  pass


def _run_atomic_components_graph_test_mona():
  # A = _load_mona_graph()
  # L = normalized_laplacian(A)
  # S = spectral_bipartitions(L, 13)[:, 1:]

  # A_reduced, components = atomic_components(A, S, min_atom_size=100)

  # # check neighbourhoods in reduced graph:
  # for u in range(A_reduced.shape[0]):
  #   V_u = components[u]
  #   V_neighbors = [components[neighbor] for neighbor in np.nonzero(A_reduced[u, :])[1]]

  #   # all neighbors of V_u must be in the 'hull' of V_u here!
  #   neighbors_in_G = A[V_u, :].nonzero()[1]
  #   if not set(neighbors_in_G).issubset(set(V_u).union(*V_neighbors)):
  #     print("__testAtomicGraphForImage: vertex in atom has neighbor outside of neighboring atoms -> FAILURE")
  #     return False

  #   # there must be an edge to V_u from every V_neighbor
  #   for V_v in V_neighbors:
  #     if A[V_u, :][:, V_v].data.shape[0] == 0:
  #       print("__testAtomicGraphForImage: nonadjacent atoms have an edge in reduced graph -> FAILURE")
  #       return False

  # S_d = graph_distance_partitions(A_reduced, 2)
  # S_b = atomic_to_seps(S_d, components)  # this direction should work
  # S_a = seps_to_atomic(S_b, components)

  # if np.any(S_a != S_d):
  #   print("__testAtomicGraphForImage: atomSepsToSeps(atomSepsToSeps(...))  -> FAILURE")
  #   return False

  # print("__testAtomicGraphForImage: all ok :-)")

  # return True
  pass

# def graph_distance_partitions(A, max_dist:int = 2) -> np.ndarray:
#     """
#     Separations generated by the graph distance partitions. For every node, and every distance less than
#     or equal the max_dist, there is one separation which has on one side the nodes of at most that distance
#     and on the other side the other nodes.

#     Parameters
#     ----------
#     max_dist (int): The maximum distance up to which the distance partitions are to be calculated.

#     Returns
#     -------
#     np.ndarray: The separations.
#     """
#     num_vertices = A.shape[0]
#     num_sep_per_node = (max_dist + 1)
#     seps = -np.ones((num_vertices, num_vertices * num_sep_per_node), dtype=np.int8)
#     distance_d_graph = np.identity(num_vertices)
#     for d in range(num_sep_per_node):
#         for u in range(num_vertices):
#             neighbors = np.nonzero(distance_d_graph[[u], :])[1]
#             seps[neighbors, u*num_sep_per_node+d:(u+1)*num_sep_per_node] = 1
#         distance_d_graph = distance_d_graph @ A

#     _, idx = np.unique(seps, axis=1, return_index=True)
#     return seps[:,np.sort(idx)]

# def _run_atomic_tangles_test():
#   A = _load_mona_graph()
#   L = normalized_laplacian(A)
#   S = spectral_bipartitions(L, 16)[:, 1:]

#   # create graph on atomic sets (don't split and don't combine small atoms, as we want to compare to the original
#   # tangles)
#   A_reduced, atoms = atomic_components(A, S, min_atom_size=0)
#   S_a = seps_to_atomic(S, atoms)

#   if np.any(atomic_to_seps(S_a, atoms) != S):
#     print("_run_atomic_tangles_test: conversion seps <-> atomSeps  -> FAILURE")
#     return False

#   atom_sizes = np.array([len(a) for a in atoms], dtype=int)
#   min_sup_size = 400

#   def atomic_agreement_min_sup(idcs, orientations):
#     return atom_sizes @ (S_a[:, idcs] * orientations > 0).all(axis=1)

#   ts_atomic = TangleSweep(atomic_agreement_min_sup, le_func=lambda x_1, x_2, x_3, x_4: False)
#   for i in range(S_a.shape[1]):
#     ts_atomic.append_separation(i, min_sup_size)

#   sep_sys, sep_info = SetSeparationSystem.with_array(S, return_sep_info=True)
#   ts_original = TangleSweep(agreement_func(sep_sys), le_func=sep_sys.is_le)
#   for i in sep_info[0]:
#     ts_original.append_separation(i, min_sup_size)

#   if not is_tst_valid(ts_atomic.tree, sep_sys.is_le, atomic_agreement_min_sup, min_sup_size):
#     print("_run_atomic_tangles_test: np.any(atomic_tangles != original_tangles)  -> FAILURE")
#     return False

#   return True

# def _run_atomic_orders_test():
#   A = _load_mona_graph()
#   L = laplacian(A)
#   S = spectral_bipartitions(L, 16)[:, 1:]

#   A_reduced, atoms = contract_graph(A, S)
#   L_reduced = -A_reduced.copy()
#   L_reduced.setdiag(A_reduced.sum(axis=0).flat)
#   S_reduced = seps_to_atomic(S, atoms)

#   orders_org = np.sum(np.multiply(L @ S, S), axis=0)
#   orders_red = np.sum(np.multiply(L_reduced @ S_reduced, S_reduced), axis=0)

#   if np.any(np.abs(orders_org - orders_red) > 1e-10):
#     print("_run_atomic_order_test: orders (unnormalized laplacian) do not match")
#     return False

#   return True


# ##########################
# @pytest.mark.short
# def test_atomic_graph_barabasi():
#   assert _run_atomic_graph_barabasi_test()

# def test_atomic_components_graph_mona():
#   assert _run_atomic_components_graph_test_mona()


# def test_atomic_tangles():
#   assert _run_atomic_tangles_test()


# @pytest.mark.short
# def test_atomic_orders():
#   assert _run_atomic_orders_test()