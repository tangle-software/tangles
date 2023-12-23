import numpy as np

from tangles.util.tree import TreeNode


class simple_tree_node(TreeNode):
  def __init__(self, id):
    self._id = id
    self._neighbours = []

  @property
  def neighbours(self):
    return self._neighbours

def connect_nodes(n1, n2):
  n1._neighbours.append(n2)
  n2._neighbours.append(n1)


def create_random_tree(num_nodes=100):
  nodes = [simple_tree_node(0), simple_tree_node(1)]
  connect_nodes(nodes[0], nodes[1])
  for i in range(2, num_nodes):
    new_node = simple_tree_node(i)
    p = np.array([n.degree() for n in nodes], dtype=float)
    p = np.sqrt(p)
    p /= p.sum()
    rand_node_idx = np.random.choice(len(nodes), p=p)
    node = nodes[rand_node_idx]
    connect_nodes(node, new_node)
    nodes.append(new_node)

  return nodes


def create_simple_tree():
  nodes = [simple_tree_node(i) for i in range(11)]

  def connect_nodes_simple(n1, n2):
    nodes[n1]._neighbours.append(nodes[n2])
    nodes[n2]._neighbours.append(nodes[n1])

  connect_nodes_simple(0, 1)
  connect_nodes_simple(1, 3)
  connect_nodes_simple(2, 3)
  connect_nodes_simple(3, 4)
  connect_nodes_simple(4, 5)
  connect_nodes_simple(4, 6)
  connect_nodes_simple(4, 7)
  connect_nodes_simple(4, 8)
  connect_nodes_simple(8, 9)
  connect_nodes_simple(8, 10)

  return nodes

def test_dfs_bfs():
  nodes = create_simple_tree()
  expected_dfs = [3,4,8,10,9,7,6,5,2,1,0]
  found_dfs = [n._id for n in nodes[3].node_iter(depth_first=True)]
  assert expected_dfs == found_dfs

  expected_bfs = [3, 1, 2, 4, 0, 5, 6, 7, 8, 9, 10]
  found_bfs = [n._id for n in nodes[3].node_iter(depth_first=False)]
  assert expected_bfs == found_bfs


def test_distance_classes_simple():
  nodes = create_simple_tree()
  dc = list(nodes[3].distance_classes())
  expected = [{nodes[3]},{nodes[1],nodes[2],nodes[4]},{nodes[0], nodes[5], nodes[6], nodes[7], nodes[8]}, {nodes[9], nodes[10]}]
  assert dc == expected

  dc = list(nodes[7].distance_classes())
  expected = [{nodes[7]}, {nodes[4]}, {nodes[6], nodes[5], nodes[3], nodes[8]}, {nodes[9], nodes[10], nodes[2], nodes[1]}, {nodes[0]}]
  assert dc == expected


def test_distance_classes_random():
  num_nodes = 100
  nodes = create_random_tree(num_nodes)
  for n in nodes:
    distance_classes = list(n.distance_classes())
    num = sum(len(dc) for dc in distance_classes)
    assert num == num_nodes
    for i,dc in enumerate(distance_classes):
      for dc2 in distance_classes[i+1:]:
        assert len(dc & dc2) == 0

