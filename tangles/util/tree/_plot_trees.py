# type: ignore

import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import collections as mc
from typing import Tuple, Callable, Union, Any
from ._tree import TreeNode

PlotNodeFunc = Callable[[TreeNode, plt.Axes], None]
EdgeColorFunc = Callable[[TreeNode, TreeNode], Any]
PlotEdgeAnnotationFunc = Callable[[TreeNode, TreeNode, plt.Axes], None]

def _count_nodes_and_leaves_rek(node, prev_node):
  if node.is_leaf() and prev_node is not None:
    node.num_leaves = 1
    return 1
  else:
    num_leaves = 0
    num_nodes = 1
    for n in filter(lambda n:n != prev_node, node.neighbours):
      num_nodes += _count_nodes_and_leaves_rek(n, node)
      num_leaves += n.num_leaves
    node.num_leaves = num_leaves
    return num_nodes

def _set_coord_rek(node, prev_node, angle_start, angle_end, radius):
  angle = 0.5*(angle_start+angle_end)
  node.coord = (radius*math.cos(angle), radius*math.sin(angle))

  d_angle = (angle_end-angle_start)/node.num_leaves
  angle = angle_start
  for n in filter(lambda n:n != prev_node, node.neighbours):
    next_angle = angle+n.num_leaves*d_angle
    _set_coord_rek(n, node, angle, next_angle, radius+1/(1+radius))
    angle = next_angle

def compute_tree_node_positions(central_tree_node: TreeNode) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  # returns nodes, their positions as pairs and edges as pairs of pairs
  # all coordinates are in [-1,1]x[-1,1]
  total_nodes = _count_nodes_and_leaves_rek(central_tree_node, None)
  _set_coord_rek(central_tree_node, None, 0, 2*math.pi,0)
  node_coords, edge_coords = np.empty((total_nodes,2)), np.empty((total_nodes-1,2,2))
  nodes = np.empty(total_nodes, dtype=object)
  for i,n in enumerate(central_tree_node.node_iter()):
    node_coords[i,:] = n.coord
    nodes[i] = n

  edges = []
  for i,(n1,n2) in enumerate(central_tree_node.edge_iter()):
    edge_coords[i,0,:], edge_coords[i,1,:] = n1.coord, n2.coord
    edges.append((n1,n2))
  edges = np.array(edges)

  for n in central_tree_node.node_iter(): # cleanup...
    del n.coord, n.num_leaves

  center = node_coords.mean(axis=0)
  node_coords -= center
  max_norm = np.sqrt(np.square(node_coords).sum(axis=1)).max()
  node_coords /= max_norm

  edge_coords -= center
  edge_coords /= max_norm
  return nodes, node_coords, edges, edge_coords



def print_node_label(node, ax):
  """
  A default callback function that plots the node's label.

  Parameters
  ----------
  node : TreeNode
    The node to plot.

  ax : matplotlib.axes.Axes
    The matplotlib axes object where the node is plotted 
    (the axes is already moved to the position of the node and scaled according to the parameter node_size of plot_tree).
  """

  ax.axis('off')
  ax.text(0.5,0.5,node.label, horizontalalignment='center', verticalalignment='center', backgroundcolor="w")



def _ev_central_node(some_tree_node: TreeNode):
    all_tree_nodes = some_tree_node.all_nodes()
    node_idx_map = dict(zip(all_tree_nodes, range(len(all_tree_nodes))))
    A = np.zeros((len(all_tree_nodes), len(all_tree_nodes)), dtype=np.int8)
    for n_i,n in enumerate(all_tree_nodes):
      for m in n.neighbours:
        m_i = node_idx_map[m]
        A[n_i,m_i] = A[m_i, n_i] = 1
    lbda, U = np.linalg.eigh(A)
    return all_tree_nodes[U[:,-1].argmax()]


def _find_central_tree_node_bfs(some_tree_node: TreeNode):
  far_node_1 = list(some_tree_node.node_iter(depth_first=False))[-1]
  far_node_2 = list(far_node_1.node_iter(depth_first=False))[-1]
  path = far_node_1.shortest_path(far_node_2)
  if path and (l:=len(path))>2:
    n = path[l//2]
    if l%2==0 and (m := path[l//2-1]).degree()>n.degree():
      return m
    else:
      return n
  else:
    return some_tree_node




def plot_tree(some_tree_node: TreeNode, ax: Union[plt.Axes, None] = None,
              search_center: bool = True, plot_node: PlotNodeFunc = None,
              edge_color: EdgeColorFunc = None,
              node_size: Tuple[float,float] | dict[TreeNode, tuple] = (0.01,0.01),
              ax_projection: str = None, 
              plot_edge_annotation: PlotEdgeAnnotationFunc = None, 
              edge_annotation_size: Tuple[float,float] = (0.01,0.01)):
  """
  Plot a tree.

  Parameters
  ----------
  some_tree_node : TreeNode
    A tree node of the tree to plot. Any node of the tree can be used.
  ax : matplotlib.axes.Axes or None
    The axes object where the tree is plotted. If None, the currently active axis of matplotlib is used.
  search_center : bool
    Whether to search a better center node (currently a node with max degree).
  plot_node : PlotNodeFunc
    A callback function that actually plots the tree. If None, a small circle is plotted at each node position. 
    See :func:`print_node_label` for an example of such a callback function. 
    It must take the parameters `node` of type TreeNode and `ax` of type Matplotlib.axes.ax.
  node_size : 2-tuple of float or dict[TreeNode, (float,float)]
    The width and height of a node as a fraction of the total size of the figure 
    or a dict that returns the node size for each tree node.
  ax_projection : str
    The matplotlib projection type of the axes of the node plots. 
    Defaults to 'rectilinear' which fits most plots, so it only needs to be set for uncommon plot types.
  plot_edge_annotation : PlotEdgeAnnotationFunc
    A callback function that plots an edge annotation at each edge. If None, no annotation is plotted.
  edge_annotation_size : 2-tuple of float
    The width and height of an edge annotation plot as a fraction of the total size of the figure.
  """

  if ax is None:
    ax = plt.gca()

  if isinstance(node_size, tuple):
    node_size = {node: node_size for node in some_tree_node.all_nodes()}
  max_node_size = max(node_size.values(), key=sum)  

  xlim, ylim = ax.get_xlim(), ax.get_ylim()
  scale = 0.5 * (xlim[1] - xlim[0] - max_node_size[0]), 0.5 * (ylim[1] - ylim[0] - max_node_size[1])
  center = (0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1]))

  #central_node = some_tree_node.search_max_degree_node() if search_center else some_tree_node
  central_node = _find_central_tree_node_bfs(some_tree_node) if search_center else some_tree_node
  if central_node.degree()==0:
    nodes, node_coords, edges, edge_coords = np.array([central_node]), np.array([[0,0]]), np.empty(0), np.empty(0)
    node_size_list = [max_node_size]
  else:
    nodes, node_coords, edges, edge_coords = compute_tree_node_positions(central_node)
    node_size_list = [node_size[node] for node in nodes]
    edge_coords = edge_coords * scale + center
  node_coords = node_coords*scale + center

  if edge_color is None:
    edge_colors = 'tab:blue'
  else:
    edge_colors = []
    for (n1,n2) in edges:
      edge_colors.append(edge_color(n1,n2))

  edge_lines = mc.LineCollection(edge_coords, colors=edge_colors, linewidths=1)
  ax.add_collection(edge_lines)

  trans = ax.transData.transform  # from data to figure
  trans2 = ax.get_figure().transFigure.inverted().transform  # from figure to axes
  child_axes = []

  if plot_edge_annotation:
    for i, (n1,n2) in enumerate(edges):
      start, end = edge_coords[i,0,:], edge_coords[i,1,:]
      x_pos, y_pos = 0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1])
      xa, ya = trans2(trans((x_pos, y_pos)))
      edge_annotation_ax = plt.axes([xa-edge_annotation_size[0]*0.5, ya-edge_annotation_size[1]*0.5, edge_annotation_size[0], edge_annotation_size[1]])
      plot_edge_annotation(n1, n2, edge_annotation_ax)
      child_axes.append(edge_annotation_ax)

  if plot_node is None:
    node_sizes, angles = np.ones(node_coords.shape[0])*max_node_size[0], np.zeros(node_coords.shape[0])
    node_circles = mc.EllipseCollection(node_sizes, node_sizes, angles, offsets=node_coords, units='x', transOffset=ax.transData)
    ax.add_collection(node_circles)
  else:
    for i in range(nodes.shape[0]):
      xa, ya = trans2(trans(node_coords[i]))
      size = node_size_list[i] 
      node_ax = plt.axes([xa-size[0]*0.5, ya-size[1]*0.5, size[0], size[1]], projection=ax_projection)
      plot_node(nodes[i], node_ax)
      child_axes.append(node_ax)

  ax.axis('off')
  return child_axes
