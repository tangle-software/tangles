from typing import List,Union
import numpy as np
from tangles.util.tree import TreeNode

class ToTNode(TreeNode):
    """
    A node in a tree of tangles.
    """

    def __init__(self, tangle_idx: int, reduced_tangle: np.ndarray):
        self.edges: List[ToTEdge] = []

        self.tangle_idx = tangle_idx
        self.reduced_tangle = reduced_tangle

        self.label = str(tangle_idx)

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf.

        Returns
        ----------
        bool
            Whether this node is a leaf.
        """

        return len(self.edges) == 1

    def degree(self) -> int:
        """
        Return the degree of this node.

        Returns
        ----------
        int
            The degree of this node.
        """

        return len(self.edges)

    def set_label(self, label):
        """
        Set the label of this node.

        Parameters
        ----------
        label : object
            A label for this node.
            It can be of any type, but usually it is a string describing the node e.g. in a visualisation.
        """

        self.label = label

    @property
    def neighbours(self) -> List['ToTNode']:
        """
        All neighbors of this node.
        """

        return [e.node1 if e.node2 == self else e.node2 for e in self.edges]

    def all_tot_edges(self):
        edge_set = set()
        for node in self.node_iter(depth_first=False):
            edge_set.update(node.edges)
        return list(edge_set)


    @property
    def star(self):
        sep_ids = np.array([e.sep_id for e in self.edges], dtype=int)
        orientations = self.reduced_tangle[[e.sep_idx for e in self.edges]]
        return sep_ids, orientations

class ToTEdge:
    """
    An edge in a tree of tangles.
    """

    def __init__(self, sep_id: int, sep_idx: int):
        self.sep_id = sep_id
        self.sep_idx = sep_idx
        self.node1 = None
        self.node2 = None

    def other_end(self, node: ToTNode):
        """
        Return the node contained in this edge that is not equal to the given `node`.

        Parameters
        ----------
        node : ToTNode
            One of the nodes of this edge. The other node of this edge is returned.

        Returns
        ----------
        ToTNode
            The node incident to this edge that is not equal to the given `node`.
        """

        return self.node2 if self.node1 == node else self.node1


class TreeOfTangles:
    """
    A tree which nodes are precisely the maximal tangles in the tangle search tree.

    Each edge between two nodes corresponds to the efficient distinguisher of the incident nodes.
    """

    def __init__(self, sep_ids: np.ndarray, nodes: List[ToTNode], edges: Union[List[ToTEdge],None] = None):
        self.sep_ids = sep_ids
        self.nodes = nodes
        self.tangle_matrix = None
        if edges is None:
            self.edges = list(set().union(*[n.edges for n in nodes]))
        else:
            self.edges = edges

    def any_node(self):
        """
        Return a node of the tree.

        Returns
        ----------
        ToTNode
            A node of the tree.
        """

        return self.nodes[0] if len(self.nodes)>0 else None

    def __eq__(self, other):    # does not check the tangle_idx, this means, trees coming from permuted tangle_matrices should still be equal
        if len(self.sep_ids) != len(other.sep_ids):
            return False
        if np.any(self.sep_ids != other.sep_ids):
            return False
        if len(self.nodes) != len(other.nodes):
            return False
        for n1, n2 in zip(self.nodes, other.nodes):
            if n1.reduced_tangle.shape != n2.reduced_tangle.shape:
                return False
            if (n1.reduced_tangle != n2.reduced_tangle).any():
                return False

            e1 = sorted(n1.edges, key=lambda n: n.sep_id)
            e2 = sorted(n2.edges, key=lambda n: n.sep_id)
            if len(e1) != len(e2):
                return False
            for a,b in zip(e1, e2):
                if a.sep_id != b.sep_id:
                    return False
                if a.sep_idx != b.sep_idx:
                    return False
                if (a.other_end(n1).reduced_tangle != b.other_end(n2).reduced_tangle).any():
                    return False

        return True


def create_tot(bin_tree_root, sep_ids, sep_idx, reduced_tangle_mat, le_func):
    children = bin_tree_root.children()
    while len(children) == 1:
        bin_tree_root = children[0]
        sep_idx += 1
        children = bin_tree_root.children()

    if len(children) == 0:
        return ToTNode(bin_tree_root.indicator_row, reduced_tangle_mat[bin_tree_root.indicator_row,:])

    edge = ToTEdge(sep_id = sep_ids[sep_idx], sep_idx = sep_idx)
    node1 = create_tot(children[0], sep_ids, sep_idx+1, reduced_tangle_mat, le_func)
    node2 = create_tot(children[1], sep_ids, sep_idx+1, reduced_tangle_mat, le_func)

    node1 = _find_incident_node(edge.sep_id, -1, node1, le_func)
    node2 = _find_incident_node(edge.sep_id, 1, node2, le_func)

    edge.node1, edge.node2 = node1, node2
    node1.edges.append(edge)
    node2.edges.append(edge)

    return node1

def _find_incident_node(sep_id, sep_orientation, subtree_node, le_func):
    sub_tree_edges = subtree_node.all_tot_edges()
    if len(sub_tree_edges) == 0:
        return subtree_node

    best_edge = sub_tree_edges[0]
    best_orientation = -1 if le_func(best_edge.sep_id, -1, sep_id, sep_orientation) else 1
    for edge in sub_tree_edges[1:]:
        orientation = -1 if le_func(edge.sep_id, -1, sep_id, sep_orientation) else 1
        if le_func(best_edge.sep_id, best_orientation, edge.sep_id, orientation):
            best_edge = edge
            best_orientation = orientation

    return best_edge.node2 if best_edge.node1.reduced_tangle[best_edge.sep_idx] == best_orientation else best_edge.node1
