from typing import TypeVar, Optional
import numpy as np
from tangles.util.tree._tree import TreeNode

Node = TypeVar("Node", bound='BinTreeNode')
NodeRef = Optional[Node]


class BinTreeNode(TreeNode):
    """ Node of a BinTree.

    Note: when a node is created, the `parent` attribute of `left_child` and of `right_child` is set to this node.

    Parameters
    ----------
    parent : BinTreeNode, optional
        The parent of the node.
    left_child : BinTreeNode, optional
        The left child of the node.
    right_child : BinTreeNode, optional
        The right child of the node.

    Attributes
    ----------
    parent : BinTreeNode or None
        The parent of the node. If `parent` is None then the node has no parent.
    left_child : BinTreeNode or None
        The left child of the node. If `left_child` is None then the node has no left child.
    right_child : BinTreeNode or None
        The right child of the node. If `right_child` is None then the node has no right child.
    """

    def __init__(self: Node, parent: NodeRef = None, left_child: NodeRef = None, right_child: NodeRef = None) -> None:
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        if left_child is not None:
            left_child.parent = self
        if right_child is not None:
            right_child.parent = self

    @property
    def neighbours(self):
        neighbours = []
        if self.parent:
            neighbours.append(self.parent)
        if self.left_child:
            neighbours.append(self.left_child)
        if self.right_child:
            neighbours.append(self.right_child)
        return neighbours

    @property
    def root(self):
        node = self
        while node.parent:
            node = node.parent
        return node

    def set_left_child(self: Node, node: NodeRef) -> None:
        """Set the left child of the node.

        The previous left child does not point to this node after this method call, and instead `node` does.

        Parameters
        ----------
        node
            The new left child.
        """

        if self.left_child is not None and self.left_child.parent is self:
            self.left_child.parent = None
        self.left_child = node
        if node is not None:
            node.parent = self

    def set_right_child(self: Node, node: NodeRef) -> None:
        """Set the right child of the node.

        The previous right child does not point to this node after this method call, and instead `node` does.

        Parameters
        ----------
        node
            The new right child.
        """

        if self.right_child is not None and self.right_child.parent is self:
            self.right_child.parent = None
        self.right_child = node
        if node is not None:
            node.parent = self

    def num_children(self):
        return (1 if self.left_child else 0) + (1 if self.right_child else 0)


    def remove_single_parents_in_subtree(self):
        if self.left_child: self.left_child._remove_single_parents_in_subtree_internal(self, -1)
        if self.right_child: self.right_child._remove_single_parents_in_subtree_internal(self, 1)

    def _attach_to_parent(self, new_parent, lr_indicator):
        if lr_indicator<0: new_parent.set_left_child(self)
        else: new_parent.set_right_child(self)

    def _remove_single_parents_in_subtree_internal(self, parent_to_keep, lr_indicator):
        if self.left_child and self.right_child:
            self._attach_to_parent(parent_to_keep, lr_indicator)
            self.left_child._remove_single_parents_in_subtree_internal(self, -1)
            self.right_child._remove_single_parents_in_subtree_internal(self, 1)
        elif self.left_child:
            self.left_child._remove_single_parents_in_subtree_internal(parent_to_keep, lr_indicator)
        elif self.right_child:
            self.right_child._remove_single_parents_in_subtree_internal(parent_to_keep, lr_indicator)
        else:
            self._attach_to_parent(parent_to_keep, lr_indicator)


    def detach(self: Node) -> None:
        """Detach the node by removing its pointer to its parent and the parents pointer to it."""

        if not self.parent:
            return
        if self == self.parent.left_child:
            self.parent.left_child = self.parent = None
        else:
            self.parent.right_child = self.parent = None

    def copy(self) -> Node:
        """Create a new BinTreeNode (note: this is a kind of abstract base function)."""

        return BinTreeNode()

    def copy_subtree(self) -> Node:
        """
        Copy the subtree starting at this node. The returned subtree is not connected to the original tree.

        Returns
        -------
        Node
            Root of the new subtree.
        """

        start_node_copy = self.copy()
        stack = [(self, start_node_copy)]
        while stack:
            node, node_copy = stack.pop()

            if node.left_child:
                left_copy = node.left_child.copy()
                node_copy.set_left_child(left_copy)
                stack.append((node.left_child, left_copy))
            if node.right_child:
                right_copy = node.right_child.copy()
                node_copy.set_right_child(right_copy)
                stack.append((node.right_child, right_copy))

        return start_node_copy

    def copy_subtree_into_children(self, left: bool = True, right: bool = True):   # TODO: Find official name for this operation!
        """
        Replace each child by a copy of the subtree starting at this node.
        """

        left_child, right_child = self.left_child, self.right_child
        left_node, right_node = None, None
        copied_node = self.copy()
        copied_node.set_left_child(left_child)
        copied_node.set_right_child(right_child)
        if left and right:
            left_node = self.copy_subtree()
            right_node = copied_node
        elif left:
            left_node = copied_node
        elif right:
            right_node = copied_node
        self.set_left_child(left_node)
        self.set_right_child(right_node)

    def children(self: Node) -> list[Node]:
        """ Return the list of children of this node.

        Returns
        -------
        list of BinTreeNode
            The list of children of this node.
        """

        children = []
        if self.left_child is not None:
            children.append(self.left_child)
        if self.right_child is not None:
            children.append(self.right_child)
        return children


    def child(self, child_identifier:int):
        return self.left_child if child_identifier<0 else self.right_child

    def sibling(self) -> Optional[Node]:
        if self.parent:
            return self.parent.right_child if self is self.parent.left_child else self.parent.left_child
        return None

    def leaves_in_subtree(self: Node, max_depth: Optional[int] = None) -> list[Node]:
        """ Find all leaves in the binary tree.

        Parameters
        ----------
        max_depth
            The maximum depth to search. If `max_depth` is not None then `max_depth` is the
            number of layers which are checked.

        Returns
        -------
        list of BinTreeNode
            The found nodes.
        """

        leaves = []
        nodes = [self]
        depth = 1
        while nodes and (max_depth is None or depth <= max_depth):
            next_nodes = []
            for node in nodes:
                children = node.children()
                next_nodes.extend(children)
                if len(children) == 0:
                    leaves.append(node)
            nodes = next_nodes
            depth += 1
        return leaves + nodes

    def level_in_subtree(self: Node, depth: int) -> list[Node]:
        """ Return all the nodes at a certain `depth` below this node.

        Parameters
        ----------
        depth
            The depth of the level to be returned.
            Depth of 0 means that you get a list just containing the node itself.

        Returns
        -------
        list of BinTreeNode
            The list of all nodes at the specified depth.
        """

        nodes = [self]
        for _ in range(depth):
            next_nodes = []
            for node in nodes:
                next_nodes.extend(node.children())
            nodes = next_nodes
            if not nodes:
                break
        return nodes

    def all_levels_in_subtree(self):
        current_level = [self]
        levels = []
        while current_level:
            levels.append(current_level)
            current_level = sum((n.children() for n in current_level),[])
        return levels

    def is_leaf(self: Node) -> bool:
        """Whether this node is a leaf. 
        
        A leaf is a node with no children.
        """

        return self.left_child is None and self.right_child is None

    def level(self: Node) -> int:
        """ Returns the level of this node.

        The root has level 0.

        Returns
        -------
        int
            The nodes' level.
        """

        level = 0
        node = self
        while node.parent:
            node = node.parent
            level += 1
        return level

    def path_from_root_indicator(self: Node) -> list[int]:
        """ Returns the list of sides one has to take to go from the root to this node.

        The value 1 denotes going right, the value -1 denotes going left.

        Returns
        -------
        list[int]
            The list of indicator values.
        """

        path = []
        node = self
        while node.parent:
            indicator = -1 if (node == node.parent.left_child) else 1
            path.append(indicator)
            node = node.parent
        path.reverse()
        return path

    def subtree_height(self) -> int:
        height = -1
        nodes = [self]
        while nodes:
            next_nodes = []
            for node in nodes:
                if node.left_child:
                    next_nodes.append(node.left_child)
                if node.right_child:
                    next_nodes.append(node.right_child)
            nodes = next_nodes
            height += 1
        return height

    @staticmethod
    def to_indicator_matrix(nodes: list[Node]) -> np.ndarray:
        """Turn a list of nodes into an indicator matrix.
        
        The indicator matrix is the matrix containing, in its rows, the path-from-root-indicators of the nodes given.
        The rows are sorted the same way the nodes are.

        Parameters
        ----------
        nodes : list of Node
            The nodes to encode in the matrix.

        Returns
        -------
        np.ndarray
            The indicator matrix of shape (``len(num_nodes)``, `max_depth`), where `max_depth` is the maximum
            depth of a node in `nodes`.
        """

        if len(nodes)==0:
            return np.empty((0,0))

        indicators = [node.path_from_root_indicator() for node in nodes]
        if len(indicators) == 0:
            return np.empty((0,0))
        max_len = max(len(indicator) for indicator in indicators)
        return np.vstack([indicator + [0] * (max_len - len(indicator)) for indicator in indicators])

    @classmethod
    def from_indicator_matrix(cls, indicator_matrix: np.ndarray, root: Node = None) -> 'BinTreeNode':
        """Turn an indicator matrix back into a binary tree.

        Parameters
        ----------
        indicator_matrix
            The indicator matrix to turn into a binary tree.

        Returns
        -------
        BinTreeNode
            The root of a binary tree which contains every path from the indicator matrix as a node.
        """
        
        if len(indicator_matrix.shape)==1:
            indicator_matrix = indicator_matrix.reshape(1,indicator_matrix.shape[0])

        root = root or cls()
        for i, path in enumerate(indicator_matrix):
            node = root
            for direction in path:
                if direction == 0:
                    break
                if direction == 1:
                    if node.right_child is None:
                        node.right_child = cls(parent=node)
                    node = node.right_child
                elif direction == -1:
                    if node.left_child is None:
                        node.left_child = cls(parent=node)
                    node = node.left_child
            node.indicator_row = i
        return root


if __name__=="__main__":
    from tangles.util.tree._nx_bin_tree import BinTreeNetworkX
    mat = np.array([[-1,1,-1,1,1],
                    [-1,-1,1,-1,1],
                    [-1,-1,-1,1,1],
                    [-1,1,-1,1,-1],
                    [-1,-1,1,1,1],
                    [ 1,-1,1,1,1]])

    bintree = BinTreeNode.from_indicator_matrix(mat)
    levels = bintree.all_levels_in_subtree()
    for i, l in enumerate(levels):
        for n in l:
            n.depth = i

    bintree.remove_single_parents_in_subtree()

    nxbintree = BinTreeNetworkX(bintree.all_nodes())
    nxbintree.tst_layout([n.depth for n in nxbintree._nodes])
    nxbintree.draw()