from typing import Optional
from tangles.util.tree import BinTreeNode
from tangles._typing import OrientedSep


class Tangle(BinTreeNode):
    """ A node in the tangle search tree. It represents a tangle.

    The path of the tangle search tree from the root to this node determines the features in this tangle:
    Each edge on the path determines the orientation of a separation.

    More precisely, if the path contains an edge from a node on tree level :math:`k-1` to a node on tree level
    :math:`k` which is a left child, then this tangle contains the left orientation of the separation corresponding to
    tree level :math:`k` (and vice versa for a right child).

    Attributes
    ----------
    parent : Tangle or None
        The parent in the tangle search tree. It is a tangle that orients every separation except for the highest order
        separation of this tangle.
        None if the tangle is the root tangle.
    left_child : Tangle or None
        The left child in the tangle search tree. It is a tangle that orients one more separation than this tangle.
        The left child orients this additional separation to the left.
        None if there is no left child.
    right_child : Tangle or None
        The right child in the tangle search tree. It is a tangle that orients one more separation than this tangle.
        The right child orients this additional separation to the right.
        None if there is no right child.
    core : set[OrientedSep]
        The set of all minimal oriented separations contained within this tangle.
    agreement : int
        The agreement value of this tangle.
    """

    def __init__(self, agreement: int, core:Optional[set[OrientedSep]] = None, parent: Optional['Tangle'] = None):
        super().__init__(parent)
        self.core:set[OrientedSep] = core or set()
        self.agreement = agreement

    def copy(self) -> 'Tangle':
        """Create a new tangle with the same core and agreement value.

        Returns
        -------
        Tangle
            New copy of this tangle.
        """

        return Tangle(agreement=self.agreement, core=self.core)

    def open(self):
        """Open the tangle by removing its children."""

        self.set_left_child(None)
        self.set_right_child(None)

    def leaves_in_subtree(self, agreement: int = 0) -> list['Tangle']:
        leaves = []
        nodes = [self]
        while nodes:
            next_nodes = []
            for node in nodes:
                children = [tangle for tangle in node.children() if tangle.agreement >= agreement]
                next_nodes.extend(children)
                if len(children) == 0:
                    leaves.append(node)
            nodes = next_nodes
        return leaves

    def equal_data(self, node: 'Tangle') -> bool:
        """Check whether two tangles have the same data (core and agreement value).

        Returns
        -------
        bool
            Whether the data is equal.
        """

        return self.agreement == node.agreement and self.core == node.core
