from typing import Optional, Union, Generator
import pickle
import numpy as np
from collections import Counter
from tangles.util.tree import BinTreeNode
from tangles import Tangle

class TangleSearchTree:
    """ 
    A rooted binary tree that is created while searching tangles. It represents the search result. 

    Nodes in the tree represent tangles and edges represent orientations of separations.
    Each level of the tree (except the root level) corresponds to one separation. The root node is the empty tangle.

    Each node on level :math:`k` that is a left child of a node on level :math:`k-1` represents a tangle that orients
    the separation corresponding to level :math:`k` to the left (and vice versa for a right child).

    During the tangle search, the tangle search tree is extended adding new edges to the tree.
    
    A TangleSearchTree object contains the actual binary tree of Tangle objects as well as a list of separation ids
    corresponding to the levels of the tangle search tree.
    
    Parameters
    ----------
    root : Tangle
        The root node of the tangle search tree.
    sep_ids : np.ndarray
        The separation ids in a 1-dimensional array.
    """

    def __init__(self, root:Tangle, sep_ids: np.ndarray):
        self._root = root
        self._sep_ids = sep_ids

    def max_leaf_agreement(self, check_last_level=False) -> int:
        limit = 0
        for _, level in self._levels(self._root, 0, len(self._sep_ids) + (1 if check_last_level else 0)):
            leafs = [node for node in level if node.is_leaf()]
            for leaf in leafs:
                if leaf.agreement >= limit:
                    limit = leaf.agreement
        return limit

    @property
    def limit(self):    # TODO: I assume, we usually don't want to check the last level's limit (it is not extendable anyways)
        """ The maximum order of a node which we have not yet extended. 
        
        For every value greater than this limit we know with certainty every tangle (as the agreeement value decreases
        as one increases the order of the tangles).
        But for the limit itself it might be the case that extending the node which has the limit agreement yields
        another node of the same agreement.

        Returns
        -------
        int
            The agreement search limit.
        """
        
        return self.max_leaf_agreement(False)

    def tree_height_for_agreement(self, agreement: int = None) -> int:
        """ Compute the tree height for the given agreement

        Parameters
        ----------
        agreement: int
            An agreement value

        Returns
        -------
        int:
            the tree height

        """
        level_sizes = self.level_sizes_for_agreement(agreement)
        first_empty_level = (level_sizes == 0).argmax()
        if first_empty_level == 0:
            return len(level_sizes)-1 if level_sizes[0]>0 else 0
        else:
            return first_empty_level-1

    def search_tree(self, agreement: int = 1, max_level: Optional[int] = None) -> Tangle:
        """
        Build a copy of the tangle search tree.
        
        The returned search tree does not contain tangles of agreement less than the specified `agreement` value 
        and if `max_level` is specified it contains only tangles on levels up to `max_level`.

        Parameters
        ----------
        agreement : int
            Tangles in the tangle search tree will have at least this agreement values.
        max_level : int, optional
            Only build the tree up to this level.

        Returns
        -------
        Tangle
            The root tangle of the new tree.
        """

        new_root = self.root.copy_subtree()
        level = [new_root]
        while level:
            new_level = []
            for tangle in level:
                if tangle.left_child:
                    if tangle.left_child.agreement < agreement:
                        tangle.set_left_child(None)
                    else:
                        new_level.append(tangle.left_child)
                if tangle.right_child:
                    if tangle.right_child.agreement < agreement:
                        tangle.set_right_child(None)
                    else:
                        new_level.append(tangle.right_child)
            level = new_level
        return new_root


    def maximal_tangles(self, agreement: int = 1, max_level:Optional[int] = None, include_splitting:str = "nope"):
        """Return all maximal tangles of at least the specified `agreement` and level at most `max_level`.

        Guaranteed to return every tangle (on the set of sep ids the sweep knows about) if the limit is below the
        specified `agreement`.

        Parameters
        ----------
        agreement : int
            All tangles of at least this agreement value are returned.
        max_level : int, optional
            Only return tangles with level below or equal `max_level`.
        include_splitting: {"nope", "nodes", "levels"}
            If equal to "nodes" all nodes which distinguish two maximal tangles are included.
            If equal to "levels" all nodes on a path to a maximal tangle at levels that contain a distinguishing node are included.
            Defaults to "nope".
        
        Returns
        -------
        list
            The maximal tangles.
        """

        include_spl_levels, include_spl_nodes = include_splitting == "levels", include_splitting == "nodes"

        nodes = []
        last_level = None
        for _, level in self._levels(self.root, agreement, max_level if max_level is not None else None):
            last_level = level
            if include_spl_levels:
                if next((n for n in level if len([c for c in n.children() if c.agreement >= agreement]) != 1), 0): nodes.extend(level)
            elif include_spl_nodes:
                nodes.extend(n for n in level if len([c for c in n.children() if c.agreement >= agreement]) != 1)
            else:
                nodes.extend(n for n in level if len([c for c in n.children() if c.agreement >= agreement]) == 0)

        # add all children of the last level (We want a num_tangles x max_level - matrix!, we should be more carefully specificate what is a level!)
        if last_level:
            nodes.extend(c.left_child for c in last_level if c.left_child and c.left_child.agreement >= agreement)
            nodes.extend(c.right_child for c in last_level if c.right_child and c.right_child.agreement >= agreement)
        return nodes


    def tangle_matrix(self, agreement: int = 1, max_level:Optional[int] = None, include_splitting = False, return_nodes = False) -> Union[np.ndarray, tuple[np.ndarray, list[Tangle]]]:
        """Return the tangle matrix of all maximal tangles, taken from the set of tangles of at least the specified agreement.

        Guaranteed to return every tangle (on the set of separation ids the sweep knows about) if the limit is below
        the specified `agreement`.

        Parameters
        ----------
        agreement : int
            All tangles of at least this agreement value are returned.
        max_level : int, optional
            Only return tangles with level less than `max_level`.

        Returns
        -------
        np.ndarray
            Tangle matrix.
        """

        nodes = self.maximal_tangles(agreement, max_level, include_splitting='nodes' if include_splitting else 'nope')
        matrix = BinTreeNode.to_indicator_matrix(nodes)
        return (matrix, nodes) if return_nodes else matrix


    def get_efficient_distinguishers(self, return_ids=True, agreement:int = 1, max_level: Optional[int] = None) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Find the efficient distinguishers of the tangles. 
        
        The efficient distinguishers are separations for which there exists a pair of tangles such that they
        distinguish the two separations with minimal order.

        Parameters
        ----------
        return_ids : bool
            Whether to return the separation ids of the efficient distinguishers.
        agreement : int
            The efficient distinguishers will be returned for all tangles of at least this agreement value.
        max_level : int, optional
            The efficient distinguishers will be returned for all tangles with level below `max_level`.

        Returns
        -------
        np.ndarray or tuple
            If return_ids is False, the levels of the efficient distinguishers.
            If return_ids is True, a tuple with the levels of the efficient distinguishers in its first entry and the
            separation ids of the efficient distinguishers in its second entry.
        """

        level_idcs = []
        ids = []
        for i, level in self._levels(self._root, agreement, max_level or len(self._sep_ids)):
            for node in level:
                if node.left_child is not None and node.right_child is not None and \
                   node.left_child.agreement >= agreement and node.right_child.agreement >= agreement:
                    level_idcs.append(i)
                    ids.append(self.sep_ids[i])
                    break
        return (np.array(level_idcs), np.array(ids)) if return_ids else np.array(level_idcs)


    def k_tangles(self, k:int, agreement:int, include_splitting:str = "nope") -> list[Tangle]:
        """
        Return all tangles of the `k`-th level which have at least the specified `agreement` value.

        Parameters
        ----------
        k : int
            The level of which to find every tangle.
        agreement : int
            Only find tangles of at least the specified agreement value.
        include_splitting: {"nope", "nodes", "levels"}
            If equal to "nodes" all nodes which distinguish two `k`-tangles are included.
            If equal to "levels" all nodes on a path to a `k`-tangle at levels that contain a distinguishing node are included.
            Defaults to "nope".

        Returns
        -------
        list of Tangles
            The tangles of the `k`-th level.
        """

        nodes = [node for node in self._root.level_in_subtree(k) if node.agreement >= agreement]
        include_levels, include_nodes = include_splitting == "levels", include_splitting == "nodes"
        if include_levels or include_nodes:
            parents = [node.parent for node in nodes]
            while parents and parents[0] is not None:
                counter = Counter(parents)
                splitting = [n for n,c in counter.items() if c>1]
                if include_nodes:
                    nodes.extend(splitting)
                elif splitting:
                    nodes.extend(list(counter.keys()))
                parents = [n.parent for n in counter.keys()]

        return nodes

    def level_sizes_for_agreement(self, agreement:int):
        sizes = np.empty(len(self.sep_ids)+1, dtype=int)
        for i,nodes in self._levels(self._root, agreement=agreement):
            sizes[i] = len(nodes)
        return sizes


    def k_tangle_matrix(self, k:int, agreement:int) -> np.ndarray:
        nodes = self.k_tangles(k, agreement)
        return BinTreeNode.to_indicator_matrix(nodes)

    @property
    def number_of_separations(self) -> int:
        """The number of separations in the search tree."""

        return len(self._sep_ids)

    def _insert_sep_id(self, insertion_idx:int, new_sep_id:int):
        self._sep_ids = np.insert(self._sep_ids, insertion_idx, new_sep_id)

    def _sep_ids_to_update_after_insertion(self, insertion_idx:int) -> np.ndarray:
        return self._sep_ids[(insertion_idx + 1):]

    @property
    def root(self):
        """The root of the search tree."""

        return self._root

    @property
    def sep_ids(self):
        """
        The separation ids associated with the levels. 

        Note that index 0 corresponds to the separation added after the root layer.
        """

        return self._sep_ids

    def _levels(self, starting_node:Tangle, agreement: int, max_depth:Optional[int]=None) -> Generator[tuple[int, list[Tangle]], None, None]:
        level = [starting_node]
        starting_level = level[0].level()
        max_depth = max_depth if max_depth is not None else len(self._sep_ids)+1   # Danger: max_depth or None behaves in an unexpected way for max_depth == 0
        for i in range(starting_level, max_depth):
            level = [node for node in level if node.agreement >= agreement]
            yield i, level
            level = [node.left_child for node in level if node.left_child] + [node.right_child for node in level if node.right_child]
            if not level:
                break

    def is_subtree_of(self, tree: 'TangleSearchTree') -> bool:
        level = [tree.root]
        for _, my_level in self._levels(self.root, -2):
            new_level_left, new_level_right = [], []
            for my_node, node in zip(my_level, level):
                if not my_node.equal_data(node):
                    print("Nodes not equal", my_node.__dict__, node.__dict__)
                    return False
                if ((not node.left_child) and my_node.left_child) or ((not node.right_child) and my_node.right_child):
                    print("Your tree has extra nodes")
                    return False
                if my_node.left_child:
                    new_level_left.append(node.left_child)
                if my_node.right_child:
                    new_level_right.append(node.right_child)
            level = new_level_left + new_level_right
        return True

    def save(self, filename: str):
        """Saves the search tree data to a file."""

        nodes = []
        parent_index = []
        which_child = []
        cores = []
        tangle_agreements = []
        for _, level in self._levels(self.root, -2):
            for node in level:
                nodes.append(node)
                cores.append(node.core)
                tangle_agreements.append(node.agreement)
                if node.parent:
                    parent_index.append(nodes.index(node.parent))
                    if node is node.parent.left_child:
                        which_child.append(-1)
                    else:
                        which_child.append(1)
                else:
                    parent_index.append(-1)
                    which_child.append(0)
        with open(filename, 'wb') as f:
            pickle.dump([parent_index, which_child, cores, tangle_agreements, self.sep_ids], f)

    @staticmethod
    def load(filename: str) -> 'TangleSearchTree':
        """
        Build a search tree from the data in `filename`.

        Parameters
        ----------
        filename : str
            The path of the file containing the data of the tangle search tree.

        Returns
        -------
        TangleSearchTree
            The tangle search saved in `filename`.
        """

        with open(filename, 'rb') as f:
            list = pickle.load(f)
        parent_index = list[0]
        which_child = list[1]
        cores = list[2]
        tangle_agreements = list[3]
        sep_ids = list[4]
        nodes = []
        for i in range(len(parent_index)):
            parent = None if parent_index[i] == -1 else nodes[parent_index[i]]
            node = Tangle(tangle_agreements[i], cores[i], parent)
            if parent:
                if which_child[i] == -1:
                    parent.left_child = node
                else:
                    parent.right_child = node
            nodes.append(node)
        return TangleSearchTree(nodes[0], sep_ids)
