import networkx as nx
import matplotlib.pyplot as plt
from tangles.util.tree import BinTreeNode
from typing import Optional
from scipy.stats import rankdata


class BinTreeNetworkX:
    """
    Responsible for generating and managing a NetworkX representation of a list of BinTreeNodes.

    A NetworkX binary tree is created from `nodes`.
    The nodes are indexed by a list of ids.
    Every node is connected to every direct successor.

    Depending on the choice of nodes this means that there might be nodes with more than two descendents.

    Parameters
    ----------
    nodes : list of BinTreeNode
        The list of nodes which should be included in the tree.
    ids : list of int, optional
        The ids of the input nodes. Defaults to the indices of the nodes.
    labels : dict, optional
        The labels of the input nodes. Defaults to string representations of the ids.
    """

    # TODO: this class is horrible... -> get rid of networkx!

    node_attr_bin_tree_node = 'btn'
    node_attr_is_originally_left_child = 'lc'

    def __init__(self, nodes: list[BinTreeNode], ids: Optional[list[int]] = None, labels: Optional[dict[int, str]] = None,
                 depths_for_layout=None):
        if next((n for n in nodes if n.parent is None), None) is None:
            nodes.append(nodes[0].root)

        if ids is None:
            ids = list(range(len(nodes)))
        elif len(ids)<len(nodes):
            ids.extend(list(range(-1,len(ids)-len(nodes)-1,-1)))

        self._labels = labels or {tid: str(tid) for tid in ids}
        self._graph = self._build_graph(nodes, ids)
        self._root = next(n for n in self._graph.nodes if self._graph.nodes[n][BinTreeNetworkX.node_attr_bin_tree_node].parent is None)
        self._pos = None

        self.tst_layout(rankdata([node.level() for node in nodes], method="dense") if not depths_for_layout else depths_for_layout)

    def _build_graph(self, nodes: list[BinTreeNode], ids: list[int]) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(ids)
        nx.set_node_attributes(graph, dict(zip(ids,nodes)), BinTreeNetworkX.node_attr_bin_tree_node)
        child_side_dict = {}
        for id, node in zip(ids, nodes):
            test_node = node.parent
            while test_node is not None and test_node not in nodes:
                node = test_node
                test_node = test_node.parent
            if test_node:
                parent_idx = nodes.index(test_node)
                graph.add_edge(ids[parent_idx], id)
                child_side_dict[id] = test_node.left_child == node
        nx.set_node_attributes(graph, child_side_dict, BinTreeNetworkX.node_attr_is_originally_left_child)
        return graph

    @property
    def graph(self) -> nx.DiGraph:
        """
        The underlying directed graph of the tree.

        Returns
        -------
        nx.DiGraph
            The tangle search tree given as a tree.
            The nodes are indexed by the tangle ids.
            The edges point away from the root.
        """

        return self._graph

    def relabel(self, node_id: int, new_label: str):
        """
        Relabel a node of the tree.

        Parameters
        ----------
        node_id : int
            The id of the node to relabel.
        new_label : str
            The new label of the node.
        """

        self._labels[node_id] = new_label

    def get_label(self, node_id: int) -> str:
        """
        Get the label of the node representing the tangle with the specified tangle id.

        Parameters
        ----------
        node_id : int
            The id of the tangle to access.

        Returns
        -------
        str
            The label of the node.
        """

        return self._labels[node_id]

    def get_ids_from_label(self, label: str) -> list[int]:
        """
        Get a list of all tangle ids which are labeled in the tree by the specified label.

        Parameters
        ----------
        label : str
            The label to look for in the tree.

        Returns
        -------
        list of int
            The list of all node ids which are labeled by `label` in the tree. 
            If the label does not exist in the tree an empty list is returned.
        """

        return [node_id for node_id in self._labels.keys() if self._labels[node_id] == label]

    def tst_layout(self, depths: dict[int, float]) -> Optional[dict[int, tuple[float, float]]]:
        """
        Position the nodes of the tree in the following way:
        
        - the y-coordinates are determined by `depths`. Since the y-axis of the coordinate system
          points up the heights are inverted for the coordinates so that the tree grows towards the bottom.
        - the x coordinates are chosen such that each subtree of a node lies entirely on one side of the
          node. If a node has two subtrees they must lie on different side. This layout fails if a node
          has 3 subtrees and in that case the function returns None.

        Parameters
        ----------
        depths : dict
            The depths of each node.

        Returns
        -------
        dict or None
            The positions if the layout works. Otherwise None.
        """

        if self._has_degree_3_outgoing_node():
            return None
        node_order = self._find_tst_layout_node_order()
        self._pos = {}
        for x, node in enumerate(node_order):
            self._pos[node] = (x, -depths[node])
        return self._pos

    def _has_degree_3_outgoing_node(self) -> bool:
        for node in self._graph.nodes:
            if self._graph.out_degree(node) > 2:
                return True

    def _find_tst_layout_node_order(self) -> list[int]:
        node_order = [self._root]
        checked = [False]
        try_again = True
        while try_again:
            try_again = False
            new_node_order = []
            new_checked = []
            for node, checked_already in zip(node_order, checked):
                if checked_already:
                    new_node_order.append(node)
                    new_checked.append(True)
                    continue
                children = list(self._graph.neighbors(node))
                if len(children) == 0:
                    new_node_order.append(node)
                    new_checked.append(True)
                elif len(children) == 1:
                    new_node_order.append(node)
                    new_node_order.append(children[0])
                    new_checked.append(True)
                    new_checked.append(False)
                    try_again = True
                else:
                    new_node_order.append(children[1])
                    new_node_order.append(node)
                    new_node_order.append(children[0])
                    new_checked.append(False)
                    new_checked.append(True)
                    new_checked.append(False)
                    try_again = True
            node_order = new_node_order
            checked = new_checked
        return node_order


    @property
    def x_lim(self):
        return min(p[0] for p in self._pos.values()), max(p[0] for p in self._pos.values())

    @property
    def y_lim(self):
        return min(p[1] for p in self._pos.values()), max(p[1] for p in self._pos.values())


    def draw(self, ax = None, draw_node_label_func=None, draw_edge_label_func=None,
            node_label_size=0.05, edge_label_size=0.03, level_label_width=1, draw_levels=False, level_label_func=None):
        """Draw the tree and label the tangles with their id."""

        fig, ax = plt.subplots(1, 1, figsize=(10, 10)) if ax is None else (ax.get_figure(), ax)

        tr_figure = ax.transData.transform
        tr_axes = fig.transFigure.inverted().transform

        if draw_node_label_func is None:
            nx.draw_networkx(self._graph, ax=ax, pos=self._pos, labels=self._labels)

            xlim = ax.get_xlim()    # hack to avoid moving positions out of sync later... TODO v1.1
            ax.set_xlim(xlim[0] - level_label_width, xlim[1])
        else:
            nx.draw_networkx_edges(self._graph, pos = self._pos, ax=ax,
                                   arrows=True, arrowstyle="-",
                                   min_source_margin=15, min_target_margin=15)
            xlim = ax.get_xlim()     # hack to avoid moving positions out of sync later... TODO v1.1
            ax.set_xlim(xlim[0] - level_label_width, xlim[1])

            G = self._graph
            for node_id in G.nodes:
                xa, ya = tr_axes(tr_figure(self._pos[node_id]))
                node_ax = plt.axes([xa-node_label_size/2, ya-node_label_size/2, node_label_size, node_label_size])
                node_ax.set_xlim(xmin=-1, xmax=1)
                node_ax.set_ylim(ymin=-1, ymax=1)
                draw_node_label_func(G,node_id, node_ax)
                node_ax.axis("off")


        if draw_edge_label_func is not None:
            G = self._graph
            for parent_id in G.nodes:
                children = list(G.neighbors(parent_id))
                if not children: continue

                parent_pos = self._pos[parent_id]
                label_y = (parent_pos[1] + max(self._pos[child_id][1] for child_id in children))/2
                for child_id in children:
                    child_pos = self._pos[child_id]
                    label_x = parent_pos[0] + ((label_y-parent_pos[1])/(child_pos[1]-parent_pos[1]))*(child_pos[0]-parent_pos[0])
                    xa, ya = tr_axes(tr_figure((label_x, label_y)))
                    label_ax = plt.axes([xa-edge_label_size/2.0, ya-edge_label_size/2.0, edge_label_size, edge_label_size])
                    label_ax.set_xlim(xmin=-1, xmax=1)
                    label_ax.set_ylim(ymin=-1, ymax=1)
                    draw_edge_label_func(G, parent_id, child_id, label_ax)
                    label_ax.axis("off")

        if draw_levels and self._pos:
            txt_x = 0.5 * (ax.get_xlim()[0] + min(p[0] for p in self._pos.values()))
            for ypos, level in sorted(set((self._pos[n][1], self._graph.nodes[n][BinTreeNetworkX.node_attr_bin_tree_node].level()) for n in self.graph.nodes)):
                ax.axhline(y=ypos, linestyle='--', zorder=-2)
                level_string, v_alignment, y_offset = str(level), 'center', 0
                if level_label_func:
                    level_string = level_label_func(level)
                    if isinstance(level_string,tuple):
                        if len(level_string)==2:
                            level_string, v_alignment = level_string
                        elif len(level_string)==3:
                            level_string, v_alignment, y_offset = level_string

                level_label = level_string
                ax.text(txt_x, ypos+y_offset, level_label , backgroundcolor="w", horizontalalignment='center', verticalalignment=v_alignment, zorder=-1)

