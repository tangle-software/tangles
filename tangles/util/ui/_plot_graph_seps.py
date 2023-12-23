import numpy as np
import math
import networkx as nx
import scipy.sparse as sparse
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as sparse_linalg
from typing import List


def _ellipses_positions(part_indicators:List[np.ndarray], radius_f1:float = 0.5, radius_f2:float = 1.0, d_f:float = 1.2) -> np.ndarray:
    pos = np.empty((part_indicators[0].shape[0],2))
    x = 0
    for part_idx, part in enumerate(part_indicators):
        s = part.sum()

        r1 = 0.5 * s * radius_f1
        r2 = 0.5 * s * radius_f2
        x += d_f * r1

        steps = np.arange(s)
        pos[part,0] = x + r1*np.cos(2*math.pi*steps/s)
        pos[part,1] = r2*np.sin(2*math.pi*steps/s)

        x += d_f * r1
    pos = pos - 0.5*(pos.min(axis=0) + pos.max(axis=0))
    return pos


def _stacked_circles_positions(part_indicators:List[List[np.ndarray]], radius_f:float = 1, d_w = 1, d_h=1) -> np.ndarray:
    n_nodes = 0
    for parts in part_indicators:
        if len(parts)>0:
            n_nodes = parts[0].shape[0]
            break
    if n_nodes == 0:
        return np.empty((0,0))

    pos = np.empty((n_nodes, 2))
    max_part_size = max(max(indi.sum() for indi in part) if len(part)>0 else 0 for part in part_indicators)
    d_w *= max_part_size*radius_f  / 3
    d_h *= max_part_size*radius_f  / 20

    x = 0
    for indis in part_indicators:
        if len(indis)==0:
            x += 2*d_w
            continue
        radii = [0.5*indi.sum()*radius_f for indi in indis]
        x += max(radii)
        y = -(sum(radii) + (len(radii)-1)*0.5*d_h)
        for i in range(len(radii)):
            y += radii[i]

            s = indis[i].sum()
            steps = np.arange(s)
            pos[indis[i], 0] = x + radii[i] * np.cos(2*math.pi * steps/s)
            pos[indis[i], 1] = y + radii[i] * np.sin(2*math.pi * steps/s)

            y += radii[i] + d_h

        x += max(radii) + d_w

    pos = pos - 0.5*(pos.min(axis=0) + pos.max(axis=0))
    return pos


def svd_positions_with_data(data : np.ndarray):
    std = data.std(axis=0)
    std[std==0]=1
    data_normed = (data - data.mean(axis=0))/std
    U_svd, S_svd, Vt_svd = sparse_linalg.svds(data_normed, 2)
    pos = U_svd * S_svd
    return dict(zip(range(pos.shape[0]), [(pos[i, 0], pos[i, 1]) for i in range(pos.shape[0])]))


class SplitGraph:
    def __init__(self, A: sparse.csr_matrix, sep:np.ndarray):
        self.A = A
        self.sel_pos = sep>0
        self.sel_neg = sep<0

    @property
    def A_neg(self):
        if not hasattr(self, "_A_neg"):
            self._A_neg = self.A[self.sel_neg,:][:,self.sel_neg]
        return self._A_neg

    @property
    def A_pos(self):
        if not hasattr(self, "_A_pos"):
            self._A_pos = self.A[self.sel_pos,:][:,self.sel_pos]
        return self._A_pos

    @property
    def boundary_neg(self):
        if not hasattr(self, "_boundary_neg"):
            sel = self.sel_neg.copy()
            sel[sel] = self.A[self.sel_neg, :][:, self.sel_pos].sum(axis=1).A1 > 0
            self._boundary_neg = sel
        return self._boundary_neg

    @property
    def boundary_pos(self):
        if not hasattr(self, "_boundary_pos"):
            sel = self.sel_pos.copy()
            sel[sel] = self.A[self.sel_pos, :][:, self.sel_neg].sum(axis=1).A1 > 0
            self._boundary_pos = sel
        return self._boundary_pos

    @property
    def boundary(self):
        if not hasattr(self, "_boundary"):
            self._boundary = self.boundary_pos | self.boundary_neg
        return self._boundary

    @property
    def inner_neg(self):
        if not hasattr(self, "_inner_neg"):
            self._inner_neg = self.sel_neg & ~self.boundary_neg
        return self._inner_neg

    @property
    def inner_pos(self):
        if not hasattr(self, "_inner_pos"):
            self._inner_pos = self.sel_pos & ~self.boundary_pos
        return self._inner_pos

    def draw_ellipses(self, ax=None, inner_node_color_neg='k', boundary_node_color_neg='r',
                                        inner_node_color_pos='k', boundary_node_color_pos='g',
                                        edge_color=(0, 0, 0, 0.03), node_size=10):
        """Visualise a bipartition by plotting the nodes arranged in four ellipses.

        Draws two ellipses for each side of the bipartition: one that contains all nodes that have connections only to nodes on the same side,
        and one that contains nodes with connections to the other side.

        Parameters
        ----------
        inner_node_color_neg, boundary_node_color_neg
            Colors for the negative side.
        inner_node_color_pos, boundary_node_color_pos
            Colors for the positive side.
        edge_color
            Edge color.
        node_size
            Node size.
        """

        pos = _ellipses_positions([self.inner_neg, self.boundary_neg, self.boundary_pos, self.inner_pos])
        node_colors = [boundary_node_color_neg if self.boundary_neg[i] else
                                     boundary_node_color_pos if self.boundary_pos[i] else
                                     inner_node_color_neg if self.inner_neg[i] else
                                     inner_node_color_pos
                                     for i in range(self.A.shape[0])]

        nx.draw(nx.from_scipy_sparse_matrix(self.A), ax=ax, pos=pos,
                        node_color=node_colors, edge_color=edge_color, node_size=node_size)

    def compute_node_positions(self, horizontal=False, flip_pos_side=False,
                                                             center_boundary_parts=True,
                                                             fixed_stack_distance = None):
        part_components = []
        for p in  [self.inner_neg, self.boundary_neg, self.boundary_pos, self.inner_pos]:
            isolated_nodes = np.zeros(self.A.shape[0], dtype=bool)
            ncc, labels = csg.connected_components(self.A[p,:][:,p], return_labels=True)
            comp_indi = []
            for i in range(ncc):
                c = p.copy()
                c[c] = labels==i
                size = c.sum()
                if size == 1:
                    isolated_nodes |= c
                else:
                    comp_indi.append(c)

            comp_indi.sort(key=sum, reverse=True)
            if isolated_nodes.sum() > 0:
                comp_indi.append(isolated_nodes)
            part_components.append(comp_indi)

        pos = _stacked_circles_positions(part_components)
        if flip_pos_side:
            pos[self.sel_pos,1]*=-1

        if center_boundary_parts:
            if self.boundary.sum() > 0:
                c = 0.5*(pos[self.boundary,:].min(axis=0)+pos[self.boundary,:].max(axis=0))
                pos -= c
            else:
                print("could not center boundary:it's empty")

        if fixed_stack_distance is not None:
            x = -1.5*fixed_stack_distance
            for p in [self.inner_neg, self.boundary_neg, self.boundary_pos, self.inner_pos]:
                if p.sum() > 0:
                    p_x = 0.5*(pos[p,0].max()+pos[p,0].min())
                    pos[p, 0] += x - p_x
                x += fixed_stack_distance

        if horizontal:
            pos = pos[:,::-1]

        return pos

    def draw_stacked_circles(self, ax=None, inner_node_color_neg='k', boundary_node_color_neg='r',
                                                     inner_node_color_pos='k', boundary_node_color_pos='r',
                                                     edge_color=(0, 0, 0, 0.05), node_size=10,
                                                     marked_nodes_indicator=None, marked_node_sizes=None, marked_nodes_colors=None,
                                                     horizontal=False, flip_pos_side=False,
                                                     center_boundary_parts=True,
                                                     fixed_stack_distance=None,
                                                     prev_pos = None, update = 0.5):
        """Visualise a bipartition by plotting the nodes arranged in circles.

        Like in :meth:`draw_ellipses` we have four parts:

        - the first contains the nodes of the negative side of sep that don't have a connection to the positive side;
        - the second contains the nodes of the negative side of sep that have a connection to the positive side;
        - the third contains the nodes of the positive side of sep that have a connection to the negative side;
        - the fourth contains the nodes of the positive side of sep that don't have a connection to the negative side.

        Additionally the parts are split into connected components.
        Every one of these connected components is drawn as circle with a diameter proportional to the size, the connected components of a part are
        stacked horizontally.

        Parameters
        ----------
        ax
            A matplotlib axes object.
        inner_node_color, boundary_node_color
            Colors for the different types of nodes.
        edge_color
            Color for the edges.
        node_size
            Size of the nodes (in 'NetworkX units').
        marked_nodes_indicator
            A vector containing non-negative integers, nodes at indices with a non-zero entry are 'marked' and drawn in a different color.
        marked_node_sizes
            A list of sizes, the size of node at index ``i`` is ``marked_nodes_sizes[marked_nodes_indicator[i] - 1]`` if ``i>0``, 
            else the normal size is used.
        marked_nodes_colors
            A list of colors, the color of node at index ``i`` is ``marked_nodes_colors[marked_nodes_indicator[i] - 1]`` if ``i>0``,
            else the normal color is used.
        horizontal
            Whether the four stacks are drawn horizontal.
        flip_pos_side
            Whether the stack of the positive side is drawn in reversed order.
        center_boundary_parts
            Whether to force the mid between the boundary stacks to be at coordinate zero.
        fixed_stack_distance
            Whether to keep the distance between stacks fixed.
        prev_pos
            Previous node positions.
        update
            Update factor to interpolate between positions: 
                
                drawn_pos = (1-update) * prev_pos + update*pos.

        Returns
        -------
        np.ndarray
            The new positions (used mainly for the movie).
        """

        tmp_pos = self.compute_node_positions(horizontal, flip_pos_side, center_boundary_parts, fixed_stack_distance)
        if prev_pos is None:
            pos = tmp_pos
        else:
            pos = prev_pos*(1-update)+tmp_pos*update

        node_colors = [boundary_node_color_neg if self.boundary_neg[i] else
                                     boundary_node_color_pos if self.boundary_pos[i] else
                                     inner_node_color_neg if self.inner_neg[i] else
                                     inner_node_color_pos
                                     for i in range(self.A.shape[0])]
        Gnx = nx.from_scipy_sparse_matrix(self.A)
        nx.draw_networkx(Gnx, ax=ax, pos=pos, node_color=node_colors, edge_color=edge_color,
                        node_size=node_size, with_labels=False)

        if marked_nodes_indicator is not None:
            marked_idcs = np.flatnonzero(marked_nodes_indicator > 0)
            if marked_idcs.shape[0]>0:
                if marked_nodes_colors is None:
                    marked_colors = 'g'
                else:
                    marked_colors = [marked_nodes_colors[marked_nodes_indicator[i]-1] if marked_nodes_indicator[i]-1 < len(marked_nodes_colors) else marked_nodes_colors[-1] for i in marked_idcs]

                if marked_node_sizes is None:
                    marked_sizes = node_size
                else:
                    marked_sizes = [marked_node_sizes[marked_nodes_indicator[i] - 1] if marked_nodes_indicator[i]-1 < len(marked_node_sizes) else marked_node_sizes[-1] for i in marked_idcs]
                nx.draw_networkx_nodes(nx.convert_node_labels_to_integers(Gnx.subgraph(marked_idcs)), ax=ax, pos=pos[marked_idcs,:], node_color=marked_colors, node_size=marked_sizes)

        return pos


    def draw_side_and_boundary(self, pos_side:bool, ax=None, node_positions = None,
                                                         inner_node_color='k', boundary_node_color='r',
                                                         inner_node_size=10, boundary_node_size=10,
                                                         edge_color=(0, 0, 0, 0.05)):
        """Draw one side of the bipartition.

        Parameters
        ----------
        pos_side
            If True draw the positive side, and otherwise the negative.
        ax
            A matplotlib axes object.
        node_positions
            Positions for the nodes. If None, the default NetworkX layout is used.
        inner_node_color, boundary_node_color
            Colors of the inner nodes and boundary nodes.
        inner_node_size, boundary_node_size
            Sizes of the inner nodes and boundary nodes.
        edge_color
            The edge color.
        """
        
        if pos_side:
            A_pos_nx = nx.from_scipy_sparse_matrix(self.A_pos)
            nx.draw_networkx(A_pos_nx, ax=ax, with_labels=False,
                                             node_color=[boundary_node_color if b else inner_node_color for b in self.boundary_pos[self.sel_pos]], edge_color=edge_color,
                                             pos=node_positions, node_size=[boundary_node_size if b else inner_node_size for b in self.boundary_pos[self.sel_pos]])
        else:
            A_neg_nx = nx.from_scipy_sparse_matrix(self.A_neg)
            nx.draw_networkx(A_neg_nx, ax=ax, with_labels=False,
                                             node_color=[boundary_node_color if b else inner_node_color for b in self.boundary_neg[self.sel_neg]], edge_color=edge_color,
                                             pos=node_positions, node_size=[boundary_node_size if b else inner_node_size for b in self.boundary_neg[self.sel_neg]])