import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import Callable, Optional

from tangles.convenience.SurveyTangles import SurveyTangles
from tangles.convenience.SurveyVariable import SurveyVariableValues
from tangles.util.tree import BinTreeNetworkX
import matplotlib.patches as patches

_nodelabelfont = {'family': 'sans-serif',
                  'color': 'red',
                  'weight': 'normal',
                  'size': 15}
_edgelabelfont = {'family': 'sans-serif',
                  'color': 'black',
                  'weight': 'normal',
                  'size': 7}


def default_draw_tangle_func(survey_tangles: SurveyTangles, ax: mpl.axes.Axes, tangle_id: int):
    ax.text(0, 0, f"{tangle_id}", horizontalalignment='center', verticalalignment='center',
            fontdict=_nodelabelfont,
            bbox=dict(facecolor='white', pad=5, boxstyle='round,pad=0.2', edgecolor='black', lw=0.5))



def plot_tangle_matrix(survey_tangles: SurveyTangles, ax :mpl.axes.Axes = None):
    if ax is None:
        _, ax = plt.subplots(1 ,1)
    tangle_mat, meta_data = survey_tangles.tangle_matrix(return_metadata=True, remove_duplicate_rows=False)
    ax.imshow(tangle_mat, interpolation='none', aspect='auto')
    ax.set_yticks(range(tangle_mat.shape[0]))
    ax.set_xticks(range(len(meta_data)), [f"{m.info[0]} {m.info[1]} {m.info[2]}" for m in meta_data], rotation=90)




def plot_tangle_search_tree(survey_tangles: SurveyTangles, ax :mpl.axes.Axes = None, remove_incomplete_tangles: bool = False, max_tree_level: Optional[int] = None,
                            draw_tangle_func: Callable[[mpl.axes.Axes, int, SurveyTangles], None] = default_draw_tangle_func,
                            draw_edge_labels=False, edge_label_func_for_meta_data=None, edge_label_is_for_child=False,
                            level_label_func = None, node_label_size=0.005, edge_label_size=0.005, level_label_width=1, _include_splitting='nodes'):
    if remove_incomplete_tangles:
        if max_tree_level is None:
            max_tree_level = len(survey_tangles.sepcified_features(only_original_features=False))
        nodes = survey_tangles.sweep.tree.k_tangles(max_tree_level, survey_tangles.agreement, include_splitting=_include_splitting)
    else:
        nodes = survey_tangles.sweep.tree.maximal_tangles(agreement=survey_tangles.agreement, max_level=max_tree_level, include_splitting=_include_splitting)
    bintree_nx = BinTreeNetworkX(nodes)

    if ax is None:
        xlim, ylim = bintree_nx.x_lim, bintree_nx.y_lim
        _, ax = plt.subplots(1, figsize=(10 ,15 *(ylim[1 ] -ylim[0] ) /(xlim[1 ] -xlim[0])))

    def draw_tangle(G, node_id, node_ax):
        bin_tree_node = G.nodes[node_id][BinTreeNetworkX.node_attr_bin_tree_node]
        draw_tangle_func(survey_tangles, node_ax, node_id)

    if draw_edge_labels:
        if edge_label_is_for_child:
            def edge_func(G, parent_id, child_id, ax):  # feel free to move out and make more generic :-)
                bin_tree_child = G.nodes[child_id][BinTreeNetworkX.node_attr_bin_tree_node]
                if len(parent_successors := list(G.neighbors(parent_id))) > 1:
                    sibling_id = parent_successors[0] if parent_successors[1] == child_id else parent_successors[1]
                    bin_tree_sibling = G.nodes[sibling_id][BinTreeNetworkX.node_attr_bin_tree_node]
                    level = min(bin_tree_child.level(), bin_tree_sibling.level())
                else:
                    level = bin_tree_child.level()

                sep_id = survey_tangles.sweep.tree.sep_ids[level-1]
                meta = survey_tangles.feature_system.separation_metadata(sep_id)
                var, op, val = meta.info[0], meta.info[1], meta.info[2] if len(meta.info) == 3 else None
                if G.nodes[child_id][BinTreeNetworkX.node_attr_is_originally_left_child]:
                    op = SurveyVariableValues.invert_op(op)
                sep_info_string = edge_label_func_for_meta_data(var, op, val) if edge_label_func_for_meta_data else f"{var} {op}" + ("" if val is None else f" {val}")
                is_visually_left_child = bintree_nx._pos[child_id][0] < bintree_nx._pos[parent_id][0]

                ax.text(-2 if is_visually_left_child else 2, 0, sep_info_string, fontdict=_edgelabelfont,
                       horizontalalignment='right' if is_visually_left_child else 'left', verticalalignment='center',
                       bbox=dict(facecolor='white', pad=2, edgecolor='black', lw=0.5))
        else:
            def edge_func(G, parent_id, child_id, ax):  # feel free to move out and make more generic :-)
                bin_tree_parent = G.nodes[parent_id][BinTreeNetworkX.node_attr_bin_tree_node]
                level = bin_tree_parent.level()

                sep_id = survey_tangles.sweep.tree.sep_ids[level]
                meta = survey_tangles.feature_system.separation_metadata(sep_id)
                var, op, val = meta.info[0], meta.info[1], meta.info[2] if len(meta.info) == 3 else None
                if G.nodes[child_id][BinTreeNetworkX.node_attr_is_originally_left_child]:
                    op = SurveyVariableValues.invert_op(op)
                sep_info_string = edge_label_func_for_meta_data(var, op, val) if edge_label_func_for_meta_data else f"{var} {op}" + ("" if val is None else f" {val}")
                is_visually_left_child = bintree_nx._pos[child_id][0] < bintree_nx._pos[parent_id][0]

                ax.text(-2 if is_visually_left_child else 2, 0, sep_info_string, fontdict=_edgelabelfont,
                       horizontalalignment='right' if is_visually_left_child else 'left', verticalalignment='center',
                       bbox=dict(facecolor='white', pad=2, edgecolor='black', lw=0.5))

    else:
        edge_func = None

    bintree_nx.draw(draw_node_label_func=draw_tangle, draw_edge_label_func=edge_func, ax=ax,
                    node_label_size=node_label_size, edge_label_size=edge_label_size, draw_levels=True,
                    level_label_func=level_label_func, level_label_width=level_label_width)



