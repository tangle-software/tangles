import numpy as np
import matplotlib.pyplot as plt
from src.radar_plot import radar_factory


def soft_max(x, scaling=1):
    return np.exp(scaling * x) / np.sum(np.exp(scaling * x))


def create_factor_weights_radar_plot(ax, factor_weights, title=None):
    labels = ["E", "N", "A", "C", "O"]
    colors = [
        "olive" if w > 0 else "firebrick" if w < 0 else "grey" for w in factor_weights
    ]
    case_data = convert_to_radar_data(factor_weights)

    # plotting
    theta = radar_factory(5, frame="polygon")
    for d, color in zip(case_data, colors):
        ax.plot(theta, d, color=color)
        ax.fill(theta, d, facecolor=color, alpha=0.4, label="_nolegend_")

    # plot formatting
    ax.set_title(title, y=0.92, fontsize=7, color="grey")
    ax.grid(False)
    ax.set_ylim(0, 10)
    ax.set_rgrids([])
    ax.spines["polar"].set_color("grey")

    # set up ticks
    radian_xticks_locations = [(2 / 5) * np.pi * i + (1 / 5) * np.pi for i in range(5)]
    ax.set_xticks(radian_xticks_locations)
    ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="x", colors="grey", length=0)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(-7)
    bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.8)
    plt.setp(ax.get_xticklabels(), bbox=bbox)


def convert_to_radar_data(factor_weights):
    absolute_factor_weights = np.abs(factor_weights)
    data = np.diag(absolute_factor_weights)
    data[np.arange(5), (np.arange(5) + 1) % 5] = absolute_factor_weights
    return data


def node_sizes_by_location_sizes(
    sweep, tot, max_size=(0.1, 0.1), sizing_func=lambda x: np.sqrt(x)
):
    if len(tot.nodes) == 1:
        return {
            tot.nodes[0]: (max_size[0] * sizing_func(1), max_size[1] * sizing_func(1))
        }

    loc_sizes = {
        node: np.sum(sweep.sep_sys.compute_infimum(*node.star) == 1)
        for node in tot.nodes
    }
    max_loc_size = np.max(list(loc_sizes.values()))
    node_sizes = {}
    for node, loc_size in loc_sizes.items():
        node_size_x = max_size[0] * sizing_func((loc_size / max_loc_size))
        node_size_y = max_size[1] * sizing_func((loc_size / max_loc_size))
        node_sizes[node] = (node_size_x, node_size_y)
    return node_sizes


def efficient_distinguisher_tree_level_annotation(sweep, tot):
    eff_dist_levels = {}
    for edge in tot.edges:
        tree_level = np.where(sweep.tree._sep_ids == edge.sep_id)[0][0]
        eff_dist_levels[frozenset([edge.node1, edge.node2])] = tree_level

    def plot_edge_annotation(node_1, node_2, ax):
        level = eff_dist_levels[frozenset([node_1, node_2])]
        bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.8)
        ax.set_axis_off()
        ax.text(
            0.5, 0.5, level, size=8, c="tab:blue", ha="center", va="center", bbox=bbox
        )

    return plot_edge_annotation
