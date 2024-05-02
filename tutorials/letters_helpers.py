import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from skimage.color import rgb2gray
from skimage import io
from skimage.measure import find_contours
from scipy.signal import find_peaks
from tangles.util.graph.cut_weight import CutWeightOrder

from tangles.util.ui import XYPicker


def read_letters(letters_img_path, binary=True, thresh=0.5):
    all_letters_img = rgb2gray(io.imread(letters_img_path))
    if binary:
        all_letters_img = all_letters_img > thresh
    letter_size = all_letters_img.shape[0]
    num_letters = all_letters_img.shape[1] // letter_size
    return [
        all_letters_img[:, i * letter_size : (i + 1) * letter_size]
        for i in range(num_letters)
    ]


def k_smallest_local_min(
    f: np.ndarray, k: int = 2, include_diff0: bool = True
) -> np.ndarray:
    """small helper function to compute the 'k' smallest local minima in an array given by 'f'

    Parameters
    ----------
        f:
            array where we want to find local minima

        k:
            number of local minima

        include_diff0:
            if True, we include indices where the first difference is zero (plateaus)
    Returns
    -------
        np.ndarray: list of indices
    """
    idcs = find_peaks(-f)[0]  # search peaks in negated array
    if include_diff0:
        idcs = np.append(
            idcs, np.flatnonzero((f[:-1] - f[1:]) == 0)
        )  # add plateaus (if requested)
    return idcs[np.argpartition(f[idcs], k)[:k]]  # return


def plot_letter_sep_examples(letter_img, boundary_point_idcs):
    boundary_points = find_contours(letter_img)[0]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(letter_img, cmap="gray")
    ax.axis("off")

    lines = []

    def update_lines(sel_point_idcs):
        for l in lines:
            l.remove()
        lines.clear()
        for sel_point_idx in sel_point_idcs:
            loc_min_idcs = k_smallest_local_min(
                np.abs(boundary_points - boundary_points[sel_point_idx, :]).sum(axis=1)
            )
            if loc_min_idcs[0] != sel_point_idx:
                loc_min_idcs[1] = sel_point_idx
            points = boundary_points[loc_min_idcs, :]
            lines.extend(
                ax.plot(points[:, 1], points[:, 0], marker="o", c="r", linestyle="-")
            )
        fig.canvas.draw_idle()

    mouse_down = False

    def mouse_event(event):
        global mouse_down
        if event.inaxes == ax:
            mouse_down = event.name == "button_press_event" or (
                mouse_down and event.name != "button_release_event"
            )
            if mouse_down:
                sel_point_idx = (
                    np.square(boundary_points - np.array([event.ydata, event.xdata]))
                    .sum(axis=1)
                    .argmin()
                )
                update_lines([sel_point_idx])

    fig.canvas.mpl_connect("button_press_event", mouse_event)
    fig.canvas.mpl_connect("button_release_event", mouse_event)
    fig.canvas.mpl_connect("motion_notify_event", mouse_event)

    update_lines(boundary_point_idcs)


def create_zoom_plot(
    letter_img, adjacency, sep=None, zoom_half_size=15, figsize=(10, 5)
):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    img_rbg = np.repeat((letter_img[:, :, np.newaxis] > 0).astype(float), 3, axis=2)
    if sep is not None:
        img_rbg[sep.reshape(letter_img.shape) > 0, :] = 0.5
        order = CutWeightOrder(adjacency)(sep)
        fig.suptitle(f"The letter's graph and a cut of size {order}")
    else:
        fig.suptitle("The letter's graph")

    def draw_image_zoom_picker(ax, sel_xy):
        marked_im = img_rbg.copy()
        if sel_xy is not None:
            sel_x, sel_y = sel_xy
            min_y, max_y = sel_y - zoom_half_size, sel_y + zoom_half_size
            min_x, max_x = sel_x - zoom_half_size, sel_x + zoom_half_size
            marked_im[min_y - 1, min_x - 1 : max_x + 1, :] = (1, 0, 0)
            marked_im[max_y + 1, min_x - 1 : max_x + 1, :] = (1, 0, 0)
            marked_im[min_y - 1 : max_y + 1, min_x - 1, :] = (1, 0, 0)
            marked_im[min_y - 1 : max_y + 1, max_x + 1, :] = (1, 0, 0)
        ax.axis("off")
        ax.imshow(marked_im)

    def update_zoom_img(sel_x, sel_y, ax):
        ax.clear()
        ax.axis("equal")
        ax.axis("off")

        sel_x, sel_y = int(sel_x), int(sel_y)
        min_y, max_y = sel_y - zoom_half_size, sel_y + zoom_half_size + 1
        min_x, max_x = sel_x - zoom_half_size, sel_x + zoom_half_size + 1

        G = nx.grid_2d_graph(max_y - min_y, max_x - min_x)
        for n1, n2 in G.edges():
            y1, x1 = min_y + n1[0], min_x + n1[1]
            y2, x2 = min_y + n2[0], min_x + n2[1]
            if (
                adjacency[y1 * letter_img.shape[1] + x1, y2 * letter_img.shape[1] + x2]
                == 0
            ):
                G.remove_edge(n1, n2)

        edge_colors = "k"
        edge_widths = 1
        if sep is not None:
            edge_colors = []
            edge_widths = []
            for n1, n2 in G.edges():
                y1, x1 = min_y + n1[0], min_x + n1[1]
                y2, x2 = min_y + n2[0], min_x + n2[1]
                if (
                    sep[y1 * letter_img.shape[1] + x1]
                    != sep[y2 * letter_img.shape[1] + x2]
                ):
                    edge_colors.append((1, 0, 0))
                    edge_widths.append(2)
                else:
                    edge_colors.append((0, 0, 0))
                    edge_widths.append(0.5)

        node_colors = [img_rbg[min_y + y, min_x + x] for y, x in G.nodes()]
        nx.draw_networkx(
            G,
            ax=ax,
            pos={(x, y): (y, -x) for x, y in G.nodes()},
            node_color=node_colors,
            edge_color=edge_colors,
            width=edge_widths,
            with_labels=False,
            node_size=10,
        )
        return sel_x, sel_y

    picker = XYPicker(
        axes[0],
        update_zoom_img,
        [zoom_half_size + 1, letter_img.shape[0] - zoom_half_size - 2],
        [zoom_half_size + 1, letter_img.shape[1] - zoom_half_size - 2],
        draw_custom_picker=draw_image_zoom_picker,
    )
    picker.callback_object = axes[1]

    return fig, picker


def all_scores_in_one_image(scores, imshape, max_cols=8):
    nrows = int(np.ceil(scores.shape[1] / max_cols))
    ncols = int(np.ceil(scores.shape[1] / nrows))
    im = [
        np.hstack(
            [
                (
                    scores[:, idx].reshape(imshape)
                    if (idx := i * ncols + j) < scores.shape[1]
                    else np.zeros(imshape)
                )
                for j in range(ncols)
            ]
        )
        for i in range(nrows)
    ]
    return np.vstack(im)


def plot_image_seps(S, image, nrows=None, ncols=None):
    if ncols is None:
        if nrows is None:
            nrows = int(max(1, np.floor(np.sqrt(S.shape[1]))))
        ncols = int(np.ceil(S.shape[1] / nrows))
    elif nrows is None:
        nrows = int(np.ceil(S.shape[1] / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(min(ncols * 1.5, 15), min(nrows * 1.5, 15))
    )
    for i, ax in enumerate(axes.flat):
        if i < S.shape[1]:
            ax.imshow(S[:, i].reshape(image.shape))
            ax.axis("off")
        else:
            ax.set_visible(False)
    fig.tight_layout()
