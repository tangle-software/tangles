import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors


def load_big_five_data():
    # metadata
    metadata_df = pd.read_csv("data/big5.txt", sep="\t", header=None, index_col=0)
    metadata_df.columns = ["statement"]
    names = metadata_df.index.to_numpy()
    labels = metadata_df["statement"].to_numpy()
    factors = ["E", "N", "A", "C", "O"]

    # ratings
    raw_data = pd.read_csv("data/big5.csv", sep="\t")
    raw_data = remove_rows_containing_zero(raw_data)
    X = raw_data.iloc[:, 7:].to_numpy()
    return X, names, labels, factors


def remove_rows_containing_zero(df: pd.DataFrame):
    return df[-(df == 0).any(axis=1)]


def remove_rows_with_zero_std(X: np.ndarray):
    return X[np.std(X, axis=1) != 0]


def standardize_ratings(X: np.ndarray):
    # for each row: center at mean and scale by std
    X = X - np.mean(X, axis=1)[:, np.newaxis]
    row_std = np.std(X, axis=1)
    non_zero_std_rows = row_std != 0
    X[non_zero_std_rows, :] /= row_std[non_zero_std_rows, np.newaxis]
    return X


keying_E = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
keying_N = np.array([1, -1, 1, -1, 1, 1, 1, 1, 1, 1])
keying_A = np.array([-1, 1, -1, 1, -1, 1, -1, 1, 1, 1])
keying_C = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, 1])
keying_O = np.array([1, -1, 1, -1, 1, -1, 1, 1, 1, 1])
keying = np.concatenate((keying_E, keying_N, keying_A, keying_C, keying_O))


# --------------
# -- Plotting --
# --------------

# We create a custom color map that goes blue -> grey -> yellow.
# These colors will corrspond to the orientations -1, 0, 1, respectively.
colors = [
    mcolors.TABLEAU_COLORS["tab:blue"],
    (0.2, 0.2, 0.2),
    (247 / 256, 223 / 256, 106 / 256),
]
cmap = LinearSegmentedColormap.from_list("blue_grey_orange", colors, N=100)


def styled_bar_plot(
    data: np.ndarray,
    title: str,
    xticks: list[str] | np.ndarray,
    figsize: tuple = (12, 2.5),
):
    _, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(data)), data)
    ax.set_xticks(range(len(xticks)), xticks, rotation=90), ax.set_title(title)


def plot_tangle_matrix(
    tangle_matrix,
    title="Tangle matrix",
    feature_names=None,
    only_maximal_tangles=False,
    ax=None,
):
    if only_maximal_tangles:
        tangle_matrix = tangle_matrix[tangle_matrix[:, -1] != 0]
    if not ax:
        fig, ax = plt.subplots()

    ax.imshow(tangle_matrix, cmap=cmap)
    ax.set_title(title)
    if feature_names is not None:
        ax.set_xticks(
            range(tangle_matrix.shape[1]),
            feature_names[: tangle_matrix.shape[1]],
            rotation=90,
        )
    return ax


def plot_big5_tangle_matrix(
    tangle_matrix, agreement, feature_names, figsize=(10, 5), only_maximal_tangles=True
):
    fix, ax = plt.subplots(1, 1, figsize=figsize)
    ax = plot_tangle_matrix(
        tangle_matrix,
        f"Tangle Matrix (agreement={agreement})",
        feature_names,
        only_maximal_tangles,
        ax,
    )
    # ax.vlines(np.arange(start=9.5, stop=tangle_matrix.shape[1]-0.5, step=10), ymin=0, ymax=0, color='w', linewidth=2, transform=ax.get_xaxis_transform())
    for coord in np.arange(start=9.5, stop=tangle_matrix.shape[1] - 0.5, step=10):
        ax.axvline(coord, color="w", linewidth=2)


def plot_mindsets(mindsets: np.ndarray, title: str):
    num_mindsets = mindsets.shape[0]
    num_factors = mindsets.shape[1]

    plt.figure(figsize=(4.5, 3))
    plt.imshow(np.zeros(mindsets.shape), cmap="binary", aspect="auto")
    plt.title(title)
    factors_long = [
        "extroverted",
        "neurotic",
        "agreeable",
        "conscientious",
        "open for exp.",
    ]
    plt.xticks(range(num_factors), factors_long, rotation=60)
    plt.yticks([])

    for row_idx in range(num_mindsets):
        for col_idx in range(num_factors):
            yes_or_no_tick = "✓" if mindsets[row_idx, col_idx] == 1 else "×"
            color = "green" if mindsets[row_idx, col_idx] == 1 else "red"
            plt.text(
                col_idx,
                row_idx,
                yes_or_no_tick,
                size=22,
                color=color,
                weight="bold",
                horizontalalignment="center",
                verticalalignment="center",
            )
    plt.hlines(
        np.arange(num_mindsets) + 0.5,
        xmin=-0.5,
        xmax=num_factors - 0.5,
        colors="black",
        linewidth=0.5,
    )
