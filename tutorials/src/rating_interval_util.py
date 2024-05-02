import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# We create a custom color map that goes grey -> red -> yellow -> green.
# The last 5 colors will be the colors in the plot for rating 1 to 5.
colors = [
    (0.2, 0.2, 0.2),
    (0.2, 0.2, 0.2),
    (0.2, 0.2, 0.2),
    (0.2, 0.2, 0.2),
    (0.7, 0, 0),
    (0.9, 0.4, 0),
    (0.95, 0.9, 0),
    (0.5, 0.9, 0),
    (0, 0.6, 0),
]
cmap = LinearSegmentedColormap.from_list("black_red_green", colors, N=100)


def plot_typical_answers(typical_answers, agreement, statement_labels, figwidth=12):
    # Assemble matrix m. m will look like intervals when plotted with imshow below.
    interval_max = 5
    m = np.zeros((typical_answers.shape[0], typical_answers.shape[1] * interval_max))
    for tangle_idx in range(typical_answers.shape[0]):
        for i in range(typical_answers.shape[1]):
            base_idx = i * interval_max
            answers = np.asarray(typical_answers[statement_labels[i]].iloc[tangle_idx])
            m[tangle_idx, :][answers + base_idx - 1] = 1

    # plotting logic
    color_gradient = np.tile(
        [0.5, 0.625, 0.75, 0.875, 1.0], reps=(1, typical_answers.shape[1])
    )
    m_with_gradients = m * color_gradient
    relative_height = figwidth / 15
    _, ax = plt.subplots(figsize=(figwidth, m.shape[0] * (relative_height * 0.55)))
    ax.imshow(m_with_gradients, aspect="auto", interpolation="none", cmap=cmap)
    ax.hlines(
        np.arange(m.shape[0] + 1) - 0.5,
        xmin=-0.5,
        xmax=m.shape[1] - 0.5,
        color="black",
        linewidth=relative_height * 21,
    )
    ax.hlines(
        np.arange(m.shape[0] + 1) - 0.5,
        xmin=-0.5,
        xmax=m.shape[1] - 0.5,
        color="gray",
        linewidth=1,
    )
    ax.vlines(
        np.arange(start=-0.5, stop=m.shape[1] - 1, step=interval_max),
        ymin=-0.5,
        ymax=m.shape[0] - 0.5,
        color="gray",
    )
    # ax.vlines(np.arange(start=-0.5, stop=m.shape[1], step=max_interval_len*10), ymin=-0.5, ymax=m.shape[0]-0.5, color='white', linewidth=2)
    ax.set_xticks(
        np.arange(start=2, stop=m.shape[1], step=interval_max), labels=statement_labels
    )
    ax.xaxis.tick_top()
    # ax.get_yaxis().set_visible(False)
    ax.set_yticks(range(m_with_gradients.shape[0]), range(m_with_gradients.shape[0]))
    ax.set_title(
        f"Interval Tangles (#tangles={m.shape[0]}, agreement={agreement})", pad=38
    )
