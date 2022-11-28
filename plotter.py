from sys import platlibdir
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
import torch


def plot_acc_data_distribution(acc_dists, save_path, label):
    total_dists = acc_dists.shape[0]
    colors = sn.color_palette("flare", as_cmap=True)

    norm = mpl.colors.Normalize(acc_dists.min(), acc_dists.max())
    sm = cm.ScalarMappable(cmap=colors, norm=norm)
    sm.set_array([])
    fig = plt.figure(figsize=(10, 10))

    for idx, acc_dist in enumerate(acc_dists):
        ax = fig.add_subplot(total_dists + 1, 1, idx + 1)
        if idx == 0:
            ax.set_title(label)
        ax.set_frame_on(False)
        for idx, val in enumerate(acc_dist):
            # print(idx, row.value, row.colors)
            if idx == acc_dist.shape[0] - 1:
                break
            line = sn.lineplot(
                x=[idx, idx + 1], y=[0, 0], color=colors(norm(val)), ax=ax,linewidth=15
            )
            line.tick_params(left=False)
            line.tick_params(bottom=False)
            line.set(xlabel=None)
            # line.set(ylabel=labels[idx])
            line.set(ylabel=None)
            line.set(xticklabels=[])
            line.set(yticklabels=[])

    ax = fig.add_subplot(total_dists + 1, 1, total_dists + 1)
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False)
    ax.set(xlabel=None, ylabel=None, xticklabels=[],yticklabels=[])
    ax.figure.colorbar(
        sm,
        location="bottom",
        ticks=[acc_dists.min(), acc_dists.max()],
        drawedges=False,
        fraction=0.5,
        shrink=0.7,
        aspect=60,
    )
    # sn.despine(left=True)
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    a = np.linspace(0, 1, 100) * 0.7
    b = np.linspace(1, 0, 50) * 0.4 + 0.1
    data = np.concatenate((a, b))
    data = torch.from_numpy(data).unsqueeze(0).expand(20, -1)
    plot_acc_data_distribution(data, save_path="test.png", label="TEst")
