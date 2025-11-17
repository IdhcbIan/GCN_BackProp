import bz2
import pickle
import _pickle as cPickle
from sklearn.model_selection import StratifiedKFold
from textwrap import wrap

import umap
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

sns.set_style("darkgrid")
sns.set_palette("muted")
sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2.5})


def fold_split(features, labels, n_folds=10):
    # Split in folds
    kf = StratifiedKFold(n_splits=n_folds, shuffle=False)
    res = kf.split(features, labels)
    return list(res)


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title, "w") as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)

    return data


# UMAP and Plots
def umap_flowers_resnet(features):
    results_path = os.path.join(
        "results", os.path.join("visualization_flowers+resnet", "plots")
    )
    try:
        os.makedirs(results_path)
    except FileExistsError:
        # directory already exists
        pass

    # Optimal parameters for Flowers+Resnet
    reducer = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.75)
    embedding_full = reducer.fit_transform(features)
    return embedding_full, results_path


def scatter_full(x, colors, dataset, descriptor, results_path):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    ax.set_title(
        str(dataset).upper()
        + "+"
        + str(descriptor).upper()
        + " features using UMAP embedding.",
        weight="bold",
    )
    ax.tick_params(labelbottom=False, labelleft=False)
    # ax.axis("off")
    ax.axis("tight")

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=15)
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
        txts.append(txt)

    f.savefig(
        os.path.join(
            results_path.replace("folds", ""),
            "UMAP_" + str(dataset) + "+" + str(descriptor) + ".pdf",
        ),
        bbox_inches="tight",
    )
    f.savefig(
        os.path.join(
            results_path.replace("folds", ""),
            "UMAP_" + str(dataset) + "+" + str(descriptor) + ".png",
        ),
        bbox_inches="tight",
    )
    plt.close()
    # return f, ax, sc, txts


def scatter_fold(
    option,
    correlationMeasure,
    classifier,
    fold_id,
    x,
    labeled_index,
    unlabeled_index,
    train_colors,
    colors,
    dataset,
    descriptor,
    reranking,
    results_path,
):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(train_colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # cmap = sns.color_palette("hls", num_classes).as_hex()

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")
    sc_labeled = ax.scatter(
        x[labeled_index, 0],
        x[labeled_index, 1],
        lw=0,
        s=40,
        c=palette[train_colors.astype(np.int)],
        label="Labeled Data",
    )
    sc_unlabeled = ax.scatter(
        x[unlabeled_index, 0],
        x[unlabeled_index, 1],
        lw=0,
        s=40,
        c="grey",
        alpha=0.7,
        label="Unlabeled Data",
    )
    plt.legend()
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # plt.colorbar(ax=ax, boundaries=np.arange(num_classes+1)-0.5).set_ticks(np.arange(len(colors)))

    # f.colorbar(sc_labeled, ax=ax, boundaries=np.arange(num_classes+1)-0.5).set_ticks(np.arange(len(colors)))
    if option == "initial":
        ax.set_title(
            str(dataset).upper()
            + "+"
            + str(descriptor).upper()
            + " features using UMAP embedding.",
            weight="bold",
        )
    elif option == "best_fold" and not reranking:
        ax.set_title(
            "\n".join(
                wrap(
                    "Expansion with "
                    + str(correlationMeasure).upper()
                    + " that achieved the best accuracy with "
                    + str(classifier).upper()
                    + ".",46
                )
            ),
            weight="bold",
        )
    else:  # option == "best_fold" and reranking:
        ax.set_title(
            "\n".join(
                wrap(
                    "Expansion with "
                    + str(correlationMeasure).upper()
                    + "+"
                    + str(reranking).upper()
                    + " that achieved the best accuracy with "
                    + str(classifier).upper()
                    + ".",46
                )
            ),
            weight="bold",
        )
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.axis("tight")

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=15)
        txt.set_path_effects(
            [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
        )
        txts.append(txt)

    if option == "initial":
        f.savefig(
            os.path.join(
                results_path,
                str(option)
                + "set_UMAP_"
                + "fold-"
                + str(fold_id)
                + "_"
                + str(dataset)
                + "+"
                + str(descriptor)
                + ".pdf",
            ),
            bbox_inches="tight",
        )
        f.savefig(
            os.path.join(
                results_path,
                str(option)
                + "set_UMAP_"
                + "fold-"
                + str(fold_id)
                + "_"
                + str(dataset)
                + "+"
                + str(descriptor)
                + ".png",
            ),
            bbox_inches="tight",
        )
    elif option == "best_fold" and not reranking:
        f.savefig(
            os.path.join(
                results_path,
                "UMAP_expansion_"
                + str(classifier)
                + "_"
                + "fold-"
                + str(fold_id)
                + "_"
                + str(dataset)
                + "+"
                + str(descriptor)
                + "+"
                + str(correlationMeasure)
                + ".pdf",
            ),
            bbox_inches="tight",
        )
        f.savefig(
            os.path.join(
                results_path,
                "UMAP_expansion_"
                + str(classifier)
                + "_"
                + "fold-"
                + str(fold_id)
                + "_"
                + str(dataset)
                + "+"
                + str(descriptor)
                + "+"
                + str(correlationMeasure)
                + ".png",
            ),
            bbox_inches="tight",
        )
    else:  # option == "best_fold" and reranking:
        f.savefig(
            os.path.join(
                results_path,
                "UMAP_expansion_"
                + str(reranking)
                + "_"
                + str(classifier)
                + "_"
                + "fold-"
                + str(fold_id)
                + "_"
                + str(dataset)
                + "+"
                + str(descriptor)
                + "+"
                + str(correlationMeasure)
                + ".pdf",
            ),
            bbox_inches="tight",
        )
        f.savefig(
            os.path.join(
                results_path,
                "UMAP_expansion_"
                + str(reranking)
                + "_"
                + str(classifier)
                + "_"
                + "fold-"
                + str(fold_id)
                + "_"
                + str(dataset)
                + "+"
                + str(descriptor)
                + "+"
                + str(correlationMeasure)
                + ".png",
            ),
            bbox_inches="tight",
        )

    plt.close()
    # return f, ax, txts

