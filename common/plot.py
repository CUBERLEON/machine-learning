from typing import List

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

from common.config import COLOR_PALETTE


def get_colors(count: int):
    assert count <= len(COLOR_PALETTE)
    return COLOR_PALETTE[:count]


def plot_pca_2d(x, y, *,
                ax: Axes = None,
                title: str = '2 Component PCA',
                class_names: List = None,
                point_size: int = 4,
                model=None,
                heatmap_detail_level: int = 500,
                heatmap_padding: float = 0.2,
                verbose: bool = False):

    ax_ = ax if ax else plt.subplots(nrows=1, ncols=1)[1]

    targets = np.unique(y)
    targets_cnt = len(targets)

    colors = get_colors(targets_cnt) / 255

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)

    if verbose:
        print(f'PCA sum of variance ratios: {pca.explained_variance_ratio_.cumsum()}')

    principal_df = pd.DataFrame(data=np.hstack((principal_components, y.reshape((len(y), 1)))),
                                columns=['principal component 1', 'principal component 2', 'target'])

    ax_.set_title(title)

    if model is not None:
        min_x, min_y = np.min(principal_components, axis=0) - heatmap_padding
        max_x, max_y = np.max(principal_components, axis=0) + heatmap_padding

        xx = np.linspace(min_x, max_x, heatmap_detail_level)
        yy = np.linspace(max_y, min_y, heatmap_detail_level)
        XX, YY = np.meshgrid(xx, yy)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T

        x_grid = pca.inverse_transform(xy)

        y_grid = model.predict_proba(x_grid)
        y_grid /= np.max(y_grid)
        y_grid = np.dot(y_grid, colors)
        y_grid = y_grid.reshape((heatmap_detail_level, heatmap_detail_level, 3))

        ax_.imshow(y_grid / 1.5, interpolation='nearest', extent=[min_x, max_x, min_y, max_y])

    for target, color in zip(targets, colors):
        indices_to_keep = principal_df['target'] == target
        ax_.scatter(principal_df.loc[indices_to_keep, 'principal component 1'],
                    principal_df.loc[indices_to_keep, 'principal component 2'],
                    c=[color], s=point_size)
    ax_.legend(targets if class_names is None else class_names)

    if not ax:
        plt.show()


def plot_tsne_2d(x, y, *,
                 ax: Axes = None,
                 title: str = '2 Component TSNE',
                 class_names: List = None,
                 point_size: int = 4,
                 verbose: bool = False):

    ax_ = ax if ax else plt.subplots(nrows=1, ncols=1)[1]

    targets = np.unique(y)
    targets_cnt = len(targets)

    colors = get_colors(targets_cnt) / 255

    tsne = TSNE(n_components=2, verbose=verbose)
    components = tsne.fit_transform(x)

    tsne_df = pd.DataFrame(data=np.hstack((components, y.reshape((len(y), 1)))),
                                columns=['component 1', 'component 2', 'target'])

    ax_.set_title(title)

    for target, color in zip(targets, colors):
        indices_to_keep = tsne_df['target'] == target
        ax_.scatter(tsne_df.loc[indices_to_keep, 'component 1'],
                    tsne_df.loc[indices_to_keep, 'component 2'],
                    c=[color], s=point_size)
    ax_.legend(targets if class_names is None else class_names)

    if not ax:
        plt.show()


def plot_roc(ax, y, y_score, model_name):
    fpr, tpr, thresholds = roc_curve(y, y_score)
    roc_auc = roc_auc_score(y, y_score)

    ax.plot(fpr, tpr, label=f'{model_name} (area={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    ax.grid()


def plot_curves(curves: List[np.ndarray], title=None, xlabel=None, ylabel=None, legend=None):
    for curve in curves:
        plt.plot(curve)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(legend, loc='upper left')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def plot_cnn_layer(model, *, test_img, layer, title=None, cmap='gray', normalize: bool = False):
    functor = keras.backend.function([model.input, keras.backend.learning_phase()], [layer.output])

    values = functor([test_img.reshape(1, *test_img.shape), 0.])[0][0]
    assert len(values.shape) == 3

    k = int(np.ceil(np.sqrt(values.shape[2])))
    h, w = values.shape[:2]
    map = np.ones(((h + 1) * k - 1, (w + 1) * k - 1))

    for i in range(values.shape[2]):
        x_i = i % k
        y_i = i // k
        if normalize:
            values[:, :, i] /= np.max(values[:, :, i]) + 1e-10

        map[y_i * (h + 1): (y_i + 1) * (h + 1) - 1, x_i * (w + 1): (x_i + 1) * (w + 1) - 1] = values[:, :, i]

    map *= 255
    np.clip(map, 0, 255)

    plt.imshow(map.astype(np.uint8), cmap=cmap)

    title = '' if title is None else title
    plt.title(title + f"\nshape={values.shape}")
    plt.colorbar()
    plt.show()
