"""
embed.py — t-SNE and UMAP embedding plots for Fashion-MNIST penultimate-layer
feature visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf

from config import CLASS_NAMES, RESULTS_DIR, SEED


def extract_features(model, x: np.ndarray, layer_name: str = None) -> np.ndarray:
    """Extract activations from the penultimate (or named) layer of a model.

    Args:
        model: Trained Keras model.
        x: Input images [N, H, W, C].
        layer_name: Layer whose output to extract. Defaults to the second-to-last layer.

    Returns:
        Feature matrix [N, D].
    """
    if layer_name is None:
        layer_name = model.layers[-2].name
    feature_model = tf.keras.Model(
        inputs=model.input, outputs=model.get_layer(layer_name).output
    )
    return feature_model.predict(x, verbose=0)


def plot_tsne(features: np.ndarray, labels: np.ndarray,
              save_path: str = None, perplexity: float = 30.0,
              n_components: int = 2):
    """Compute and plot t-SNE embedding of feature vectors.

    Args:
        features: Feature matrix [N, D].
        labels: Integer class labels [N].
        save_path: Path to save the figure.
        perplexity: t-SNE perplexity parameter.
        n_components: Embedding dimensionality (2 or 3).
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=n_components, random_state=SEED, perplexity=perplexity)
    embedding = tsne.fit_transform(features)
    _scatter_2d(embedding, labels, title="t-SNE Feature Embedding", save_path=save_path)


def plot_umap(features: np.ndarray, labels: np.ndarray, save_path: str = None):
    """Compute and plot UMAP embedding of feature vectors.

    Args:
        features: Feature matrix [N, D].
        labels: Integer class labels [N].
        save_path: Path to save the figure.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")

    reducer = umap.UMAP(random_state=SEED)
    embedding = reducer.fit_transform(features)
    _scatter_2d(embedding, labels, title="UMAP Feature Embedding", save_path=save_path)


def _scatter_2d(embedding: np.ndarray, labels: np.ndarray,
                title: str = "", save_path: str = None):
    """2-D scatter plot colored by class label.

    Args:
        embedding: 2-D array [N, 2].
        labels: Integer class labels [N].
        title: Plot title.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = labels == cls_idx
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[cmap(cls_idx)],
            label=cls_name,
            s=4,
            alpha=0.7,
        )
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
