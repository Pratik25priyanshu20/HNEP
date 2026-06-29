"""Activation Atlas — 2-D projection of the quantum representation.

Reveals the structure the quantum branch is actually carving out of input
space, coloured by target or any molecule-level scalar. Useful for spotting
collapsed (1-D) representations, target-aligned clusters, or quantum outputs
that ignore the target entirely.

The projection uses UMAP when available, then falls back to t-SNE, then PCA
— so the function works in any environment without forcing heavy optional
dependencies.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _project_2d(X: np.ndarray, method: str = "auto",
                seed: int = 0) -> tuple[np.ndarray, str]:
    """Project X to 2-D. Returns (embedding, method_used)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] == 1:
        # 1-D representation — pad with zeros so we can plot it as a line.
        return np.column_stack([X[:, 0], np.zeros(X.shape[0])]), "raw-1d"
    if X.shape[1] == 2:
        return X.copy(), "raw-2d"

    chosen = method
    if chosen == "auto":
        for candidate in ("umap", "tsne", "pca"):
            try:
                return _project_2d(X, method=candidate, seed=seed)
            except Exception:
                continue
        raise RuntimeError("No projection backend available")

    if chosen == "umap":
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=seed,
                            n_neighbors=min(15, max(2, X.shape[0] - 1)))
        return reducer.fit_transform(X), "umap"

    if chosen == "tsne":
        from sklearn.manifold import TSNE
        perplexity = float(min(30, max(5, X.shape[0] // 5)))
        reducer = TSNE(n_components=2, random_state=seed,
                       perplexity=perplexity, init="pca")
        return reducer.fit_transform(X), "tsne"

    if chosen == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X), "pca"

    raise ValueError(f"Unknown projection method: {method!r}")


def plot_activation_atlas(
    quantum_outputs: np.ndarray,
    color_by: np.ndarray | None = None,
    color_label: str = "target",
    method: str = "auto",
    seed: int = 0,
    figsize: tuple[float, float] = (6.5, 5.5),
    title: str = "Activation Atlas — quantum representation",
    cmap: str = "viridis",
    s: float = 18.0,
    alpha: float = 0.8,
):
    """Plot a 2-D projection of the quantum branch's per-sample outputs.

    Parameters
    ----------
    quantum_outputs
        Array of shape (N, q_dim) — the raw quantum outputs to project.
    color_by
        Optional per-sample scalar to colour points by (e.g. the regression
        target). If ``None``, points are coloured uniformly.
    color_label
        Label for the colour bar.
    method
        One of ``"auto"``, ``"umap"``, ``"tsne"``, ``"pca"``. ``"auto"`` tries
        UMAP first, then t-SNE, then PCA.
    seed
        Random seed for the projection.
    figsize
        Matplotlib figure size.
    title
        Figure title.
    cmap, s, alpha
        Standard matplotlib scatter styling controls.
    """
    Q = np.asarray(quantum_outputs)
    if Q.ndim != 2:
        raise ValueError(
            f"quantum_outputs must be 2-D (N, q_dim); got shape {Q.shape}"
        )

    embedding, used = _project_2d(Q, method=method, seed=seed)

    fig, ax = plt.subplots(figsize=figsize)
    if color_by is None:
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   s=s, alpha=alpha, c="#1f77b4", edgecolor="none")
    else:
        c = np.asarray(color_by, dtype=np.float64)
        if c.shape[0] != Q.shape[0]:
            raise ValueError(
                f"color_by length {c.shape[0]} does not match "
                f"quantum_outputs N={Q.shape[0]}"
            )
        sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=c, cmap=cmap, s=s, alpha=alpha, edgecolor="none")
        cb = fig.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label(color_label, fontsize=9)

    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel(f"{used}-1", fontsize=9)
    ax.set_ylabel(f"{used}-2", fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Diagnostic annotation — 1-D collapse is a known QML failure mode.
    if used == "raw-1d":
        ax.text(0.02, 0.95,
                "⚠ quantum output is 1-D — representation may have collapsed",
                transform=ax.transAxes, fontsize=9, color="#d62728",
                va="top")

    fig.tight_layout()
    return fig


def plot_activation_atlas_grid(
    quantum_outputs_by_dataset: dict,
    targets_by_dataset: dict | None = None,
    method: str = "auto",
    seed: int = 0,
    cmap: str = "viridis",
    title: str = "Activation Atlas across datasets",
) -> "plt.Figure":
    """Side-by-side atlases for multiple datasets.

    Parameters
    ----------
    quantum_outputs_by_dataset
        Mapping ``dataset_name -> quantum_output_array``.
    targets_by_dataset
        Optional mapping ``dataset_name -> per-sample target array``.
    """
    if not quantum_outputs_by_dataset:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    names: Sequence[str] = list(quantum_outputs_by_dataset.keys())
    n = len(names)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(4.0 * cols, 3.6 * rows),
                             squeeze=False)

    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        Q = np.asarray(quantum_outputs_by_dataset[name])
        embedding, used = _project_2d(Q, method=method, seed=seed)
        y = (None if targets_by_dataset is None
             else targets_by_dataset.get(name))

        if y is None:
            ax.scatter(embedding[:, 0], embedding[:, 1],
                       s=14, alpha=0.75, c="#1f77b4", edgecolor="none")
        else:
            sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                            c=np.asarray(y), cmap=cmap, s=14, alpha=0.75,
                            edgecolor="none")
            fig.colorbar(sc, ax=ax, shrink=0.75)

        ax.set_title(f"{name}  ({used})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide trailing empty axes
    for idx in range(len(names), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_axis_off()

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig
