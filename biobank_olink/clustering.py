from itertools import product
from typing import Optional, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from hdbscan import HDBSCAN
from matplotlib.offsetbox import AnchoredText
from numpy.typing import ArrayLike
from phate import PHATE
from scipy.stats import f_oneway, mannwhitneyu, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
    accuracy_score,
)
from umap import UMAP
from xgboost import XGBClassifier


def get_umap_embedding(data, n_components=2, metric="euclidean", n_neighbors=15, seed=42, **kw):
    reducer = UMAP(
        n_neighbors=n_neighbors,
        metric=metric,
        n_components=n_components,
        min_dist=0,
        random_state=seed,
        **kw,
    )
    return reducer.fit_transform(data)


def get_tsne_embedding(data, n_components=2, seed=42, **kw):
    reducer = TSNE(n_components=n_components, random_state=seed, **kw)
    return reducer.fit_transform(data)


def get_phate_embedding(data, n_components=2, seed=42, **kw):
    reducer = PHATE(n_components=n_components, verbose=0, random_state=seed, **kw)
    return reducer.fit_transform(data)


def to_test_cases(grid_spec: dict):
    return [
        {"algorithm": algorithm, **dict(zip(algorithm_kw.keys(), param_set))}
        for algorithm, algorithm_kw in grid_spec.items()
        for param_set in product(*list(algorithm_kw.values()))
    ]


def get_embedding(data: ArrayLike, reducer: str, seed=42, **reducer_kw: Mapping) -> np.ndarray:
    if reducer.lower() == "umap":
        emb = get_umap_embedding(data, seed=seed, **reducer_kw)
    elif reducer.lower() == "tsne":
        emb = get_tsne_embedding(data, seed=seed, **reducer_kw)
    elif reducer.lower() == "phate":
        emb = get_phate_embedding(data, seed=seed, **reducer_kw)
    else:
        raise ValueError("Unknown algorithm, possible are: 'umap', 'tsne', 'phate'.")

    return emb


def get_cluster_labels(
    data: ArrayLike, algorithm: str = "kmeans", verbose: bool = False, seed: int = 42, **alg_kwargs
) -> pd.Series:
    if algorithm.lower() == "kmeans":
        clf = KMeans(**alg_kwargs, random_state=seed, n_init="auto")
    elif algorithm.lower() == "hdbscan":
        clf = HDBSCAN(**alg_kwargs)
    else:
        raise ValueError("Unknown algorithm, possible are: 'kmeans' and 'hdbscan'.")

    cluster_labels = clf.fit_predict(data)
    cluster_labels = pd.Series(cluster_labels, name="cluster_labels")

    if verbose:
        print(f"Silhouette score: {silhouette_score(data, cluster_labels, random_state=seed):.4f}")

    return cluster_labels


def plot_clustering(
    embedding,
    cluster_labels,
    title="UMAP clustering",
    figsize=(12, 8),
    legend_outside: bool = False,
    **kwargs,
):
    _, ax = plt.subplots(figsize=figsize)
    ax = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=cluster_labels.astype(str),
        hue_order=[str(label) for label in np.unique(cluster_labels)],
        **kwargs,
    )
    ax.set_title(title)
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(False)
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

    return ax


def score_clustering(data: np.ndarray, cluster_labels: ArrayLike, seed: int = 42):
    label_counts = np.unique(cluster_labels, return_counts=True)[1]
    scores = {}

    if data.size and label_counts.size > 1:
        scores.update(
            {
                "silhouette": silhouette_score(data, cluster_labels, random_state=seed),
                "calinski_harabasz": calinski_harabasz_score(data, cluster_labels),
                "davies_bouldin": davies_bouldin_score(data, cluster_labels),
            }
        )

    scores["label_counts"] = label_counts
    return scores


def explain_clusters(
    cluster_labels: ArrayLike,
    x: ArrayLike,
    x_orig: pd.DataFrame,
    xgb_params: Optional[Mapping] = None,
    max_display: int = 10,
    key_feature: Optional[str] = None,
    seed: int = 42,
    out_path: str = "../figures",
    out_name: Optional[str] = None,
):
    if xgb_params is None:
        xgb_params = dict(n_estimators=1, max_depth=6, tree_method="exact")

    clf = XGBClassifier(**xgb_params, eval_metric="mlogloss", random_state=seed)
    clf.fit(x, y=cluster_labels)
    y_pred = clf.predict(x)
    print(f"Explanation accuracy: {accuracy_score(cluster_labels, y_pred)}")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(x)

    unq_cluster_labels = np.unique(cluster_labels)
    if len(unq_cluster_labels) <= 2:
        shap.plots.beeswarm(shap_values, max_display=max_display, show=False)

        if out_name is not None:
            plt.savefig(f"{out_path}/{out_name}.png", bbox_inches="tight")
    else:
        for i, cluster_no in enumerate(unq_cluster_labels):
            fig_title = f"Cluster={cluster_no}"
            if key_feature is not None:
                key_feat_counts = (
                    x_orig.loc[cluster_labels == cluster_no, key_feature].value_counts().to_dict()
                )
                key_feat_info = "/".join(
                    f"{k}={key_feat_counts[k]}" for k in sorted(key_feat_counts.keys())
                )
                fig_title += f" {key_feature}=({key_feat_info})"

            fig = plt.figure()
            fig.suptitle(fig_title)

            shap.plots.beeswarm(shap_values[..., i], max_display=max_display, show=False)

            if out_name is not None:
                plt.savefig(f"{out_path}/{out_name}_{cluster_no}.png", bbox_inches="tight")

    return explainer


def visualize_clustering(
    data,
    orig_x: pd.DataFrame,
    cluster_labels: ArrayLike,
    figsize=None,
    highlight_plots=None,
    **kwargs,
):
    plot_ncols = 2
    plot_nrows = int(np.ceil(len(orig_x.columns) / plot_ncols))

    if figsize is None:
        figsize = (plot_ncols * 6, plot_nrows * 5)

    f: plt.Figure = plt.figure(figsize=figsize)
    f.subplots_adjust(wspace=1)
    f.tight_layout()

    if highlight_plots is None:
        highlight_plots = frozenset()

    if isinstance(highlight_plots, int):
        highlight_plots = [highlight_plots]

    highlight_plots = frozenset(highlight_plots)

    plot_params = dict(alpha=0.8, palette="turbo")
    plot_params.update(kwargs)

    for i, column in enumerate(orig_x.columns, 1):
        ax: plt.Axes = plt.subplot(plot_nrows, plot_ncols, i)

        ax = sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            ax=ax,
            style=cluster_labels,
            style_order=np.unique(cluster_labels),
            hue=orig_x[column],
            **plot_params,
        )
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.grid(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)

        if i in highlight_plots:
            ax.spines["top"].set_color("red")
            ax.spines["bottom"].set_color("red")
            ax.spines["left"].set_color("red")
            ax.spines["right"].set_color("red")


def eval_statistics(series_gb, stat_name):
    if stat_name.lower() == "mwu":
        stat_test = mannwhitneyu
    elif stat_name.lower() == "anova":
        stat_test = f_oneway
    elif stat_name.lower() == "chi2":
        vals = series_gb.value_counts(sort=False).unstack()
        vals[vals.isnull()] = 0

        if vals.shape[1] <= 1:
            return np.nan

        _, pval, _, _ = chi2_contingency(vals)
        return pval
    else:
        raise ValueError()
    _, pval = stat_test(*list(series.values[~series.isnull()] for _, series in series_gb))
    return pval


def print_best_features(
    cluster_labels: pd.Series, data, nrows=3, ncols=3, figsize=(15, 12), categorical_cols=None
):
    data = data.reset_index(drop=True)
    data["Cluster"] = cluster_labels

    if cluster_labels.nunique() == 2:
        stat_name = "MWU"
    else:
        stat_name = "ANOVA"

    if categorical_cols is None:
        categorical_cols = ()

    df_orig_gb = data.groupby("Cluster")
    col_pvals = [
        (
            col,
            eval_statistics(df_orig_gb[col], stat_name if col not in categorical_cols else "chi2"),
        )
        for col in data.columns
        if col != "Cluster"
    ]
    col_pvals.sort(key=lambda v: v[1])

    f, axes = plt.subplots(
        nrows,
        ncols + 1,
        figsize=figsize,
        sharey="all",
        constrained_layout=True,
        gridspec_kw=dict(width_ratios=[1] * ncols + [0.6]),
    )

    plot_axes = axes[:, :3].flat
    num_plot_axes = len(plot_axes)

    for ax, (col, pval) in zip(plot_axes, col_pvals):
        if col in categorical_cols:
            sns.countplot(data=data, hue=col, y="Cluster", ax=ax)
            ax.legend(loc="lower right")
        else:
            sns.boxplot(data=data, x=col, y="Cluster", orient="h", ax=ax)
        anc = AnchoredText(
            f"{stat_name if col not in categorical_cols else 'Chi2'}: {pval:.1E}"
            f"{'' if (stars := star_pval(pval)) == '' else f' ({stars})'}",
            loc="upper right",
            frameon=True,
            pad=0.3,
            prop=dict(size=10, fontweight="bold"),
        )
        ax.set_xlabel(normalize_col_name(col, max_length=30))
        ax.add_artist(anc)

    # displays stats for the rest of the features as text
    gs = axes[0, -1].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    axbig = f.add_subplot(gs[:, -1])
    axbig.axis("off")

    info_text = f"{stat_name} for the rest:\n"
    info_text += stat_pairs_to_summary(
        [(col, pval) for col, pval in col_pvals[num_plot_axes:50] if col not in categorical_cols]
    )

    rest_categorical_cols = [
        (col, pval) for col, pval in col_pvals[num_plot_axes:50] if col in categorical_cols
    ]
    if rest_categorical_cols:
        chi_info_text = f"\n\n\nChi2 for the rest:\n"
        info_text += chi_info_text + stat_pairs_to_summary(rest_categorical_cols)

    axbig.annotate(
        info_text, (0.1, 0.5), xycoords="axes fraction", va="center", fontfamily="monospace"
    )

    return col_pvals


def normalize_col_name(col_name, max_length=20):
    fix_length = int(max_length / 2 - 1)
    return (
        col_name
        if len(col_name) <= max_length
        else col_name[:fix_length] + ".." + col_name[-fix_length:]
    )


def stat_pairs_to_summary(stat_pairs) -> str:
    last_star = None
    summary = list()
    pad = 13
    for col, pval in stat_pairs:
        curr_star = star_pval(pval)
        if last_star is None or curr_star != last_star:
            star_pad = "-" * (pad - int(len(curr_star) > 1))
            line = star_pad + f" {curr_star} " + star_pad
            last_star = curr_star
            summary.append(line)

        line = f"{normalize_col_name(col):<20}: {pval:.1E}"
        summary.append(line)

    return "\n".join(summary)


def star_pval(pval) -> str:
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return "**"
    elif pval < 0.05:
        return "*"
    else:
        return ""
