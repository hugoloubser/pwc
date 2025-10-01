"""
Visualization utilities for the PwC BA900 project.

This module centralises plotting functions used in both exploratory
data analysis and in the Streamlit app.  To keep dependencies
lightweight, plots are implemented using ``matplotlib`` and
``seaborn``.  Functions return ``matplotlib`` ``Figure`` objects so
they can be embedded in notebooks or exported as images.

The key plots include:

* ``plot_npl_over_time`` – time series of NPL ratios for selected
  institutions.
* ``plot_npl_vs_macro`` – scatter plot of NPL ratios against a
  macroeconomic variable with an optional regression line.
* ``plot_feature_importance`` – bar chart of feature importances from
  tree‑based models (e.g. decision trees or random forests).

Users can extend these helpers or build additional ones as needed.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

__all__ = [
    "plot_npl_over_time",
    "plot_npl_vs_macro",
    "plot_feature_importance",
]


def plot_npl_over_time(
    panel: pd.DataFrame,
    institutions: Optional[Sequence[str]] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot the NPL ratio over time for one or more institutions.

    Parameters
    ----------
    panel : pd.DataFrame
        A multi‑indexed DataFrame with (institution, date) index and a
        ``npl_ratio`` column.
    institutions : sequence of str, optional
        Institutions to include.  If ``None``, all are plotted.
    figsize : tuple, optional
        Size of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must have a MultiIndex of (institution, date)")
    if institutions is None:
        institutions = panel.index.get_level_values(0).unique()
    fig, ax = plt.subplots(figsize=figsize)
    for inst in institutions:
        df_inst = panel.xs(inst, level=0)
        ax.plot(df_inst.index, df_inst["npl_ratio"], label=inst)
    ax.set_xlabel("Date")
    ax.set_ylabel("NPL Ratio")
    ax.set_title("Non‑performing loan ratio over time")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_npl_vs_macro(
    merged_df: pd.DataFrame,
    macro_var: str,
    hue: Optional[str] = "institution",
    add_trend: bool = True,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Scatter plot of NPL ratio against a macroeconomic variable.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Data frame returned by ``prepare_regression_dataset`` with
        columns ``npl_ratio`` and macro variables.  Must include the
        specified macro_var.
    macro_var : str
        Name of the macroeconomic variable to plot on the x‑axis.
    hue : str, optional
        Column name to use for colouring points, by default
        ``"institution"`` to distinguish banks.
    add_trend : bool, optional
        Whether to overlay a linear regression line using seaborn's
        ``regplot``.
    figsize : tuple, optional
        Size of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting figure.
    """
    if macro_var not in merged_df.columns:
        raise ValueError(f"Macro variable '{macro_var}' not found in dataset")
    fig, ax = plt.subplots(figsize=figsize)
    # Plot scatter points coloured by hue
    if hue and hue in merged_df.columns:
        sns.scatterplot(
            data=merged_df, x=macro_var, y="npl_ratio", hue=hue, ax=ax, s=60, alpha=0.8
        )
    else:
        sns.scatterplot(
            data=merged_df, x=macro_var, y="npl_ratio", ax=ax, s=60, alpha=0.8
        )
    # Optionally add a trend line using regression
    if add_trend:
        sns.regplot(
            data=merged_df, x=macro_var, y="npl_ratio", scatter=False, ax=ax,
            color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
        )
    ax.set_xlabel(macro_var)
    ax.set_ylabel("NPL Ratio")
    ax.set_title(f"NPL ratio vs. {macro_var}")
    fig.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: Sequence[str],
    top_n: int = 10,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Visualise feature importance from a fitted tree‑based model.

    Parameters
    ----------
    model : object
        Trained model with a ``feature_importances_`` attribute (e.g.
        ``DecisionTreeRegressor`` or ``RandomForestRegressor``).  If the
        attribute is missing, a ``ValueError`` is raised.
    feature_names : sequence of str
        Names corresponding to the columns used to train the model.
    top_n : int, optional
        Number of top features to display, default 10.
    figsize : tuple, optional
        Size of the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting bar chart.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have 'feature_importances_' attribute")
    importances = model.feature_importances_
    # Pair each feature with its importance and sort descending
    feats = list(zip(feature_names, importances))
    feats = sorted(feats, key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*feats) if feats else ([], [])
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=list(values), y=list(names), ax=ax, palette="viridis")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top feature importances")
    fig.tight_layout()
    return fig
