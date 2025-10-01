"""
Machine learning utilities for the PwC BA900 project.

This module provides functions to derive target variables from the BA900
data, merge them with macroeconomic features and fit simple predictive
models.  The goal of the modelling stage in the assessment is not to
achieve state‑of‑the‑art accuracy, but rather to demonstrate a
structured workflow: feature engineering, train/test splitting,
model fitting and evaluation.

The key functions include:

* ``compute_npl_ratio`` – given a DataFrame representing an
  institution's BA900 submission, compute the non‑performing loan
  (NPL) ratio as the proportion of non‑performing loans to total
  gross loans.  The BA900 returns encode these fields in XML.
* ``aggregate_bank_data`` – combine multiple periods/institutions into
  a single panel DataFrame with a multi‑index of (institution,
  date) and compute the NPL ratio for each.
* ``prepare_regression_dataset`` – merge the bank panel data with
  macroeconomic indicators (annual or monthly) by aligning dates.
* ``train_simple_model`` – fit a linear regression or decision tree
  model using scikit‑learn and return evaluation metrics on a hold‑out
  set.

These functions are intentionally high‑level and assume that the
caller has already scraped the BA900 data using ``ba900.scraper`` and
retrieved macro variables using ``ba900.macro_fetcher``.

References
----------
* PwC assessment instructions: the modelling task should use simple
  models such as linear regression or decision trees【645111012086579†L53-L63】.
* SARB data fields: the BA900 XML return contains variables such as
  gross loans and non‑performing loans (field names vary by
  institution).  When these are unavailable, the ratio cannot be
  computed.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

__all__ = [
    "compute_npl_ratio",
    "aggregate_bank_data",
    "prepare_regression_dataset",
    "train_simple_model",
]


def compute_npl_ratio(
    df: pd.DataFrame,
    npl_field: str = "non_performing_loans",
    loans_field: str = "gross_loans",
) -> pd.Series:
    """Compute the non‑performing loan (NPL) ratio for an institution.

    The BA900 return contains numerous numerical fields.  To compute
    the NPL ratio we need the total amount of non‑performing loans and
    total gross loans.  The names of these fields may vary by
    institution and should be passed to this function.  Values are
    coerced to floats; missing values yield ``NaN`` ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing columns for ``npl_field`` and
        ``loans_field``.
    npl_field : str, optional
        Column name for the non‑performing loans amount.
    loans_field : str, optional
        Column name for the gross loans amount.

    Returns
    -------
    pd.Series
        Series of NPL ratios (floats) aligned to the input index.
    """
    npl = pd.to_numeric(df[npl_field], errors="coerce")
    loans = pd.to_numeric(df[loans_field], errors="coerce")
    ratio = npl / loans
    return ratio


def aggregate_bank_data(
    records: Iterable[pd.DataFrame],
    npl_field: str = "non_performing_loans",
    loans_field: str = "gross_loans",
    institution_id: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate multiple BA900 return dataframes into a panel.

    Each record in ``records`` should represent a single institution and
    reporting period.  The function concatenates the data, computes
    NPL ratios and returns a panel with columns ``npl_ratio`` and
    optionally ``institution``.  A multi‑index of (institution, date)
    facilitates merging with macro indicators.

    Parameters
    ----------
    records : iterable of pd.DataFrame
        Sequence of DataFrames returned by ``scraper.get_institution_data``.
        Each must include a ``date`` index or column and the NPL and
        gross loan fields.
    npl_field : str, optional
        Name of the column containing non‑performing loans.
    loans_field : str, optional
        Name of the column containing gross loans.
    institution_id : str, optional
        If provided, a constant identifier assigned to all rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with a multi‑index (institution, date) and columns
        including ``npl_ratio`` and the original numeric fields.
    """
    panels = []
    for df in records:
        if df.empty:
            continue
        # Ensure there is a date index; if not, attempt to infer from
        # a ``Period`` column
        if df.index.name is None or df.index.name != "date":
            if "date" in df.columns:
                df = df.set_index("date")
            else:
                raise ValueError("Record lacks a 'date' index or column.")
        # Compute ratio
        ratio = compute_npl_ratio(df, npl_field=npl_field, loans_field=loans_field)
        panel = df.copy()
        panel["npl_ratio"] = ratio
        if institution_id is not None:
            panel["institution"] = institution_id
        panels.append(panel)
    if not panels:
        raise ValueError("No non‑empty records provided for aggregation.")
    combined = pd.concat(panels)
    # If institution column exists, set a multi‑index
    if "institution" in combined.columns:
        combined = combined.set_index("institution", append=True)
        # Swap levels to (institution, date)
        combined = combined.reorder_levels(["institution", "date"]).sort_index()
    return combined


def prepare_regression_dataset(
    bank_panel: pd.DataFrame,
    macro_df: pd.DataFrame,
    date_freq: str = "A",
    how: str = "inner",
) -> pd.DataFrame:
    """Combine bank NPL ratios with macro indicators on matching dates.

    Because the BA900 returns are typically monthly while macro
    indicators may be annual, we must align frequencies.  The
    ``date_freq`` parameter controls how bank dates are resampled: use
    'A' (annual) to take year‑end values, or 'M' (monthly) to align
    monthly macro data.  After resampling, the function merges the
    bank panel with macro features on the date index.

    Parameters
    ----------
    bank_panel : pd.DataFrame
        Panel of bank observations with a multi‑index (institution,
        date) and an ``npl_ratio`` column.
    macro_df : pd.DataFrame
        DataFrame of macro indicators indexed by date (year or
        monthly).  Columns correspond to indicator names.
    date_freq : str, optional
        Resampling frequency for bank data, default 'A' (annual).
        Choose 'M' for monthly alignment.
    how : str, optional
        Merge strategy ('inner', 'left', 'right', 'outer'), default is
        'inner' to keep only overlapping periods.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame where each row corresponds to one
        institution and date, containing ``npl_ratio`` and macro
        features.  The index is a multi‑index of (institution, date).
    """
    if not isinstance(bank_panel.index, pd.MultiIndex):
        raise ValueError("bank_panel must have a MultiIndex of (institution, date)")
    # Resample bank data to the desired frequency for each institution
    def _resample(group: pd.DataFrame) -> pd.DataFrame:
        # ``npl_ratio`` can be aggregated by mean or last; we choose last
        resampled = group.resample(date_freq, level="date").last()
        return resampled
    resampled = bank_panel.groupby(level=0, group_keys=False).apply(_resample)
    # Align macro index to datetime if necessary
    if not isinstance(macro_df.index, pd.DatetimeIndex):
        try:
            macro_df.index = pd.to_datetime(macro_df.index.astype(str))
        except Exception:
            raise ValueError("macro_df index must be convertible to datetime")
    # Merge each institution's resampled data with macro features
    merged_panels = []
    for inst, grp in resampled.groupby(level=0):
        df_inst = grp.reset_index(level=0, drop=True)
        df_merged = df_inst.join(macro_df, how=how)
        df_merged["institution"] = inst
        merged_panels.append(df_merged)
    merged = pd.concat(merged_panels)
    merged = merged.set_index("institution", append=True)
    merged = merged.reorder_levels(["institution", merged.index.name])
    return merged


def train_simple_model(
    dataset: pd.DataFrame,
    target: str = "npl_ratio",
    test_size: float = 0.3,
    random_state: int = 42,
    model_type: str = "linear_regression",
    **model_params: Any,
) -> Tuple[Any, Dict[str, float]]:
    """Train a simple regression model on the prepared dataset.

    The function splits the data into train and test sets, fits a
    scikit‑learn model and computes the root mean squared error (RMSE)
    and coefficient of determination (R^2) on the held‑out set.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame containing the target column and feature columns.  Any
        rows with missing values are dropped prior to splitting.
    target : str, optional
        Name of the target column to predict, default 'npl_ratio'.
    test_size : float, optional
        Fraction of data to reserve as the test set, default 0.3.
    random_state : int, optional
        Random seed for train/test splitting.
    model_type : str, optional
        Type of model to train: either 'linear_regression' or
        'decision_tree'.
    **model_params : dict, optional
        Additional keyword arguments passed to the model constructor.

    Returns
    -------
    tuple
        A tuple of (fitted_model, metrics_dict) where metrics_dict
        contains 'rmse' and 'r2'.
    """
    # Drop rows with missing values in target or features
    df = dataset.dropna(subset=[target])
    feature_cols = [c for c in df.columns if c != target and not pd.api.types.is_object_dtype(df[c])]
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if model_type == "linear_regression":
        model = LinearRegression(**model_params)
    elif model_type == "decision_tree":
        model = DecisionTreeRegressor(**model_params)
    else:
        raise ValueError("Unsupported model_type: choose 'linear_regression' or 'decision_tree'")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    metrics = {"rmse": rmse, "r2": r2}
    return model, metrics
