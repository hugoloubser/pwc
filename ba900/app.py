"""
Streamlit application for exploring BA900 NPL ratios and macroeconomic
indicators.

This app provides a minimal yet functional interface to fulfil the
assessment requirements: users can select banks, view their NPL ratios
over time, examine relationships with macroeconomic variables and
inspect simple model predictions.  The app leverages the plotting
utilities in ``ba900.visualization`` and the modelling helpers in
``ba900.modeling``.

Usage
-----
To run the app locally, install the dependencies in ``requirements.txt``
and execute::

    streamlit run -m ba900.app

Ensure that the data scraping steps have been performed and that the
processed data files (CSV or pickled DataFrames) are accessible via
the paths configured below.  The app will display helpful messages
when data files are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import pandas as pd

from .visualization import (
    plot_npl_over_time,
    plot_npl_vs_macro,
    plot_feature_importance,
)
from .modeling import prepare_regression_dataset, train_simple_model


@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    """Load a DataFrame from a CSV or pickle file, with caching.

    Parameters
    ----------
    path : pathlib.Path
        File path.  Recognised extensions are .csv and .pkl.  Other
        extensions raise ``ValueError``.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=True, index_col=0)
    elif path.suffix == ".pkl":
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def main() -> None:
    st.set_page_config(page_title="BA900 NPL Dashboard", layout="wide")
    st.title("BA900 Non‑performing Loan Dashboard")

    st.markdown(
        """
        This application allows you to explore South African banks' non‑performing loan (NPL) ratios
        alongside macroeconomic indicators.  Use the sidebar to configure the
        analysis and view interactive charts.  The underlying data are
        scraped from the SARB BA900 returns and the World Bank API.  For
        details on how the data were collected, refer to the project
        documentation.
        """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")
    data_dir = st.sidebar.text_input(
        "Data directory", value=str(Path.cwd() / "data"), help="Directory containing processed data files."
    )
    bank_file = st.sidebar.text_input(
        "Bank panel file", value="bank_panel.pkl", help="Pickle or CSV file containing bank panel data."
    )
    macro_file = st.sidebar.text_input(
        "Macro file", value="macro.pkl", help="Pickle or CSV file containing macroeconomic indicators."
    )
    show_model = st.sidebar.checkbox("Show model results", value=True)

    # Load data
    try:
        bank_panel = load_dataset(Path(data_dir) / bank_file)
        macro_df = load_dataset(Path(data_dir) / macro_file)
    except Exception as exc:
        st.error(f"Error loading data: {exc}")
        st.stop()

    # Ensure index is correct type
    if not isinstance(bank_panel.index, pd.MultiIndex):
        st.error("Bank panel data must have a MultiIndex (institution, date).")
        st.stop()

    # Sidebar: bank selection
    institutions = bank_panel.index.get_level_values(0).unique().tolist()
    selected_insts = st.sidebar.multiselect(
        "Institutions", options=institutions, default=institutions[:5]
    )

    # Sidebar: macro variable selection
    macro_vars = [col for col in macro_df.columns if col != "date"]
    selected_macro = st.sidebar.selectbox(
        "Macro variable", options=macro_vars, index=0
    )

    # Main content: plots
    if selected_insts:
        st.subheader("NPL ratios over time")
        fig1 = plot_npl_over_time(bank_panel, institutions=selected_insts)
        st.pyplot(fig1)
    else:
        st.info("Select at least one institution to view NPL ratios over time.")

    # Prepare merged dataset for scatter plot and modelling
    merged = prepare_regression_dataset(bank_panel, macro_df, date_freq="A")

    # Scatter plot of NPL vs macro variable
    st.subheader(f"NPL ratio vs. {selected_macro}")
    fig2 = plot_npl_vs_macro(merged, macro_var=selected_macro, hue="institution")
    st.pyplot(fig2)

    if show_model:
        st.subheader("Simple model")
        # Train a linear regression on the merged dataset
        model, metrics = train_simple_model(
            merged, target="npl_ratio", model_type="linear_regression"
        )
        st.write(
            f"Model performance (test set): RMSE = {metrics['rmse']:.4f}, R² = {metrics['r2']:.4f}"
        )
        # Feature importance for linear regression is not available; we
        # can display coefficients instead
        coef_df = pd.DataFrame(
            {
                "feature": [c for c in merged.columns if c not in {"npl_ratio", "institution"}],
                "coefficient": model.coef_,
            }
        ).sort_values(by="coefficient", key=lambda s: s.abs(), ascending=False)
        st.write("Model coefficients (sorted by magnitude):")
        st.dataframe(coef_df)

        # Optionally switch to a decision tree for feature importance
        if st.checkbox("Train decision tree instead"):
            tree_model, tree_metrics = train_simple_model(
                merged, target="npl_ratio", model_type="decision_tree", max_depth=3
            )
            st.write(
                f"Decision tree performance: RMSE = {tree_metrics['rmse']:.4f}, "
                f"R² = {tree_metrics['r2']:.4f}"
            )
            # Plot feature importances
            fig3 = plot_feature_importance(tree_model, feature_names=coef_df["feature"].tolist())
            st.pyplot(fig3)


if __name__ == "__main__":
    main()
