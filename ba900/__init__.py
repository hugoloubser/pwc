"""
Top‑level package for the PwC BA900 analysis project.

This package exposes the key sub‑modules used throughout the notebooks
and Streamlit application.  For example:

```python
from ba900.scraper import get_periods, fetch_period_data
from ba900.macro_fetcher import fetch_worldbank_indicators
```

"""

"""Top‑level package for the PwC BA900 project.

This package bundles the following components:

* **scraper** – functions for interacting with the SARB BA900 API; see
  :mod:`ba900.scraper`.
* **macro_fetcher** – helpers to retrieve macroeconomic indicators
  from the World Bank and SARB APIs; see :mod:`ba900.macro_fetcher`.
* **modeling** – utilities for feature engineering, preparing datasets and
  training simple models; see :mod:`ba900.modeling`.
* **visualization** – plotting utilities for EDA and the Streamlit app;
  see :mod:`ba900.visualization`.
* **app** – a Streamlit application to visualise NPL ratios and macro
  variables interactively; see :mod:`ba900.app`.

Importing this package exposes the primary scraping functions for
convenience.  Other modules should be imported explicitly.
"""

from .scraper import (
    get_periods,
    get_institutions,
    get_institution_data,
    fetch_period_data,
    load_cached_data,
)

__all__ = [
    "get_periods",
    "get_institutions",
    "get_institution_data",
    "fetch_period_data",
    "load_cached_data",
]
from .macro_fetcher import fetch_worldbank_indicators
from .modeling import train_test_split_regression, train_linear_regression, train_decision_tree
from .visualization import plot_npl_vs_macro

__all__ = [
    "get_periods",
    "get_institutions",
    "get_institution_data",
    "fetch_period_data",
    "load_cached_data",
    "fetch_worldbank_indicators",
    "train_test_split_regression",
    "train_linear_regression",
    "train_decision_tree",
    "plot_npl_vs_macro",
]