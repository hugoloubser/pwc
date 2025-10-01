"""
Module for retrieving macroeconomic indicators needed for the PwC BA900
technical assessment.  The task specification asks for macro variables such
as interest rates, inflation, unemployment, household debt to disposable
income and GDP growth.  Several different data providers expose these
indicators; this module implements helper functions for two sources:

1. **World Bank API** – using ``pandas_datareader``.  The World Bank
   publishes a vast catalogue of development indicators.  Indicators are
   identified by codes such as ``FP.CPI.TOTL.ZG`` for consumer price
   inflation or ``NY.GDP.MKTP.KD.ZG`` for GDP growth.  Data are returned
   on an annual basis and cover many countries, including South Africa
   (country code ``ZAF``).  ``pandas_datareader`` wraps the World Bank
   service and returns tidy data frames.

2. **South African Reserve Bank (SARB) web API** – using ``requests``.
   The SARB web API also exposes economic time series used on the SARB
   website.  In particular, the endpoint
   ``WebIndicators/Shared/GetTimeseriesObservations/{timeseriesCode}/{startDate}/{endDate}``
   returns observations for a single time series between two dates.  The
   series codes are the same as those used on the SARB website (e.g.
   ``MMRD002A`` for the repurchase rate).  The API returns JSON where
   each record includes a timestamp, description and numeric value【261692235555049†L7-L31】.

The functions in this module are designed to be flexible.  Users can
request multiple World Bank indicators at once or fetch a single SARB
series.  Each function performs rudimentary error handling and can be
used in conjunction with caching logic in higher‑level modules.

This module requires the following third‑party libraries which are
declared in ``requirements.txt``:

* ``pandas`` – for working with data frames.
* ``requests`` – for issuing HTTP requests to the SARB API.
* ``pandas_datareader`` – for retrieving data from the World Bank.

Note: network access may not always be available in the execution
environment.  When fetching data from remote endpoints fails, the
functions will raise a ``RuntimeError`` with a descriptive message so
callers can decide how to proceed (e.g. load cached data instead).

References
----------
* SARB Web API documentation for ``GetTimeseriesObservations``: the
  endpoint requires a time series code and date range and returns a
  collection of observations with fields ``Period``, ``Description``
  and ``Value``【261692235555049†L7-L31】.  Sample output is shown in the
  API help page.
* World Bank API documentation (via ``pandas_datareader``) for
  retrieving indicator data.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Dict, Optional
from datetime import datetime

import pandas as pd
import requests

try:
    # ``pandas_datareader`` is optional but recommended.  If it isn't
    # installed (e.g. in constrained environments), import errors will
    # propagate so users know to adjust ``requirements.txt``.
    from pandas_datareader import wb  # type: ignore
    _WB_AVAILABLE = True
except ImportError:
    _WB_AVAILABLE = False

__all__ = [
    "get_world_bank_indicators",
    "get_sarb_timeseries",
]


def get_world_bank_indicators(
    indicators: Dict[str, str],
    country_code: str = "ZAF",
    start_year: int = 2000,
    end_year: int = datetime.now().year,
) -> pd.DataFrame:
    """Fetch multiple macroeconomic indicators from the World Bank API.

    Parameters
    ----------
    indicators : dict
        Mapping of human readable names to World Bank indicator codes.  For
        example ``{"gdp_growth": "NY.GDP.MKTP.KD.ZG", "inflation": "FP.CPI.TOTL.ZG"}``.
    country_code : str, optional
        ISO 3‑letter country code to retrieve data for, by default
        ``"ZAF"`` (South Africa).
    start_year : int, optional
        First year of data to request, default is 2000.
    end_year : int, optional
        Final year of data to request, default is the current year.

    Returns
    -------
    pd.DataFrame
        A data frame indexed by year with one column per indicator.  Years
        with no available data are dropped.  The columns are named
        according to the keys of the ``indicators`` dictionary.

    Raises
    ------
    RuntimeError
        If ``pandas_datareader`` is unavailable or the data request
        fails.
    """
    if not _WB_AVAILABLE:
        raise RuntimeError(
            "pandas_datareader is required for World Bank data retrieval. "
            "Please install it via pip and ensure it is listed in requirements.txt."
        )

    # The World Bank API returns annual observations.  We request
    # everything at once for efficiency.
    codes = list(indicators.values())
    try:
        data = wb.download(
            indicator=codes, country=country_code, start=start_year, end=end_year
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download World Bank data: {exc}")

    # ``wb.download`` returns a multi‑indexed DataFrame with country and
    # year.  Flatten to a DataFrame indexed by year.
    if data.empty:
        raise RuntimeError(
            f"World Bank returned no data for country {country_code} over "
            f"{start_year}-{end_year}.  Please check the indicator codes and date range."
        )

    # Drop the country level (always the same) and pivot indicators into columns.
    data = data.reset_index().rename(columns={"year": "Year"})
    # Keep only columns of interest and pivot to wide format
    pivoted = data.pivot(index="Year", columns="indicator", values="value")
    # Rename columns to human friendly names
    pivoted = pivoted.rename(columns={v: k for k, v in indicators.items()})
    # Drop years where all values are missing
    pivoted = pivoted.dropna(how="all")
    return pivoted


def get_sarb_timeseries(
    timeseries_code: str,
    start_date: str,
    end_date: str,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Retrieve a single time series from the SARB Web API.

    The SARB endpoint returns a collection of observations with fields
    ``Period``, ``Description`` and ``Value``【261692235555049†L7-L31】.  This helper
    wraps the API call and converts the result into a ``pandas`` data
    frame with ``datetime`` index and a single numeric column named
    ``"value"``.

    Parameters
    ----------
    timeseries_code : str
        The SARB time series code (e.g. ``"MMRD002A"`` for the repurchase
        rate).  Codes can be obtained from the SARB website or by calling
        one of the ``ReleaseOfSelectedData`` or ``GetCategoryInformation``
        endpoints.
    start_date : str
        Start date in ISO ``YYYY-MM-DD`` format.
    end_date : str
        End date in ISO ``YYYY-MM-DD`` format.
    session : requests.Session, optional
        A requests session to reuse TCP connections.  Pass ``None`` to
        create a temporary session.

    Returns
    -------
    pd.DataFrame
        Data frame indexed by ``pd.DatetimeIndex`` with a ``value``
        column.  The ``description`` is attached as a column if
        available.

    Raises
    ------
    RuntimeError
        If the HTTP request fails or returns a non‑JSON response.
    """
    base_url = "https://custom.resbank.co.za/SarbWebApi/WebIndicators/Shared"
    url = f"{base_url}/GetTimeseriesObservations/{timeseries_code}/{start_date}/{end_date}"
    sess = session or requests.Session()
    try:
        resp = sess.get(url, timeout=30)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch SARB time series: {exc}") from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"SARB API returned status {resp.status_code} for {timeseries_code}. "
            f"Check the code and date range."
        )
    # Attempt to parse JSON; some series may return plain text if the
    # content type is wrong.  We'll handle both gracefully.
    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise RuntimeError(
            f"SARB API response could not be decoded as JSON. Raw text: {resp.text[:200]}"
        )
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response type: {type(data)}")
    if not data:
        raise RuntimeError(
            f"No observations returned for series {timeseries_code} between {start_date} and {end_date}."
        )
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Expect columns: Period, Description, Value, FormatNumber, FormatDate
    # Convert Period to datetime and Value to numeric
    df["Period"] = pd.to_datetime(df["Period"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    # Rename columns to lower case for consistency
    df = df.rename(columns={"Period": "date", "Value": "value", "Description": "description"})
    df = df.set_index("date").sort_index()
    return df[["value", "description"]] if "description" in df.columns else df[["value"]]
