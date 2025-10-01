"""Utilities for retrieving BA900 returns from the SARB Web API.

The South African Reserve Bank exposes a public API (documented at
`https://custom.resbank.co.za/SarbWebApi/Help`) which powers the
`Banks BA900 Economic Returns` page on its website.  The relevant
endpoints used in this module are:

* `GetPeriods/{ifType}` – Returns a list of available reporting
  periods for the selected return type.  For BA900 the API returns
  strings in the form `yyyy‑mm‑dd`【773677845686591†L7-L33】.
* `GetInstitutions/{ifType}/{period}` – Returns a collection of
  institutions (banks) that filed a return for the given period.  Each
  record includes an identifier (`Id`), the bank name and the last
  update date【977277454646387†L7-L64】.
* `GetInstitutionData/{ifType}/{period}/{institutionId}` – Returns
  the full return for a single institution.  The result is a JSON
  object whose `XMLData` field contains an XML fragment representing
  the BA900 submission【14402629044933†L7-L84】.

These helpers wrap the above endpoints using `requests` and provide
convenience functions for caching responses, handling basic rate
limiting and converting XML into pandas DataFrames.

Because the API does not always respond with JSON unless the client
supplies an appropriate `Accept` header, this module explicitly
requests XML where necessary.  Downstream processing uses
`xmltodict` and `pandas` to transform the XML into a tabular format.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import xmltodict


# Base URLs for the SARB Web API
SARB_IFDATA_BASE = "https://custom.resbank.co.za/SarbWebApi/SarbData/IFData"

# Default cache directory.  Users can override this by passing a
# different path into `fetch_period_data`.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _ensure_dir(path: Path) -> None:
    """Ensure that the provided directory exists."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def get_periods(if_type: str = "BA900", *, session: Optional[requests.Session] = None) -> List[str]:
    """Return a list of valid periods for a given return type.

    Parameters
    ----------
    if_type: str
        The type of economic return (e.g. "BA900", "DI900", "BD900").
    session: Optional[requests.Session]
        An optional session object to reuse TCP connections.

    Returns
    -------
    List[str]
        A list of date strings in ISO format (``yyyy-mm-dd``) representing
        the available reporting periods.

    Raises
    ------
    requests.HTTPError
        If the underlying HTTP request fails.
    """
    url = f"{SARB_IFDATA_BASE}/GetPeriods/{if_type}"
    sess = session or requests.Session()
    resp = sess.get(url, timeout=30)
    resp.raise_for_status()
    # The API returns whitespace separated values for XML and JSON
    # representations when no Accept header is supplied.  Split on
    # whitespace to obtain the list of periods.
    text = resp.text.strip()
    if not text:
        return []
    periods = text.split()
    return periods


def get_institutions(if_type: str, period: str, *, session: Optional[requests.Session] = None) -> List[Dict[str, str]]:
    """Fetch the list of institutions (banks) for a specific period.

    Parameters
    ----------
    if_type: str
        The type of economic return (e.g. "BA900").
    period: str
        The reporting period as returned by :func:`get_periods`, e.g. ``"2025-01-01"``.
    session: Optional[requests.Session]
        Optional requests session for connection reuse.

    Returns
    -------
    List[Dict[str, str]]
        A list of dictionaries, each containing the keys ``Id``, ``Name`` and
        ``LastUpdate``.  See the API documentation for more details【977277454646387†L7-L64】.

    Notes
    -----
    The endpoint can return either JSON or XML depending on the
    ``Accept`` header.  We request XML here to avoid potential
    mis‑parsing when the API returns space separated text.
    """
    url = f"{SARB_IFDATA_BASE}/GetInstitutions/{if_type}/{period}"
    sess = session or requests.Session()
    headers = {"Accept": "application/xml"}
    resp = sess.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    # Parse the XML response.  The root element is
    # ArrayOfspGet_IFInstitutionsPerPeriodPerType_Result
    xml_obj = xmltodict.parse(resp.content)
    # Navigate to the list of institutions.  xmltodict will convert the
    # repeating element into a list automatically.  If only one
    # institution is returned it will be a dict instead of a list.
    body = xml_obj.get("ArrayOfspGet_IFInstitutionsPerPeriodPerType_Result")
    if body is None:
        return []
    records = body.get("spGet_IFInstitutionsPerPeriodPerType_Result", [])
    if isinstance(records, dict):
        records = [records]
    institutions = []
    for rec in records:
        institutions.append({
            "Id": rec.get("Id"),
            "Name": rec.get("Name"),
            "LastUpdate": rec.get("LastUpdate"),
            "NameChange": rec.get("NameChange"),
        })
    return institutions


def _flatten_record(obj: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Flatten a nested dictionary.

    This helper is used to convert the nested XMLData structure returned
    by the API into a flat dictionary that can be consumed by
    :class:`pandas.DataFrame`.  The function recursively iterates
    through nested dictionaries and lists, concatenating keys with
    underscores.
    """
    items: List[Tuple[str, any]] = []
    for k, v in obj.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_record(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # For lists of dicts, convert each dict into a flat dict and
            # store under an index key.  Non‑dict elements are stored
            # directly.
            for idx, elem in enumerate(v):
                if isinstance(elem, dict):
                    items.extend(_flatten_record(elem, f"{new_key}{sep}{idx}", sep=sep).items())
                else:
                    items.append((f"{new_key}{sep}{idx}", elem))
        else:
            items.append((new_key, v))
    return dict(items)


def get_institution_data(if_type: str, period: str, institution_id: str, *, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Download and parse the full BA900 return for one institution.

    Parameters
    ----------
    if_type: str
        Return type, e.g. ``"BA900"``.
    period: str
        Reporting period, e.g. ``"2025-01-01"``.
    institution_id: str
        Identifier for the institution as returned by :func:`get_institutions`.
    session: Optional[requests.Session]
        Optional HTTP session.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each row corresponds to a line item from the
        return.  Column names depend on the structure of the underlying
        XML.  At a minimum the DataFrame will contain the flattened
        values of the XML fragment.

    Notes
    -----
    * The BA900 return schema is complex and may change over time.
      This function attempts to create a flat tabular representation
      of the `XMLData` field by first converting the XML into a
      nested dictionary via `xmltodict` and then flattening it with
      :func:`_flatten_record`.  For analysis you may need to rename
      or select specific columns corresponding to balance sheet items.
    * The API returns a list of objects, but only the first item is
      used as each call is scoped to a single institution and period.
    * See the API documentation for field definitions【14402629044933†L46-L84】.
    """
    url = f"{SARB_IFDATA_BASE}/GetInstitutionData/{if_type}/{period}/{institution_id}"
    sess = session or requests.Session()
    headers = {"Accept": "application/json"}
    resp = sess.get(url, headers=headers, timeout=60)
    # Some endpoints are protected by upstream filtering; when accessed
    # via programmatic clients they may return HTML with an error
    # message.  Raise a descriptive error if JSON parsing fails.
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from {url}. Response starts with: {resp.text[:200]}") from exc
    if not data:
        return pd.DataFrame()
    # Each entry in the returned list contains metadata and XMLData
    entry = data[0]
    xml_string = entry.get("XMLData")
    if not xml_string:
        return pd.DataFrame([entry])
    # Parse the XML string into a Python dictionary
    xml_dict = xmltodict.parse(xml_string)
    # The root of the BA900 XML appears to be a container with a
    # repeating list of items.  Flatten the entire dict; each key will
    # correspond to a field in the return.  Depending on the exact
    # structure this may produce a wide DataFrame with hundreds of
    # columns.  Adjust as necessary for your analysis.
    flat = _flatten_record(xml_dict)
    # Include metadata alongside the flattened XML
    for meta_key in ["IFType", "InstitutionId", "InstitutionName", "Period", "LastUpdate"]:
        if meta_key in entry:
            flat[meta_key] = entry[meta_key]
    return pd.DataFrame([flat])


def fetch_period_data(
    if_type: str,
    periods: Iterable[str],
    *,
    output_dir: Optional[Path] = None,
    max_institutions: Optional[int] = None,
    sleep_seconds: float = 1.0,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch BA900 data for multiple periods and optionally cache it to disk.

    Parameters
    ----------
    if_type: str
        Return type, usually ``"BA900"``.
    periods: Iterable[str]
        A sequence of period strings to download (see :func:`get_periods`).
    output_dir: Optional[Path]
        Directory in which to save raw and processed files.  If not
        provided, defaults to ``data/raw`` relative to the package root.
    max_institutions: Optional[int]
        Limit the number of institutions fetched per period.  Useful
        during development to reduce runtime.
    sleep_seconds: float
        Seconds to wait between consecutive API calls.  Helps avoid
        triggering rate limiting.
    session: Optional[requests.Session]
        Optional session to reuse connections.

    Returns
    -------
    pandas.DataFrame
        A concatenated DataFrame of all flattened BA900 returns for the
        specified periods.
    """
    sess = session or requests.Session()
    out_dir = output_dir or DEFAULT_CACHE_DIR
    _ensure_dir(out_dir)
    all_rows: List[pd.DataFrame] = []
    for period in periods:
        period_dir = out_dir / if_type / period
        _ensure_dir(period_dir)
        # Retrieve institution list
        institutions = get_institutions(if_type, period, session=sess)
        if max_institutions:
            institutions = institutions[:max_institutions]
        for inst in institutions:
            inst_id = inst["Id"]
            # Construct filename for cached JSON/XML
            cache_file = period_dir / f"{inst_id}.json"
            if cache_file.exists():
                # Load cached DataFrame
                try:
                    df_cached = pd.read_json(cache_file)
                    all_rows.append(df_cached)
                    continue
                except Exception:
                    # fall through to re-download
                    pass
            try:
                df = get_institution_data(if_type, period, inst_id, session=sess)
            except Exception as exc:
                print(f"Error fetching {if_type} {period} {inst_id}: {exc}")
                continue
            # Save to cache as JSON for later fast loading
            try:
                df.to_json(cache_file, orient="records")
            except Exception:
                pass
            all_rows.append(df)
            # Sleep to avoid rate limiting
            time.sleep(sleep_seconds)
    # Concatenate all rows (if any) into a single DataFrame
    if all_rows:
        return pd.concat(all_rows, ignore_index=True)
    return pd.DataFrame()


def load_cached_data(
    periods: Iterable[str],
    *,
    if_type: str = "BA900",
    input_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load previously cached BA900 data into a single DataFrame.

    Parameters
    ----------
    periods: Iterable[str]
        A sequence of period identifiers to load.
    if_type: str
        Return type, default ``"BA900"``.
    input_dir: Optional[Path]
        Directory where the cached files are stored.

    Returns
    -------
    pandas.DataFrame
        The concatenated DataFrame of all cached JSON files for the
        given periods.  If no files are found an empty DataFrame is
        returned.
    """
    in_dir = input_dir or DEFAULT_CACHE_DIR
    frames: List[pd.DataFrame] = []
    for period in periods:
        period_dir = in_dir / if_type / period
        if not period_dir.exists():
            continue
        for json_file in period_dir.glob("*.json"):
            try:
                df = pd.read_json(json_file)
                frames.append(df)
            except Exception:
                continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Download BA900 data from the SARB API")
    parser.add_argument("--if-type", default="BA900", help="Return type (BA900, DI900, etc.)")
    parser.add_argument(
        "--periods", nargs="+", help="One or more periods to fetch (e.g. 2024-01-01)")
    parser.add_argument(
        "--output", default=None, help="Directory in which to store cached data (defaults to data/raw)")
    parser.add_argument(
        "--max-institutions", type=int, default=None, help="Limit the number of institutions per period")
    parser.add_argument(
        "--sleep", type=float, default=1.0, help="Seconds to sleep between API calls")
    args = parser.parse_args()
    if not args.periods:
        # If no periods are specified, fetch all 2024 periods by default
        all_periods = get_periods(args.if_type)
        # Parse the JSON response and filter for 2024 periods
        if isinstance(all_periods, list) and len(all_periods) > 0:
            periods_list = json.loads(all_periods[0])
            args.periods = [p for p in periods_list if p.startswith("2024")]
        else:
            # Fallback to hardcoded 2024 periods if API format changes
            args.periods = [
                '2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', 
                '2024-05-01', '2024-06-01', '2024-07-01', '2024-08-01', 
                '2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01'
            ]
    df = fetch_period_data(
        args.if_type,
        periods=args.periods,
        output_dir=Path(args.output) if args.output else None,
        max_institutions=args.max_institutions,
        sleep_seconds=args.sleep,
    )
    print(f"Fetched {len(df)} rows for {len(args.periods)} period(s)")