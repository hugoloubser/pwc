# PwC BA900 Data Science Assessment

This repository provides a reference implementation for the PwC technical assessment that asks candidates to analyse **Non‑Performing Loan (NPL) ratios** in the South African banking sector using the South African Reserve Bank (SARB) **BA900** return.  The work is designed to be pragmatic and reproducible – it includes a data scraping module, preprocessing utilities, exploratory analysis notebooks, a simple machine learning workflow and a lightweight Streamlit dashboard.

## Project structure

```
pwc_ba900_project/
├── ba900/               # Python package containing all reusable code
│   ├── __init__.py
│   ├── scraper.py       # functions to fetch BA900 data from the SARB API
│   ├── macro_fetcher.py # helper functions to retrieve macro indicators (World Bank and SARB)
│   ├── modeling.py      # basic modelling routines (train/test split, regressions, decision trees)
│   ├── visualization.py # helper functions for charts used in EDA and Streamlit app
│   └── app.py           # Streamlit app entry point
├── data/
│   ├── raw/             # cached downloads from SARB API (created at runtime)
│   └── processed/       # cleaned DataFrames stored as CSV/Parquet (created at runtime)
├── notebooks/
│   └── EDA_BA900.ipynb  # Jupyter notebook illustrating exploratory data analysis
├── requirements.txt     # python dependencies
└── README.md            # this file
```

## Overview of the approach

The assessment asks you to answer two primary business questions:

1. **Drivers of NPL ratios** – investigate how macro‑economic indicators (interest rates, inflation, unemployment, household debt‑to‑income, GDP growth, etc.) are related to non‑performing loan rates in the banking sector.  The `ba900/scraper.py` module queries the SARB Web API to download BA900 returns for each bank and merges these with macro series obtained from publicly available sources (e.g. the World Bank API).  Once loaded, the dataset can be explored in the `notebooks/EDA_BA900.ipynb` notebook using pandas, seaborn and matplotlib.
2. **Market share and credit risk** – assess the relationship between a bank’s market share or growth and its credit risk (proxied by NPL ratios).  The modelling code in `ba900/modeling.py` demonstrates how to build simple predictive models (linear regression or decision trees) to explore these interactions.  The focus is on interpretability rather than accuracy, as per the task instructions【645111012086579†L53-L63】.

The `ba900/app.py` script wraps the analysis in a single‑page Streamlit application.  Users can select periods, banks and macro variables, view summary charts and model outputs, and download processed data.  The Streamlit interface is intentionally minimal to prioritise clarity and responsiveness.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Download data**

   The SARB Web API exposes several endpoints for BA900 economic returns.  The most important ones are:

   * `GetPeriods/{ifType}` – returns a list of available periods for a given return type (e.g. `BA900`).  The API documentation describes this call as returning a collection of strings【773677845686591†L7-L33】.
   * `GetInstitutions/{ifType}/{period}` – returns the banks reporting in a specific month.  Each record contains an identifier (`Id`), the bank name and its last update date【977277454646387†L7-L64】.
   * `GetInstitutionData/{ifType}/{period}/{institutionId}` – returns a JSON object whose `XMLData` field holds the full BA900 return for a bank and period【14402629044933†L7-L84】.

   The scraper uses the above endpoints to fetch data for multiple months and institutions.  Because the SARB API rate limits requests, the functions in `ba900/scraper.py` automatically cache responses on disk and throttle requests.  To download the latest data you can run:

   ```bash
   python -m ba900.scraper --if-type BA900 --periods 2025-01-01 2025-02-01 --output data/raw
   ```

   Adjust the list of periods as needed.  The script will create a sub‑directory in `data/raw/` for each period and store the raw XML and parsed CSV files there.

3. **Run the exploratory notebook**

   Open `notebooks/EDA_BA900.ipynb` in JupyterLab or VS Code.  The notebook demonstrates how to load the cached BA900 files, perform basic cleaning (e.g. converting numerical columns, dealing with missing values) and compute derived metrics such as the NPL ratio.  It also pulls macro‑economic indicators via the World Bank API and merges them with the banking data.

4. **Build a model**

   The `ba900/modeling.py` module contains helper functions for preparing and training models.  You can build a regression dataset by merging the bank panel with macro indicators using `prepare_regression_dataset`, then fit a simple model with `train_simple_model`.  For example:

   ```python
   from ba900.scraper import load_cached_data
   from ba900.macro_fetcher import get_world_bank_indicators
   from ba900.modeling import prepare_regression_dataset, train_simple_model

   # load a cached panel of bank data (multi‑index of institution and date)
   bank_panel = load_cached_data(periods=['2025-01-01', '2025-02-01'])

   # fetch macro indicators from the World Bank (GDP growth and inflation)
   macros = get_world_bank_indicators({
       'gdp_growth': 'NY.GDP.MKTP.KD.ZG',
       'inflation': 'FP.CPI.TOTL.ZG'
   }, start_year=2010, end_year=2025)

   # prepare a merged dataset, resampling bank data to annual frequency
   dataset = prepare_regression_dataset(bank_panel, macros, date_freq='A')

   # fit a linear regression
   model, metrics = train_simple_model(dataset, target='npl_ratio', model_type='linear_regression')
   print(f"Test RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
   ```

5. **Launch the dashboard**

   To run the Streamlit application use:

   ```bash
   streamlit run ba900/app.py
   ```

   The app will guide you through selecting a data period, choosing one or more banks and exploring relationships between NPL ratios and macro‑economic indicators.  It is intentionally focused on storytelling and interactivity rather than exhaustive functionality – feel free to adapt it to your own needs.

## Notes

* **Rate limiting** – The SARB API can sometimes block excessive requests.  The scraper implements simple retry logic with exponential back‑off and caches all downloaded responses in `data/raw/` to avoid re‑downloading.  When running the code yourself, consider staggering calls or downloading data during off‑peak hours.
* **NPL calculation** – The BA900 return does not provide a ready‑made “NPL ratio” field.  Instead, it contains multiple line items covering gross loans, impaired loans, specific provisions, etc.  The notebook demonstrates one way to calculate an approximate NPL ratio, but you should consult the official documentation for a precise definition.
* **Macro data sources** – In addition to the SARB API, the project uses the World Bank API for macro indicators.  You can easily substitute other sources (e.g. Stats SA, IMF, FRED) by extending `ba900/macro_fetcher.py`.

## Disclaimer

This code is provided for educational purposes as part of a PwC assessment.  It is not endorsed by the South African Reserve Bank.  Please verify any insights or conclusions derived from this data with official sources before making decisions.
