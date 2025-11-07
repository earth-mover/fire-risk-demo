# Fire Risk Analysis Demo

Analysis of changing wildfire risk in the Western US using climate projections from MACA v2 (Multivariate Adaptive Constructed Analogs). This demonstration shows how climate change is altering key fire weather drivers (temperature, precipitation, wind, and humidity) and quantifies projected increases in extreme fire risk days.

Watch the [webinar recording](https://www.youtube.com/watch?v=yI4f7ZHBTxs) for a walkthrough of this analysis.

## Features

- **Fire Risk Modeling** - Simple empirical fire risk index (0-100 scale) combining temperature, precipitation, wind speed, and relative humidity
- **Climate Scenario Comparison** - Historical baseline (1950-2005) vs. RCP 8.5 high-emissions scenario (2006-2099)
- **Extreme Event Analysis** - Annual counts of days exceeding high fire risk thresholds
- **Temporal & Spatial Trends** - Time series and maps showing regional patterns of changing fire weather
- **Cloud-Native Processing** - Distributed computation on multi-terabyte climate datasets using Coiled

## Quick Start

Install dependencies and run the analysis notebook:

```bash
pip install -r requirements.txt
jupyter lab fwi_analysis_demo.ipynb
```

## Data Sources

- **MACA v2-METDATA**: Bias-corrected and spatially downscaled climate projections accessed through Arraylake (`earthmover-demos/maca-v2-metdata`)
  - 18 CMIP5 climate models
  - Daily meteorological variables at ~4km resolution
  - Historical (1950-2005) and RCP 8.5 (2006-2099) scenarios
- **Analysis Results**: Fire risk percent change data saved to Arraylake (`earthmover-demos/maca-fire-risk-analysis`)

## Technologies

- **[arraylake](https://arraylake.ai)** - Cloud-native data lake with instant access to petabyte-scale climate datasets
- **[icechunk](https://icechunk.io)** - Transactional storage engine for Zarr with version control and ACID transactions
- **[xarray](https://xarray.dev)** - N-dimensional labeled arrays and datasets for climate data analysis
- **[coiled](https://coiled.io)** - Scalable distributed computing for processing multi-TB datasets in the cloud
