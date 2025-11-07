#!/usr/bin/env python3
"""
Simple Fire Risk Model for MACA Climate Data

A demonstration fire risk model for insurance applications that showcases
the Earthmover data platform (Arraylake + Icechunk).

This model uses a simplified approach based on key fire weather indicators:
- High temperatures (tasmax)
- Low precipitation (pr)
- High wind speeds (uas, vas)
- Low humidity (derived from huss)

The model is intentionally simple to focus on demonstrating the data platform
capabilities rather than implementing a full fire weather index system.
"""

import xarray as xr
import numpy as np
from typing import Literal


def calculate_wind_speed(uas: xr.DataArray, vas: xr.DataArray) -> xr.DataArray:
    """
    Calculate wind speed magnitude from U and V components.

    Parameters
    ----------
    uas : xr.DataArray
        Eastward wind component (m/s)
    vas : xr.DataArray
        Northward wind component (m/s)

    Returns
    -------
    xr.DataArray
        Wind speed magnitude (m/s)
    """
    return np.sqrt(uas**2 + vas**2)


def calculate_relative_humidity(
    huss: xr.DataArray,
    tasmax: xr.DataArray,
    pressure: float = 101325.0
) -> xr.DataArray:
    """
    Estimate relative humidity from specific humidity and temperature.

    Uses the August-Roche-Magnus approximation for saturation vapor pressure.

    Parameters
    ----------
    huss : xr.DataArray
        Specific humidity (kg/kg)
    tasmax : xr.DataArray
        Maximum temperature (K)
    pressure : float
        Atmospheric pressure (Pa), default 101325 Pa (sea level)

    Returns
    -------
    xr.DataArray
        Relative humidity (%)
    """
    # Convert temperature to Celsius
    temp_c = tasmax - 273.15

    # Saturation vapor pressure (hPa) using August-Roche-Magnus
    es = 6.1094 * np.exp((17.625 * temp_c) / (temp_c + 243.04))

    # Actual vapor pressure from specific humidity
    # e = (q * p) / (0.622 + 0.378 * q)
    e = (huss * pressure / 100) / (0.622 + 0.378 * huss)

    # Relative humidity
    rh = 100 * (e / es)

    # Clip to valid range [0, 100]
    return rh.clip(0, 100)


def calculate_fire_risk_index_from_maca(
    ds: xr.Dataset,
    tasmax_var: str = 'tasmax',
    pr_var: str = 'pr',
    uas_var: str = 'uas',
    vas_var: str = 'vas',
    rhsmin_var: str = 'rhsmin',
    temp_weight: float = 0.4,
    precip_weight: float = 0.3,
    wind_weight: float = 0.15,
    rh_weight: float = 0.15
) -> xr.DataArray:
    """
    Calculate fire risk index directly from MACA dataset.

    This is a wrapper that handles MACA data format and calls the main
    fire risk calculation.

    Parameters
    ----------
    ds : xr.Dataset
        MACA dataset containing required variables
    tasmax_var : str
        Name of max temperature variable (default 'tasmax')
    pr_var : str
        Name of precipitation variable (default 'pr')
    uas_var : str
        Name of eastward wind variable (default 'uas')
    vas_var : str
        Name of northward wind variable (default 'vas')
    rhsmin_var : str
        Name of minimum relative humidity variable (default 'rhsmin')
    temp_weight : float
        Weight for temperature component (default 0.4)
    precip_weight : float
        Weight for precipitation component (default 0.3)
    wind_weight : float
        Weight for wind component (default 0.15)
    rh_weight : float
        Weight for humidity component (default 0.15)

    Returns
    -------
    xr.DataArray
        Fire risk index (0-100)
    """
    # Extract variables
    tasmax = ds[tasmax_var]
    pr = ds[pr_var]
    uas = ds[uas_var]
    vas = ds[vas_var]
    rhsmin = ds[rhsmin_var]  # Use minimum RH (worst case)

    # Calculate wind speed
    wind_speed = calculate_wind_speed(uas, vas)

    # Call main calculation
    return calculate_fire_risk_index(
        tasmax=tasmax,
        pr=pr,
        wind_speed=wind_speed,
        rh=rhsmin,  # Use minimum RH (driest conditions)
        temp_weight=temp_weight,
        precip_weight=precip_weight,
        wind_weight=wind_weight,
        rh_weight=rh_weight
    )


def calculate_fire_risk_index(
    tasmax: xr.DataArray,
    pr: xr.DataArray,
    wind_speed: xr.DataArray,
    rh: xr.DataArray,
    temp_weight: float = 0.4,
    precip_weight: float = 0.3,
    wind_weight: float = 0.15,
    rh_weight: float = 0.15
) -> xr.DataArray:
    """
    Calculate a simplified fire risk index (0-100 scale).

    This is a simplified model that combines:
    - Temperature stress (higher = more risk)
    - Precipitation deficit (lower = more risk)
    - Wind speed (higher = more risk)
    - Low humidity (lower = more risk)

    Parameters
    ----------
    tasmax : xr.DataArray
        Maximum temperature (K)
    pr : xr.DataArray
        Precipitation (mm/day)
    wind_speed : xr.DataArray
        Wind speed magnitude (m/s)
    rh : xr.DataArray
        Relative humidity (%)
    temp_weight : float
        Weight for temperature component (default 0.3)
    precip_weight : float
        Weight for precipitation component (default 0.3)
    wind_weight : float
        Weight for wind component (default 0.2)
    rh_weight : float
        Weight for humidity component (default 0.2)

    Returns
    -------
    xr.DataArray
        Fire risk index (0-100, higher = more risk)
    """
    # Normalize each component to 0-100 scale using typical thresholds

    # Temperature: 0°C = 0 risk, 40°C+ = 100 risk
    temp_c = tasmax - 273.15
    temp_norm = ((temp_c - 0) / 40).clip(0, 1) * 100

    # Precipitation: 10mm+ = 0 risk, 0mm = 100 risk
    precip_norm = (1 - (pr / 10).clip(0, 1)) * 100

    # Wind speed: 0 m/s = 0 risk, 15+ m/s = 100 risk
    wind_norm = (wind_speed / 15).clip(0, 1) * 100

    # Relative humidity: 100% = 0 risk, 0% = 100 risk
    rh_norm = (1 - rh / 100) * 100

    # Weighted combination
    fire_risk = (
        temp_weight * temp_norm +
        precip_weight * precip_norm +
        wind_weight * wind_norm +
        rh_weight * rh_norm
    )

    # Add metadata
    fire_risk.attrs = {
        "long_name": "Fire Risk Index",
        "units": "index",
        "valid_range": [0, 100],
        "description": "Simplified fire risk index combining temperature, precipitation, wind, and humidity"
    }

    return fire_risk


def classify_fire_risk(fire_risk: xr.DataArray) -> xr.DataArray:
    """
    Classify fire risk into categories for visualization.

    Parameters
    ----------
    fire_risk : xr.DataArray
        Fire risk index (0-100)

    Returns
    -------
    xr.DataArray
        Fire risk category (1=Low, 2=Moderate, 3=High, 4=Very High, 5=Extreme)
    """
    categories = xr.where(fire_risk < 20, 1,
                 xr.where(fire_risk < 40, 2,
                 xr.where(fire_risk < 60, 3,
                 xr.where(fire_risk < 80, 4, 5))))

    categories.attrs = {
        "long_name": "Fire Risk Category",
        "flag_values": [1, 2, 3, 4, 5],
        "flag_meanings": "low moderate high very_high extreme"
    }

    return categories


def calculate_extreme_fire_days(
    fire_risk: xr.DataArray,
    threshold: float = 70.0,
    time_freq: Literal["month", "year", "season"] = "year"
) -> xr.DataArray:
    """
    Count days exceeding fire risk threshold.

    Useful for insurance analysis: "How many extreme fire risk days per year?"

    Parameters
    ----------
    fire_risk : xr.DataArray
        Fire risk index (0-100) with time dimension
    threshold : float
        Fire risk threshold for "extreme" classification (default 70)
    time_freq : str
        Temporal aggregation frequency: "month", "year", or "season"

    Returns
    -------
    xr.DataArray
        Count of extreme fire risk days
    """
    extreme_days = fire_risk > threshold

    if time_freq == "year":
        return extreme_days.resample(time="YE").sum()
    elif time_freq == "month":
        return extreme_days.resample(time="ME").sum()
    elif time_freq == "season":
        return extreme_days.resample(time="QE-DEC").sum()
    else:
        raise ValueError(f"Invalid time_freq: {time_freq}")


def calculate_fire_season_length(
    fire_risk: xr.DataArray,
    threshold: float = 40.0
) -> xr.DataArray:
    """
    Calculate the length of fire season (consecutive days above threshold).

    For each year, finds the longest continuous period where fire risk
    exceeds the threshold.

    Parameters
    ----------
    fire_risk : xr.DataArray
        Fire risk index (0-100) with time dimension
    threshold : float
        Fire risk threshold defining "fire season" (default 40)

    Returns
    -------
    xr.DataArray
        Maximum consecutive days above threshold per year
    """
    # This is a simplified version - a full implementation would use
    # more sophisticated run-length encoding
    above_threshold = fire_risk > threshold

    # Annual maximum of 30-day rolling mean as proxy for season length
    rolling_mean = above_threshold.rolling(time=30, center=True).mean()
    annual_max = rolling_mean.resample(time="YE").max() * 30

    annual_max.attrs = {
        "long_name": "Fire Season Length",
        "units": "days",
        "description": f"Approximate length of fire season (fire risk > {threshold})"
    }

    return annual_max


def compare_scenarios(
    historical: xr.DataArray,
    future: xr.DataArray,
    metric: Literal["mean", "max", "p90", "extreme_days"] = "mean"
) -> xr.DataArray:
    """
    Compare fire risk between historical and future scenarios.

    Useful for insurance: "How much does fire risk increase in RCP 8.5?"

    Parameters
    ----------
    historical : xr.DataArray
        Historical fire risk data
    future : xr.DataArray
        Future scenario fire risk data
    metric : str
        Comparison metric: "mean", "max", "p90" (90th percentile), or "extreme_days"

    Returns
    -------
    xr.DataArray
        Difference (future - historical) or ratio (future / historical)
    """
    if metric == "mean":
        hist_val = historical.mean(dim="time")
        future_val = future.mean(dim="time")
    elif metric == "max":
        hist_val = historical.max(dim="time")
        future_val = future.max(dim="time")
    elif metric == "p90":
        hist_val = historical.quantile(0.9, dim="time")
        future_val = future.quantile(0.9, dim="time")
    elif metric == "extreme_days":
        hist_val = calculate_extreme_fire_days(historical, time_freq="year").mean(dim="time")
        future_val = calculate_extreme_fire_days(future, time_freq="year").mean(dim="time")
    else:
        raise ValueError(f"Invalid metric: {metric}")

    # Calculate absolute difference
    difference = future_val - hist_val

    # Calculate percent change
    percent_change = 100 * (future_val - hist_val) / hist_val

    return xr.Dataset({
        "absolute_change": difference,
        "percent_change": percent_change,
        "historical": hist_val,
        "future": future_val
    })


def calculate_ensemble_statistics(
    fire_risk: xr.DataArray,
    model_dim: str = "model"
) -> xr.Dataset:
    """
    Calculate ensemble statistics across climate models.

    Important for insurance: quantify uncertainty in projections.

    Parameters
    ----------
    fire_risk : xr.DataArray
        Fire risk with model dimension
    model_dim : str
        Name of the model dimension (default "model")

    Returns
    -------
    xr.Dataset
        Ensemble mean, median, std, and percentiles (10th, 90th)
    """
    return xr.Dataset({
        "ensemble_mean": fire_risk.mean(dim=model_dim),
        "ensemble_median": fire_risk.median(dim=model_dim),
        "ensemble_std": fire_risk.std(dim=model_dim),
        "ensemble_p10": fire_risk.quantile(0.1, dim=model_dim),
        "ensemble_p90": fire_risk.quantile(0.9, dim=model_dim),
    })


# Example usage
if __name__ == "__main__":
    print("Fire Risk Model for MACA Climate Data")
    print("=" * 50)
    print("\nThis module provides simplified fire risk calculations")
    print("for demonstrating the Earthmover data platform.\n")
    print("Key functions:")
    print("  - calculate_fire_risk_index(): Main risk calculation")
    print("  - calculate_extreme_fire_days(): Count high-risk days")
    print("  - compare_scenarios(): Historical vs future comparison")
    print("  - calculate_ensemble_statistics(): Multi-model uncertainty")
    print("\nSee demo notebook for usage examples.")
