
"""
Purpose
-------
Extract seasonal JJAS precipitation totals from daily CHIRPS NetCDF data
for a user-defined geographic domain and save the area-averaged rainfall
time series as a CSV file.


"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def detect_lat_lon_names(ds):
    """Detect latitude and longitude coordinate names."""
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]

    lat_name = next((name for name in lat_candidates if name in ds.coords or name in ds.dims), None)
    lon_name = next((name for name in lon_candidates if name in ds.coords or name in ds.dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError("Could not detect latitude/longitude names in the NetCDF file.")

    return lat_name, lon_name


def subset_domain(ds, lat_name, lon_name, lat_min, lat_max, lon_min, lon_max):
    """Subset dataset to selected latitude-longitude box."""
    lat_values = ds[lat_name].values

    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    return ds.sel(
        {
            lat_name: lat_slice,
            lon_name: slice(lon_min, lon_max),
        }
    )


def calculate_jjas_precipitation(
    chirps_file,
    output_dir,
    region_name,
    precip_var,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start_year,
    end_year,
):
    """Calculate cosine-latitude weighted JJAS rainfall totals."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CHIRPS file: {chirps_file}")
    ds = xr.open_dataset(chirps_file, chunks={"time": 60})

    if precip_var not in ds.data_vars:
        available_vars = list(ds.data_vars)
        raise ValueError(
            f"Variable '{precip_var}' not found. Available variables: {available_vars}"
        )

    lat_name, lon_name = detect_lat_lon_names(ds)

    ds_region = subset_domain(
        ds,
        lat_name,
        lon_name,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )

    ds_jjas = ds_region.where(ds_region["time"].dt.month.isin([6, 7, 8, 9]), drop=True)

    weights = np.cos(np.deg2rad(ds_jjas[lat_name]))
    weights = weights / weights.mean()

    rainfall_jjas = (
        ds_jjas[precip_var]
        .groupby("time.year")
        .sum("time")
        .weighted(weights)
        .mean((lat_name, lon_name))
    )

    df = rainfall_jjas.to_series().reset_index()
    df = df.rename(
        columns={
            "year": "year",
            precip_var: f"{region_name}_CHIRPS_PRCPTOT_JJAS_mm",
        }
    )

    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    df = df.sort_values("year").reset_index(drop=True)

    output_file = output_dir / f"{region_name}_CHIRPS_PRCPTOT_JJAS_{start_year}_{end_year}.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved output: {output_file}")
    print(df.head())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract JJAS CHIRPS rainfall totals for a selected region."
    )

    parser.add_argument("--chirps_file", required=True, help="Path to daily CHIRPS NetCDF file.")
    parser.add_argument("--output_dir", default="outputs/processed", help="Directory for output CSV.")
    parser.add_argument("--region_name", default="Pakistan", help="Region name for output column/file.")
    parser.add_argument("--precip_var", default="precip", help="CHIRPS precipitation variable name.")

    parser.add_argument("--lat_min", type=float, required=True, help="Minimum latitude.")
    parser.add_argument("--lat_max", type=float, required=True, help="Maximum latitude.")
    parser.add_argument("--lon_min", type=float, required=True, help="Minimum longitude.")
    parser.add_argument("--lon_max", type=float, required=True, help="Maximum longitude.")

    parser.add_argument("--start_year", type=int, default=1981, help="Start year.")
    parser.add_argument("--end_year", type=int, default=2020, help="End year.")

    args = parser.parse_args()

    calculate_jjas_precipitation(
        chirps_file=args.chirps_file,
        output_dir=args.output_dir,
        region_name=args.region_name,
        precip_var=args.precip_var,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()