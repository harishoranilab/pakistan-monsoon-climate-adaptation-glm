
"""
Purpose
-------
Compute JJAS seasonal precipitation totals from ERA5 daily precipitation data
for a user-defined geographic domain using cosine-latitude weighting.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def detect_lat_lon_names(ds):
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]

    lat_name = next((name for name in lat_candidates if name in ds.coords or name in ds.dims), None)
    lon_name = next((name for name in lon_candidates if name in ds.coords or name in ds.dims), None)

    if lat_name is None or lon_name is None:
        raise ValueError(f"Could not detect latitude/longitude names. Dataset dimensions: {ds.dims}")

    return lat_name, lon_name


def standardize_time_name(ds):
    if "time" in ds.dims or "time" in ds.coords:
        return ds, "time"

    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
        return ds, "time"

    raise ValueError(f"No time or valid_time dimension found. Dataset dimensions: {ds.dims}")


def subset_domain(ds, lat_name, lon_name, lat_min, lat_max, lon_min, lon_max):
    if float(ds[lat_name][0]) > float(ds[lat_name][-1]):
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    if float(ds[lon_name][0]) > float(ds[lon_name][-1]):
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})


def clean_extra_dimensions(da):
    for dim in ["number", "expver"]:
        if dim in da.dims:
            da = da.mean(dim)
    return da


def convert_precip_to_mm(da):
    units = (da.attrs.get("units", "") or "").lower().strip()

    if units in ["m", "meter", "metre", "m of water", "meters", "metres"]:
        print("Converting precipitation from metres to millimetres.")
        da = da * 1000.0
        da.attrs["units"] = "mm"

    elif units in ["mm", "millimeter", "millimetre", "mm/day", "mm d-1", "mm per day"]:
        print(f"Precipitation units detected as {units!r}; no conversion applied.")

    else:
        print(f"Warning: unknown precipitation units {units!r}; assuming values are in mm or mm/day.")

    return da


def calculate_era5_jjas_precipitation_total(
    era5_tp_file,
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening ERA5 precipitation file: {era5_tp_file}")
    ds = xr.open_dataset(era5_tp_file)

    ds, time_name = standardize_time_name(ds)
    lat_name, lon_name = detect_lat_lon_names(ds)

    if precip_var not in ds.data_vars:
        raise ValueError(
            f"Variable '{precip_var}' not found. Available variables: {list(ds.data_vars)}"
        )

    da = ds[precip_var]
    da = clean_extra_dimensions(da)

    da_region = subset_domain(
        da,
        lat_name,
        lon_name,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
    )

    da_jjas = da_region.where(da_region[time_name].dt.month.isin([6, 7, 8, 9]), drop=True)
    da_jjas = convert_precip_to_mm(da_jjas)

    weights = np.cos(np.deg2rad(da_jjas[lat_name]))
    weights = weights / weights.mean()

    yearly_total = da_jjas.groupby(f"{time_name}.year").sum(time_name)
    regional_total = yearly_total.weighted(weights).mean((lat_name, lon_name))

    df = regional_total.to_series().reset_index()
    df = df.rename(
        columns={
            precip_var: f"{region_name}_ERA5_PRCPTOT_JJAS_mm",
        }
    )

    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    df = df.sort_values("year").reset_index(drop=True)

    output_file = output_dir / f"{region_name}_ERA5_tp_PRCPTOT_JJAS_{start_year}_{end_year}.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved output: {output_file}")
    print(df.head())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract JJAS ERA5 precipitation totals for a selected region."
    )

    parser.add_argument("--era5_tp_file", required=True, help="Path to ERA5 daily precipitation NetCDF.")
    parser.add_argument("--output_dir", default="outputs/processed", help="Output directory.")
    parser.add_argument("--region_name", default="Pakistan", help="Region name for output file/column.")
    parser.add_argument("--precip_var", default="tp", help="ERA5 precipitation variable name.")

    parser.add_argument("--lat_min", type=float, required=True, help="Minimum latitude.")
    parser.add_argument("--lat_max", type=float, required=True, help="Maximum latitude.")
    parser.add_argument("--lon_min", type=float, required=True, help="Minimum longitude.")
    parser.add_argument("--lon_max", type=float, required=True, help="Maximum longitude.")

    parser.add_argument("--start_year", type=int, default=1981, help="Start year.")
    parser.add_argument("--end_year", type=int, default=2020, help="End year.")

    args = parser.parse_args()

    calculate_era5_jjas_precipitation_total(
        era5_tp_file=args.era5_tp_file,
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