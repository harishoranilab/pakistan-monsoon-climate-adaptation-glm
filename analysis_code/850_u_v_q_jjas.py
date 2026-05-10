
"""

Purpose
-------
Compute JJAS seasonal means of ERA5 pressure-level variables over a
user-defined geographic domain using cosine-latitude weighting.

Default variables:
    u : zonal wind
    v : meridional wind
    q : specific humidity

Default pressure level:
    850 hPa

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
        raise ValueError(f"Could not detect latitude/longitude names. Dataset dimensions: {ds.dims}")

    return lat_name, lon_name


def standardize_time_name(ds):
    """Rename valid_time to time if needed."""
    if "time" in ds.dims or "time" in ds.coords:
        return ds, "time"

    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
        return ds, "time"

    raise ValueError(f"No time or valid_time dimension found. Dataset dimensions: {ds.dims}")


def select_pressure_level(ds, pressure_level):
    """Select pressure level if a pressure-level dimension exists."""
    level_candidates = ["level", "pressure_level", "plev", "isobaricInhPa"]

    level_name = next((name for name in level_candidates if name in ds.coords or name in ds.dims), None)

    if level_name is None:
        print("No pressure-level dimension found. Assuming file already contains one pressure level.")
        return ds

    available_levels = ds[level_name].values

    if pressure_level not in available_levels:
        raise ValueError(
            f"Pressure level {pressure_level} not found in '{level_name}'. "
            f"Available levels: {available_levels}"
        )

    return ds.sel({level_name: pressure_level})


def remove_extra_dimensions(ds):
    """Average over common extra ERA5 dimensions if present."""
    for dim in ["expver", "number"]:
        if dim in ds.dims:
            ds = ds.mean(dim)
    return ds


def subset_domain(ds, lat_name, lon_name, lat_min, lat_max, lon_min, lon_max):
    """Subset dataset using a latitude-longitude bounding box."""

    if float(ds[lat_name][0]) > float(ds[lat_name][-1]):
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    if float(ds[lon_name][0]) > float(ds[lon_name][-1]):
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})


def area_weighted_jjas_mean(ds, var_name, time_name, lat_name, lon_name):
    """Compute yearly JJAS mean for one variable using cosine-latitude weighting."""

    if var_name not in ds.data_vars:
        raise ValueError(
            f"Variable '{var_name}' not found. Available variables: {list(ds.data_vars)}"
        )

    ds_jjas = ds.where(ds[time_name].dt.month.isin([6, 7, 8, 9]), drop=True)

    weights = np.cos(np.deg2rad(ds_jjas[lat_name]))
    weights = weights / weights.mean()

    yearly_mean = ds_jjas[var_name].groupby(f"{time_name}.year").mean(time_name)

    return yearly_mean.weighted(weights).mean((lat_name, lon_name))


def extract_era5_pressure_level_jjas_means(
    era5_file,
    output_dir,
    region_name,
    variables,
    pressure_level,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    start_year,
    end_year,
):
    """Extract area-weighted JJAS seasonal means for selected ERA5 pressure-level variables."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ERA5 pressure-level file: {era5_file}")
    ds = xr.open_dataset(era5_file)

    ds, time_name = standardize_time_name(ds)
    ds = select_pressure_level(ds, pressure_level)
    ds = remove_extra_dimensions(ds)

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

    output_data = {}

    for var_name in variables:
        print(f"Processing variable: {var_name}")
        da = area_weighted_jjas_mean(
            ds_region,
            var_name,
            time_name,
            lat_name,
            lon_name,
        )

        output_data["year"] = da["year"].values

        if pressure_level is not None:
            output_data[f"{var_name}{int(pressure_level)}_JJAS"] = da.values
        else:
            output_data[f"{var_name}_JJAS"] = da.values

    df = pd.DataFrame(output_data)
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]
    df = df.sort_values("year").reset_index(drop=True)

    output_file = output_dir / (
        f"{region_name}_ERA5_pressure_level_{int(pressure_level)}_JJAS_"
        f"{start_year}_{end_year}.csv"
    )

    df.to_csv(output_file, index=False)

    print(f"Saved output: {output_file}")
    print(df.head())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract JJAS seasonal means from ERA5 pressure-level monthly data."
    )

    parser.add_argument("--era5_file", required=True, help="Path to ERA5 pressure-level NetCDF file.")
    parser.add_argument("--output_dir", default="outputs/processed", help="Output directory.")
    parser.add_argument("--region_name", default="Pakistan", help="Region name for output file.")

    parser.add_argument(
        "--variables",
        nargs="+",
        default=["u", "v", "q"],
        help="ERA5 pressure-level variable names to process.",
    )

    parser.add_argument(
        "--pressure_level",
        type=float,
        default=850,
        help="Pressure level in hPa. Default is 850.",
    )

    parser.add_argument("--lat_min", type=float, required=True, help="Minimum latitude.")
    parser.add_argument("--lat_max", type=float, required=True, help="Maximum latitude.")
    parser.add_argument("--lon_min", type=float, required=True, help="Minimum longitude.")
    parser.add_argument("--lon_max", type=float, required=True, help="Maximum longitude.")

    parser.add_argument("--start_year", type=int, default=1981, help="Start year.")
    parser.add_argument("--end_year", type=int, default=2020, help="End year.")

    args = parser.parse_args()

    extract_era5_pressure_level_jjas_means(
        era5_file=args.era5_file,
        output_dir=args.output_dir,
        region_name=args.region_name,
        variables=args.variables,
        pressure_level=args.pressure_level,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()