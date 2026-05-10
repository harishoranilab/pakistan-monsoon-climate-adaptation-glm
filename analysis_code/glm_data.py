#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: 06_merge_predictor_files_for_glm.py

Purpose
-------
Merge CHIRPS rainfall, ERA5 precipitation, ERA5 pressure-level variables,
and ERA5 single-level variables into one GLM-ready seasonal dataset.

Example
-------
python 06_merge_predictor_files_for_glm.py \
    --rain_file outputs/processed/Pakistan_CHIRPS_PRCPTOT_JJAS_1981_2020.csv \
    --era5_tp_file outputs/processed/Pakistan_ERA5_tp_PRCPTOT_JJAS_1981_2020.csv \
    --pressure_file outputs/processed/Pakistan_ERA5_pressure_level_850_JJAS_1981_2020.csv \
    --single_file outputs/processed/Pakistan_ERA5_single_levels_JJAS_1981_2020.csv \
    --output_dir outputs/processed \
    --region_name Pakistan \
    --start_year 1981 \
    --end_year 2020 \
    --baseline_start 1981 \
    --baseline_end 2010
"""

import argparse
from pathlib import Path

import pandas as pd


def merge_predictor_files_for_glm(
    rain_file,
    era5_tp_file,
    pressure_file,
    single_file,
    output_dir,
    region_name,
    start_year,
    end_year,
    baseline_start,
    baseline_end,
    rainfall_column,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading input files...")

    rain_df = pd.read_csv(rain_file)
    tp_df = pd.read_csv(era5_tp_file)
    pressure_df = pd.read_csv(pressure_file)
    single_df = pd.read_csv(single_file)

    required_files = {
        "rainfall": rain_df,
        "ERA5 precipitation": tp_df,
        "ERA5 pressure-level": pressure_df,
        "ERA5 single-level": single_df,
    }

    for name, df in required_files.items():
        if "year" not in df.columns:
            raise ValueError(f"The {name} file does not contain a 'year' column.")

    print("Merging files on year...")

    df = rain_df.merge(tp_df, on="year", how="inner")
    df = df.merge(pressure_df, on="year", how="inner")
    df = df.merge(single_df, on="year", how="inner")

    df = df[
        (df["year"] >= start_year) &
        (df["year"] <= end_year)
    ].sort_values("year").reset_index(drop=True)

    if rainfall_column not in df.columns:
        raise ValueError(
            f"Rainfall column '{rainfall_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    baseline_mask = (
        (df["year"] >= baseline_start) &
        (df["year"] <= baseline_end)
    )

    rainfall_baseline_mean = df.loc[baseline_mask, rainfall_column].mean()

    anomaly_column = f"{rainfall_column}_anom_{baseline_start}_{baseline_end}"
    df[anomaly_column] = df[rainfall_column] - rainfall_baseline_mean

    output_file = output_dir / f"{region_name}_JJAS_GLMDATA_{start_year}_{end_year}.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved merged GLM dataset: {output_file}")
    print(df.head())

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Merge seasonal rainfall and ERA5 predictor files for GLM analysis."
    )

    parser.add_argument("--rain_file", required=True, help="Path to CHIRPS rainfall CSV.")
    parser.add_argument("--era5_tp_file", required=True, help="Path to ERA5 precipitation CSV.")
    parser.add_argument("--pressure_file", required=True, help="Path to ERA5 pressure-level CSV.")
    parser.add_argument("--single_file", required=True, help="Path to ERA5 single-level CSV.")

    parser.add_argument("--output_dir", default="outputs/processed", help="Output directory.")
    parser.add_argument("--region_name", default="Pakistan", help="Region name.")

    parser.add_argument("--start_year", type=int, default=1981, help="Start year.")
    parser.add_argument("--end_year", type=int, default=2020, help="End year.")

    parser.add_argument("--baseline_start", type=int, default=1981, help="Baseline start year.")
    parser.add_argument("--baseline_end", type=int, default=2010, help="Baseline end year.")

    parser.add_argument(
        "--rainfall_column",
        default="Pakistan_CHIRPS_PRCPTOT_JJAS_mm",
        help="Rainfall column used to compute anomalies.",
    )

    args = parser.parse_args()

    merge_predictor_files_for_glm(
        rain_file=args.rain_file,
        era5_tp_file=args.era5_tp_file,
        pressure_file=args.pressure_file,
        single_file=args.single_file,
        output_dir=args.output_dir,
        region_name=args.region_name,
        start_year=args.start_year,
        end_year=args.end_year,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        rainfall_column=args.rainfall_column,
    )


if __name__ == "__main__":
    main()