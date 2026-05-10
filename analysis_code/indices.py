"""

Purpose
-------
Merge seasonal climate indices (e.g., ENSO Niño-3.4 and DMI/IOD)
into an existing GLM predictor dataset.

Features
--------
- Reads monthly climate index files
- Computes seasonal means
- Standardizes indices using a baseline climatology period
- Merges standardized indices into a main GLM dataset

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def compute_zscore_with_baseline(series, years, baseline_start, baseline_end):
    """
    Compute standardized anomaly (z-score) using a climatological baseline.
    """

    mask = (years >= baseline_start) & (years <= baseline_end)

    climatology_mean = series[mask].mean()
    climatology_std = series[mask].std(ddof=1)

    zscore = (series - climatology_mean) / climatology_std

    return zscore, climatology_mean, climatology_std


def load_monthly_index_file(file_path):
    """
    Load monthly climate index file.

    Expected format:
        year Jan Feb Mar ... Dec
    """

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        header=None,
    )

    df.columns = ["year"] + month_names

    return df


def calculate_seasonal_mean(df, months, output_column):
    """
    Calculate seasonal mean from selected months.
    """

    df[output_column] = df[months].mean(axis=1)

    return df


# =========================================================
# MAIN PROCESSING FUNCTION
# =========================================================
def add_climate_indices_to_glm_dataset(
    main_file,
    enso_file,
    iod_file,
    output_dir,
    start_year,
    end_year,
    baseline_start,
    baseline_end,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # LOAD MAIN GLM DATASET
    # =====================================================
    print(f"Loading main GLM dataset: {main_file}")

    df_main = pd.read_csv(main_file)

    df_main = df_main[
        (df_main["year"] >= start_year) &
        (df_main["year"] <= end_year)
    ].copy()

    print("Main dataset columns:")
    print(df_main.columns.tolist())

    # =====================================================
    # LOAD IOD / DMI
    # =====================================================
    print(f"\nLoading DMI/IOD file: {iod_file}")

    iod_df = load_monthly_index_file(iod_file)

    iod_df = iod_df[
        (iod_df["year"] >= start_year) &
        (iod_df["year"] <= end_year)
    ].copy()

    # JJAS seasonal mean
    iod_df = calculate_seasonal_mean(
        iod_df,
        months=["Jun", "Jul", "Aug", "Sep"],
        output_column="dmi_JJAS",
    )

    print("\nDMI JJAS seasonal means:")
    print(iod_df[["year", "dmi_JJAS"]].head())

    # =====================================================
    # LOAD ENSO Niño-3.4
    # =====================================================
    print(f"\nLoading ENSO Niño-3.4 file: {enso_file}")

    enso_df = load_monthly_index_file(enso_file)

    enso_df = enso_df[
        (enso_df["year"] >= start_year) &
        (enso_df["year"] <= end_year)
    ].copy()

    # AMJ seasonal mean
    enso_df = calculate_seasonal_mean(
        enso_df,
        months=["Apr", "May", "Jun"],
        output_column="nino34_AMJ",
    )

    print("\nNiño-3.4 AMJ seasonal means:")
    print(enso_df[["year", "nino34_AMJ"]].head())

    # =====================================================
    # MERGE INDICES
    # =====================================================
    indices_df = pd.merge(
        enso_df[["year", "nino34_AMJ"]],
        iod_df[["year", "dmi_JJAS"]],
        on="year",
        how="inner",
    )

    indices_df = indices_df.sort_values("year").reset_index(drop=True)

    # =====================================================
    # STANDARDIZATION
    # =====================================================
    years = indices_df["year"].values

    (
        indices_df["z_nino34_AMJ"],
        nino_mean,
        nino_std,
    ) = compute_zscore_with_baseline(
        indices_df["nino34_AMJ"],
        years,
        baseline_start,
        baseline_end,
    )

    (
        indices_df["z_dmi_JJAS"],
        dmi_mean,
        dmi_std,
    ) = compute_zscore_with_baseline(
        indices_df["dmi_JJAS"],
        years,
        baseline_start,
        baseline_end,
    )

    print("\nStandardized climate indices:")
    print(indices_df[["year", "z_nino34_AMJ", "z_dmi_JJAS"]].head())

    # =====================================================
    # SAVE INDICES ONLY
    # =====================================================
    indices_output = (
        output_dir /
        f"Climate_indices_ENSO_IOD_{start_year}_{end_year}.csv"
    )

    indices_df.to_csv(indices_output, index=False)

    print(f"\nSaved climate indices file:\n{indices_output}")

    # =====================================================
    # MERGE WITH MAIN GLM DATASET
    # =====================================================
    df_merged = pd.merge(
        df_main,
        indices_df[["year", "z_nino34_AMJ", "z_dmi_JJAS"]],
        on="year",
        how="inner",
    )

    df_merged = df_merged.sort_values("year").reset_index(drop=True)

    print("\nMerged GLM dataset:")
    print(df_merged.head())

    # =====================================================
    # SAVE FINAL MERGED FILE
    # =====================================================
    output_file = (
        output_dir /
        f"GLMDATA_with_climate_indices_{start_year}_{end_year}.csv"
    )

    df_merged.to_csv(output_file, index=False)

    print(f"\nSaved merged GLM dataset:\n{output_file}")

    return df_merged


# =========================================================
# COMMAND LINE INTERFACE
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Add ENSO and IOD climate indices to a GLM dataset."
    )

    parser.add_argument(
        "--main_file",
        required=True,
        help="Path to the main GLM CSV dataset.",
    )

    parser.add_argument(
        "--enso_file",
        required=True,
        help="Path to monthly ENSO Niño-3.4 index file.",
    )

    parser.add_argument(
        "--iod_file",
        required=True,
        help="Path to monthly DMI/IOD index file.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/processed",
        help="Directory for output files.",
    )

    parser.add_argument(
        "--start_year",
        type=int,
        default=1981,
        help="Start year.",
    )

    parser.add_argument(
        "--end_year",
        type=int,
        default=2020,
        help="End year.",
    )

    parser.add_argument(
        "--baseline_start",
        type=int,
        default=1981,
        help="Baseline climatology start year.",
    )

    parser.add_argument(
        "--baseline_end",
        type=int,
        default=2010,
        help="Baseline climatology end year.",
    )

    args = parser.parse_args()

    add_climate_indices_to_glm_dataset(
        main_file=args.main_file,
        enso_file=args.enso_file,
        iod_file=args.iod_file,
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )


if __name__ == "__main__":
    main()