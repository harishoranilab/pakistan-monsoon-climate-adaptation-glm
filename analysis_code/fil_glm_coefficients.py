"""
Purpose
-------
Fit Gaussian GLM models for multiple regions and export one combined
coefficient table.

Input
-----
A region-configuration CSV file with columns:

region,file,target,predictors

The predictors column should contain predictor names separated by semicolons.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def parse_predictors(predictor_string):
    """Parse semicolon-separated predictor names."""

    predictors = [
        item.strip()
        for item in str(predictor_string).split(";")
        if item.strip()
    ]

    if not predictors:
        raise ValueError("Predictor list is empty.")

    return predictors


def fit_gaussian_glm(df, target, predictors):
    """Fit Gaussian GLM and return fitted result."""

    required_columns = [target] + predictors

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    work = df[required_columns].copy()

    for column in required_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    y = work[target]
    x = sm.add_constant(work[predictors], has_constant="add")

    model = sm.GLM(
        y,
        x,
        family=sm.families.Gaussian(),
    )

    result = model.fit()

    return result, len(work)


def extract_coefficients(result, region, n_obs):
    """Extract coefficient table from fitted GLM result."""

    conf_int = result.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]

    coefficient_df = pd.DataFrame(
        {
            "region": region,
            "term": result.params.index,
            "coefficient": result.params.values,
            "standard_error": result.bse.values,
            "z_or_t_value": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_lower": conf_int["ci_lower"].values,
            "ci_upper": conf_int["ci_upper"].values,
            "n_observations": n_obs,
            "aic": result.aic,
            "bic": result.bic if hasattr(result, "bic") else np.nan,
            "deviance": result.deviance,
        }
    )

    return coefficient_df


def fit_all_regions(
    config_file,
    output_dir,
    output_prefix,
    round_digits,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = pd.read_csv(config_file)

    required_config_columns = ["region", "file", "target", "predictors"]

    missing_config_columns = [
        column for column in required_config_columns
        if column not in config.columns
    ]

    if missing_config_columns:
        raise ValueError(
            f"Config file is missing columns: {missing_config_columns}"
        )

    all_coefficients = []

    for _, row in config.iterrows():
        region = row["region"]
        input_file = Path(row["file"])
        target = row["target"]
        predictors = parse_predictors(row["predictors"])

        print(f"\n=== Fitting Gaussian GLM for region: {region} ===")
        print(f"Input file: {input_file}")

        df = pd.read_csv(input_file)

        result, n_obs = fit_gaussian_glm(
            df=df,
            target=target,
            predictors=predictors,
        )

        region_coefficients = extract_coefficients(
            result=result,
            region=region,
            n_obs=n_obs,
        )

        all_coefficients.append(region_coefficients)

        summary_file = output_dir / f"{output_prefix}_{region}_summary.txt"

        with open(summary_file, "w", encoding="utf-8") as file:
            file.write(str(result.summary()))

        print(f"Saved summary: {summary_file}")

    coefficient_table = pd.concat(
        all_coefficients,
        axis=0,
        ignore_index=True,
    )

    numeric_columns = coefficient_table.select_dtypes(
        include=["float", "int"]
    ).columns

    coefficient_table[numeric_columns] = coefficient_table[numeric_columns].round(
        round_digits
    )

    output_csv = output_dir / f"{output_prefix}_coefficients_all_regions.csv"
    coefficient_table.to_csv(output_csv, index=False)

    output_excel = output_dir / f"{output_prefix}_coefficients_all_regions.xlsx"
    coefficient_table.to_excel(output_excel, index=False)

    print(f"\nSaved combined coefficient table: {output_csv}")
    print(f"Saved Excel coefficient table: {output_excel}")

    return coefficient_table


def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian GLMs for multiple regions and export coefficient tables."
    )

    parser.add_argument(
        "--config_file",
        required=True,
        help="CSV file containing region, file, target, and predictors columns.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/glm_coefficients",
        help="Directory for model summaries and coefficient tables.",
    )

    parser.add_argument(
        "--output_prefix",
        default="glm_gaussian",
        help="Prefix for output files.",
    )

    parser.add_argument(
        "--round_digits",
        type=int,
        default=4,
        help="Number of digits for rounded numeric output.",
    )

    args = parser.parse_args()

    fit_all_regions(
        config_file=args.config_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        round_digits=args.round_digits,
    )


if __name__ == "__main__":
    main()