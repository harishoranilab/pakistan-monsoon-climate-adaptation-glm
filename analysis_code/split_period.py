"""
Purpose
-------
Evaluate coefficient stability of a Gaussian GLM across two
time periods.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fit_gaussian_glm(
    data,
    target_column,
    predictors,
    period_name,
):
    """
    Fit Gaussian GLM with identity link.
    """

    x = data[predictors].copy()

    y = data[target_column].copy()

    x = sm.add_constant(
        x,
        has_constant="add",
    )

    model = sm.GLM(
        y,
        x,
        family=sm.families.Gaussian(),
    )

    result = model.fit()

    conf_int = result.conf_int()

    out = pd.DataFrame(
        {
            "period":
                period_name,

            "predictor":
                result.params.index,

            "coefficient":
                result.params.values,

            "standard_error":
                result.bse.values,

            "z_or_t_value":
                result.tvalues.values,

            "p_value":
                result.pvalues.values,

            "ci95_lower":
                conf_int[0].values,

            "ci95_upper":
                conf_int[1].values,

            "n_years":
                len(data),
        }
    )

    return out, result


def detect_sign_change(
    early_values,
    late_values,
):
    """
    Determine whether coefficient sign changed.
    """

    early_sign = np.sign(
        early_values
    )

    late_sign = np.sign(
        late_values
    )

    return np.where(
        early_sign == late_sign,
        "Same sign",
        "Sign changed",
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def split_period_glm_stability(
    input_file,
    output_dir,
    year_column,
    target_column,
    predictors,
    early_start,
    early_end,
    late_start,
    late_end,
    output_prefix,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading dataset: {input_file}")

    df = pd.read_csv(input_file)

    required_columns = (
        [year_column, target_column] +
        predictors
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    work = df[required_columns].copy()

    for column in required_columns:

        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.dropna()

    work = work.sort_values(
        year_column
    ).reset_index(drop=True)

    print(f"Usable rows: {len(work)}")

    print(
        f"Available years: "
        f"{work[year_column].min()}–"
        f"{work[year_column].max()}"
    )

    # ========================================================
    # STANDARDIZE PREDICTORS
    # ========================================================

    scaler = StandardScaler()

    work_scaled = work.copy()

    work_scaled[predictors] = scaler.fit_transform(
        work[predictors]
    )

    # ========================================================
    # SPLIT DATASETS
    # ========================================================

    early = work_scaled[
        (
            work_scaled[year_column] >= early_start
        ) &
        (
            work_scaled[year_column] <= early_end
        )
    ].copy()

    late = work_scaled[
        (
            work_scaled[year_column] >= late_start
        ) &
        (
            work_scaled[year_column] <= late_end
        )
    ].copy()

    print(f"Early-period rows: {len(early)}")
    print(f"Late-period rows: {len(late)}")

    if len(early) < 10:
        raise ValueError(
            "Too few rows in early period."
        )

    if len(late) < 10:
        raise ValueError(
            "Too few rows in late period."
        )

    # ========================================================
    # FIT MODELS
    # ========================================================

    early_table, early_model = fit_gaussian_glm(
        data=early,
        target_column=target_column,
        predictors=predictors,
        period_name=f"{early_start}–{early_end}",
    )

    late_table, late_model = fit_gaussian_glm(
        data=late,
        target_column=target_column,
        predictors=predictors,
        period_name=f"{late_start}–{late_end}",
    )

    coefficient_long = pd.concat(
        [early_table, late_table],
        ignore_index=True,
    )

    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison = coefficient_long.pivot(
        index="predictor",
        columns="period",
        values="coefficient",
    ).reset_index()

    early_label = (
        f"{early_start}–{early_end}"
    )

    late_label = (
        f"{late_start}–{late_end}"
    )

    comparison["absolute_difference"] = (
        comparison[late_label] -
        comparison[early_label]
    ).abs()

    comparison["sign_stability"] = (
        detect_sign_change(
            comparison[early_label],
            comparison[late_label],
        )
    )

    comparison = comparison.sort_values(
        "absolute_difference",
        ascending=False,
    ).reset_index(drop=True)

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    coefficient_csv = (
        output_dir /
        f"{output_prefix}_coefficients.csv"
    )

    comparison_csv = (
        output_dir /
        f"{output_prefix}_comparison.csv"
    )

    excel_output = (
        output_dir /
        f"{output_prefix}_split_period_analysis.xlsx"
    )

    coefficient_long.to_csv(
        coefficient_csv,
        index=False,
    )

    comparison.to_csv(
        comparison_csv,
        index=False,
    )

    with pd.ExcelWriter(
        excel_output,
        engine="openpyxl",
    ) as writer:

        coefficient_long.to_excel(
            writer,
            sheet_name="Coefficient_details",
            index=False,
        )

        comparison.to_excel(
            writer,
            sheet_name="Split_comparison",
            index=False,
        )

        pd.DataFrame(
            {
                "Metric": [
                    "Early period",
                    "Late period",
                    "Early sample size",
                    "Late sample size",
                    "Target variable",
                    "Predictors standardized",
                    "Model family",
                    "Link function",
                ],

                "Value": [
                    early_label,
                    late_label,
                    len(early),
                    len(late),
                    target_column,
                    (
                        "Yes "
                        "(full-period standardization)"
                    ),
                    "Gaussian",
                    "Identity",
                ],
            }
        ).to_excel(
            writer,
            sheet_name="Model_info",
            index=False,
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n================ "
        "SPLIT-PERIOD COEFFICIENTS "
        "================"
    )

    print(
        coefficient_long.to_string(
            index=False
        )
    )

    print(
        "\n================ "
        "COEFFICIENT COMPARISON "
        "================"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nSaved files:")

    print(coefficient_csv)
    print(comparison_csv)
    print(excel_output)

    print("\nDone.")

    return (
        coefficient_long,
        comparison,
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Split-period GLM coefficient stability analysis."
        )
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to GLM-ready CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/split_period_stability",
        help="Output directory.",
    )

    parser.add_argument(
        "--year_column",
        default="year",
        help="Year column name.",
    )

    parser.add_argument(
        "--target_column",
        required=True,
        help="Target variable column.",
    )

    parser.add_argument(
        "--predictors",
        nargs="+",
        required=True,
        help="Predictor column names.",
    )

    parser.add_argument(
        "--early_start",
        type=int,
        default=1981,
        help="Early-period start year.",
    )

    parser.add_argument(
        "--early_end",
        type=int,
        default=2000,
        help="Early-period end year.",
    )

    parser.add_argument(
        "--late_start",
        type=int,
        default=2001,
        help="Late-period start year.",
    )

    parser.add_argument(
        "--late_end",
        type=int,
        default=2020,
        help="Late-period end year.",
    )

    parser.add_argument(
        "--output_prefix",
        default="glm_split_period",
        help="Prefix for output files.",
    )

    args = parser.parse_args()

    split_period_glm_stability(
        input_file=args.input_file,
        output_dir=args.output_dir,
        year_column=args.year_column,
        target_column=args.target_column,
        predictors=args.predictors,
        early_start=args.early_start,
        early_end=args.early_end,
        late_start=args.late_start,
        late_end=args.late_end,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()