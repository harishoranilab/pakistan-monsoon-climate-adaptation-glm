"""
Purpose
-------
Perform rolling-window GLM coefficient stability analysis.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

    return result


def count_sign_changes(values):
    """
    Count coefficient sign changes across rolling windows.
    """

    signs = np.sign(values)

    sign_changes = np.sum(
        signs[1:] != signs[:-1]
    )

    return int(sign_changes)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def rolling_window_glm_stability(
    input_file,
    output_dir,
    year_column,
    target_column,
    predictors,
    window_length,
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
        f"Year range: "
        f"{work[year_column].min()}–"
        f"{work[year_column].max()}"
    )

    # ========================================================
    # STANDARDIZE PREDICTORS USING FULL SAMPLE
    # ========================================================

    scaler = StandardScaler()

    work_scaled = work.copy()

    work_scaled[predictors] = scaler.fit_transform(
        work[predictors]
    )

    # ========================================================
    # ROLLING-WINDOW GLM FITTING
    # ========================================================

    rolling_rows = []

    min_year = int(
        work_scaled[year_column].min()
    )

    max_year = int(
        work_scaled[year_column].max()
    )

    for start_year in range(
        min_year,
        max_year - window_length + 2,
    ):

        end_year = (
            start_year +
            window_length - 1
        )

        subset = work_scaled[
            (
                work_scaled[year_column] >= start_year
            ) &
            (
                work_scaled[year_column] <= end_year
            )
        ].copy()

        if len(subset) != window_length:

            print(
                f"Skipping {start_year}–{end_year}: "
                f"{len(subset)} years"
            )

            continue

        result = fit_gaussian_glm(
            data=subset,
            target_column=target_column,
            predictors=predictors,
        )

        conf_int = result.conf_int()

        for predictor in result.params.index:

            rolling_rows.append(
                {
                    "window_start":
                        start_year,

                    "window_end":
                        end_year,

                    "window_midpoint":
                        (
                            start_year +
                            end_year
                        ) / 2,

                    "predictor":
                        predictor,

                    "coefficient":
                        result.params[predictor],

                    "standard_error":
                        result.bse[predictor],

                    "z_or_t_value":
                        result.tvalues[predictor],

                    "p_value":
                        result.pvalues[predictor],

                    "ci95_lower":
                        conf_int.loc[predictor, 0],

                    "ci95_upper":
                        conf_int.loc[predictor, 1],

                    "n_years":
                        len(subset),
                }
            )

    rolling_df = pd.DataFrame(
        rolling_rows
    )

    # ========================================================
    # STABILITY SUMMARY
    # ========================================================

    summary_rows = []

    for predictor in rolling_df["predictor"].unique():

        temp = rolling_df[
            rolling_df["predictor"] ==
            predictor
        ].copy()

        summary_rows.append(
            {
                "predictor":
                    predictor,

                "mean_coefficient":
                    temp["coefficient"].mean(),

                "std_coefficient":
                    temp["coefficient"].std(),

                "min_coefficient":
                    temp["coefficient"].min(),

                "max_coefficient":
                    temp["coefficient"].max(),

                "sign_changes_across_windows":
                    count_sign_changes(
                        temp["coefficient"].values
                    ),

                "n_windows":
                    temp.shape[0],

                "n_significant_p_lt_0_05":
                    int(
                        (
                            temp["p_value"] < 0.05
                        ).sum()
                    ),

                "fraction_significant_p_lt_0_05":
                    float(
                        (
                            temp["p_value"] < 0.05
                        ).mean()
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        "std_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    # ========================================================
    # SAVE OUTPUT TABLES
    # ========================================================

    rolling_csv = (
        output_dir /
        f"{output_prefix}_rolling_coefficients.csv"
    )

    summary_csv = (
        output_dir /
        f"{output_prefix}_stability_summary.csv"
    )

    rolling_df.to_csv(
        rolling_csv,
        index=False,
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    excel_output = (
        output_dir /
        f"{output_prefix}_rolling_analysis.xlsx"
    )

    with pd.ExcelWriter(
        excel_output,
        engine="openpyxl",
    ) as writer:

        rolling_df.to_excel(
            writer,
            sheet_name="Rolling_coefficients",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="Stability_summary",
            index=False,
        )

        pd.DataFrame(
            {
                "Metric": [
                    "Rolling window length",
                    "First window",
                    "Last window",
                    "Target variable",
                    "Predictors standardized",
                    "Model family",
                    "Link function",
                ],

                "Value": [
                    window_length,

                    f"{min_year}–"
                    f"{min_year + window_length - 1}",

                    f"{max_year - window_length + 1}–"
                    f"{max_year}",

                    target_column,

                    "Yes",

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
    # ROLLING-COEFFICIENT PLOT
    # ========================================================

    plt.figure(figsize=(11, 7))

    for predictor in predictors:

        temp = rolling_df[
            rolling_df["predictor"] ==
            predictor
        ].copy()

        plt.plot(
            temp["window_midpoint"],
            temp["coefficient"],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=predictor,
        )

    plt.axhline(
        0,
        linewidth=1.0,
        linestyle="--",
    )

    plt.xlabel(
        "Rolling-window midpoint year"
    )

    plt.ylabel(
        "GLM coefficient"
    )

    plt.title(
        "Rolling-window GLM coefficient stability"
    )

    plt.legend(
        fontsize=8,
        frameon=True,
        loc="best",
    )

    plt.tight_layout()

    figure_base = (
        output_dir /
        f"{output_prefix}_rolling_coefficients"
    )

    for extension in [
        "png",
        "tif",
        "jpg",
        "eps",
    ]:

        plt.savefig(
            f"{figure_base}.{extension}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.close()

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print(
        "\n================ "
        "ROLLING-WINDOW STABILITY SUMMARY "
        "================"
    )

    print(
        summary_df.to_string(index=False)
    )

    print("\nSaved outputs:")

    print(rolling_csv)
    print(summary_csv)
    print(excel_output)

    for extension in [
        "png",
        "tif",
        "jpg",
        "eps",
    ]:

        print(f"{figure_base}.{extension}")

    print("\nDone.")

    return rolling_df, summary_df


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Rolling-window GLM coefficient stability analysis."
        )
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to GLM-ready CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/rolling_stability",
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
        "--window_length",
        type=int,
        default=20,
        help="Rolling-window length in years.",
    )

    parser.add_argument(
        "--output_prefix",
        default="glm_stability",
        help="Prefix for output files.",
    )

    args = parser.parse_args()

    rolling_window_glm_stability(
        input_file=args.input_file,
        output_dir=args.output_dir,
        year_column=args.year_column,
        target_column=args.target_column,
        predictors=args.predictors,
        window_length=args.window_length,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()