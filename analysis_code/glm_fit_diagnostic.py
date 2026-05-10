"""

Purpose
-------
Fit Gaussian and optional Gamma Generalized Linear Models (GLMs)
for seasonal rainfall analysis and save diagnostic outputs.

Models
------
1. Gaussian GLM with identity link
   - Recommended for rainfall anomalies or approximately symmetric seasonal totals.

2. Gamma GLM with log link
   - Optional model for positive rainfall totals.

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def prepare_model_data(df, target_column, predictors):
    """Prepare clean model matrix and response vector."""

    required_columns = [target_column] + predictors

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

    y = work[target_column]
    x = sm.add_constant(work[predictors], has_constant="add")

    return work, y, x


def extract_coefficient_table(result):
    """Extract GLM coefficients, standard errors, p-values, and confidence intervals."""

    conf_int = result.conf_int()
    conf_int.columns = ["ci_lower", "ci_upper"]

    coef_table = pd.DataFrame(
        {
            "coefficient": result.params,
            "standard_error": result.bse,
            "z_or_t_value": result.tvalues,
            "p_value": result.pvalues,
        }
    )

    coef_table = pd.concat([coef_table, conf_int], axis=1)
    coef_table = coef_table.reset_index().rename(columns={"index": "term"})

    return coef_table


def save_residual_diagnostics(
    df,
    fitted_column,
    residual_column,
    output_dir,
    output_prefix,
):
    """Save residual-vs-fitted and QQ diagnostic plots."""

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df[fitted_column], df[residual_column])
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs fitted values")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{output_prefix}_residuals_vs_fitted.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = sm.qqplot(df[residual_column], line="45")
    plt.title("QQ plot of residuals")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{output_prefix}_qqplot_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def fit_glm_and_diagnostics(
    input_file,
    output_dir,
    region_name,
    target_anom,
    target_total,
    predictors,
    fit_gamma,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {input_file}")

    df_original = pd.read_csv(input_file)
    fitted_output = df_original.copy()

    # =====================================================
    # Gaussian GLM
    # =====================================================
    print("\nFitting Gaussian GLM...")

    gaussian_work, y_gaussian, x_gaussian = prepare_model_data(
        df_original,
        target_anom,
        predictors,
    )

    gaussian_model = sm.GLM(
        y_gaussian,
        x_gaussian,
        family=sm.families.Gaussian(),
    )

    gaussian_result = gaussian_model.fit()

    gaussian_summary_file = (
        output_dir /
        f"{region_name}_gaussian_glm_summary.txt"
    )

    with open(gaussian_summary_file, "w", encoding="utf-8") as file:
        file.write(str(gaussian_result.summary()))

    gaussian_coef_table = extract_coefficient_table(gaussian_result)

    gaussian_coef_file = (
        output_dir /
        f"{region_name}_gaussian_glm_coefficients.csv"
    )

    gaussian_coef_table.to_csv(gaussian_coef_file, index=False)

    gaussian_work["fitted_gaussian"] = gaussian_result.fittedvalues
    gaussian_work["residual_gaussian"] = gaussian_result.resid_response

    save_residual_diagnostics(
        df=gaussian_work,
        fitted_column="fitted_gaussian",
        residual_column="residual_gaussian",
        output_dir=output_dir,
        output_prefix=f"{region_name}_gaussian_glm",
    )

    # merge fitted values back by index
    fitted_output.loc[gaussian_work.index, "fitted_gaussian"] = gaussian_work["fitted_gaussian"]
    fitted_output.loc[gaussian_work.index, "residual_gaussian"] = gaussian_work["residual_gaussian"]

    print(f"Saved Gaussian GLM summary: {gaussian_summary_file}")
    print(f"Saved Gaussian GLM coefficients: {gaussian_coef_file}")

    # =====================================================
    # Gamma GLM
    # =====================================================
    if fit_gamma:
        print("\nFitting Gamma GLM with log link...")

        gamma_work, y_gamma, x_gamma = prepare_model_data(
            df_original,
            target_total,
            predictors,
        )

        gamma_work = gamma_work[y_gamma > 0].copy()
        y_gamma = gamma_work[target_total]
        x_gamma = sm.add_constant(gamma_work[predictors], has_constant="add")

        gamma_model = sm.GLM(
            y_gamma,
            x_gamma,
            family=sm.families.Gamma(
                link=sm.families.links.Log()
            ),
        )

        gamma_result = gamma_model.fit()

        gamma_summary_file = (
            output_dir /
            f"{region_name}_gamma_glm_summary.txt"
        )

        with open(gamma_summary_file, "w", encoding="utf-8") as file:
            file.write(str(gamma_result.summary()))

        gamma_coef_table = extract_coefficient_table(gamma_result)

        gamma_coef_file = (
            output_dir /
            f"{region_name}_gamma_glm_coefficients.csv"
        )

        gamma_coef_table.to_csv(gamma_coef_file, index=False)

        gamma_work["fitted_gamma"] = gamma_result.fittedvalues
        gamma_work["residual_gamma"] = gamma_result.resid_response

        fitted_output.loc[gamma_work.index, "fitted_gamma"] = gamma_work["fitted_gamma"]
        fitted_output.loc[gamma_work.index, "residual_gamma"] = gamma_work["residual_gamma"]

        print(f"Saved Gamma GLM summary: {gamma_summary_file}")
        print(f"Saved Gamma GLM coefficients: {gamma_coef_file}")

    fitted_file = output_dir / f"{region_name}_glm_fitted_values.csv"
    fitted_output.to_csv(fitted_file, index=False)

    print(f"\nSaved fitted values and residuals: {fitted_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Gaussian and optional Gamma GLMs with diagnostics."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to GLM-ready CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/glm_diagnostics",
        help="Directory for GLM output files.",
    )

    parser.add_argument(
        "--region_name",
        default="region",
        help="Region name used in output filenames.",
    )

    parser.add_argument(
        "--target_anom",
        required=True,
        help="Rainfall anomaly target column for Gaussian GLM.",
    )

    parser.add_argument(
        "--target_total",
        required=True,
        help="Positive rainfall total target column for Gamma GLM.",
    )

    parser.add_argument(
        "--predictors",
        nargs="+",
        required=True,
        help="Predictor column names.",
    )

    parser.add_argument(
        "--fit_gamma",
        action="store_true",
        help="Also fit Gamma GLM with log link for positive rainfall totals.",
    )

    args = parser.parse_args()

    fit_glm_and_diagnostics(
        input_file=args.input_file,
        output_dir=args.output_dir,
        region_name=args.region_name,
        target_anom=args.target_anom,
        target_total=args.target_total,
        predictors=args.predictors,
        fit_gamma=args.fit_gamma,
    )


if __name__ == "__main__":
    main()