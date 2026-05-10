"""
Purpose
-------
1. Run LASSO regression with K-fold cross-validation.
2. Determine the optimum regularization parameter (lambda / alpha).
3. Save all coefficients and non-zero coefficients.
4. Generate LASSO coefficient path diagnostics.
5. Generate cross-validation MSE diagnostics.
6. Export publication-ready supplementary tables.

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV, lasso_path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def run_lasso_cv_analysis(
    input_file,
    output_dir,
    target_column,
    predictors,
    n_splits,
    random_state,
    max_iter,
    n_alphas,
    output_prefix,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {input_file}")

    df = pd.read_csv(input_file)

    required_columns = ["year", target_column] + predictors

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

    for column in [target_column] + predictors:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.dropna(
        subset=[target_column] + predictors
    ).reset_index(drop=True)

    y = work[target_column].values
    x_raw = work[predictors].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw)

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    print("\nRunning LASSO cross-validation...")

    lasso_cv = LassoCV(
        cv=cv,
        n_alphas=n_alphas,
        random_state=random_state,
        max_iter=max_iter,
    )

    lasso_cv.fit(x, y)

    alpha_optimum = float(lasso_cv.alpha_)

    y_pred = lasso_cv.predict(x)

    rmse_in_sample = np.sqrt(
        mean_squared_error(y, y_pred)
    )

    coefficient_df = pd.DataFrame(
        {
            "Predictor": predictors,
            "Coefficient_at_optimum_lambda": lasso_cv.coef_,
            "Absolute_coefficient": np.abs(lasso_cv.coef_),
            "Nonzero_at_optimum_lambda": lasso_cv.coef_ != 0.0,
        }
    )

    coefficient_df = coefficient_df.sort_values(
        "Absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    nonzero_df = coefficient_df[
        coefficient_df["Nonzero_at_optimum_lambda"]
    ].copy()

    print("\nComputing coefficient path...")

    alpha_grid = np.logspace(-3, 3, 300)

    alphas_path, coefficients_path, _ = lasso_path(
        x,
        y,
        alphas=alpha_grid,
        max_iter=max_iter,
    )

    path_df = pd.DataFrame(
        coefficients_path.T,
        columns=predictors,
    )

    path_df.insert(0, "alpha_lambda", alphas_path)

    print("\nComputing cross-validation diagnostics...")

    cv_mse_mean = lasso_cv.mse_path_.mean(axis=1)
    cv_mse_std = lasso_cv.mse_path_.std(axis=1)

    cv_df = pd.DataFrame(
        {
            "alpha_lambda": lasso_cv.alphas_,
            "mean_cv_mse": cv_mse_mean,
            "std_cv_mse": cv_mse_std,
        }
    )

    cv_df["mean_cv_rmse"] = np.sqrt(cv_df["mean_cv_mse"])

    best_index = int(np.argmin(cv_df["mean_cv_mse"]))

    best_cv_mse = float(cv_df.loc[best_index, "mean_cv_mse"])
    best_cv_rmse = float(cv_df.loc[best_index, "mean_cv_rmse"])

    summary_df = pd.DataFrame(
        {
            "Metric": [
                "K-folds used",
                "Optimum alpha / lambda",
                "Number of candidate predictors",
                "Number of non-zero predictors at optimum lambda",
                "Best mean CV MSE",
                "Best mean CV RMSE",
                "In-sample RMSE at optimum lambda",
                "Target variable",
            ],
            "Value": [
                n_splits,
                alpha_optimum,
                len(predictors),
                int(nonzero_df.shape[0]),
                best_cv_mse,
                best_cv_rmse,
                rmse_in_sample,
                target_column,
            ],
        }
    )

    coefficient_df.to_csv(
        output_dir / f"{output_prefix}_coefficients.csv",
        index=False,
    )

    nonzero_df.to_csv(
        output_dir / f"{output_prefix}_nonzero_coefficients.csv",
        index=False,
    )

    path_df.to_csv(
        output_dir / f"{output_prefix}_coefficient_path.csv",
        index=False,
    )

    cv_df.to_csv(
        output_dir / f"{output_prefix}_cv_mse_by_lambda.csv",
        index=False,
    )

    summary_df.to_csv(
        output_dir / f"{output_prefix}_summary.csv",
        index=False,
    )

    excel_output = (
        output_dir /
        f"{output_prefix}_supplementary_tables.xlsx"
    )

    with pd.ExcelWriter(excel_output) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        coefficient_df.to_excel(
            writer,
            sheet_name="All_coefficients",
            index=False,
        )

        nonzero_df.to_excel(
            writer,
            sheet_name="Nonzero_coefficients",
            index=False,
        )

        cv_df.to_excel(
            writer,
            sheet_name="CV_MSE_by_lambda",
            index=False,
        )

        path_df.to_excel(
            writer,
            sheet_name="Coefficient_path",
            index=False,
        )

    print("\nGenerating coefficient-path figure...")

    plt.figure(figsize=(9, 6))

    for predictor_index, predictor in enumerate(predictors):
        plt.plot(
            alphas_path,
            coefficients_path[predictor_index, :],
            linewidth=1.5,
            label=predictor,
        )

    plt.axvline(
        alpha_optimum,
        linestyle="--",
        linewidth=1.5,
        label=f"Optimum λ = {alpha_optimum:.4g}",
    )

    plt.xscale("log")
    plt.gca().invert_xaxis()

    plt.xlabel("Regularization parameter λ (log scale)")
    plt.ylabel("LASSO coefficient")
    plt.title("LASSO coefficient path")

    plt.legend(
        fontsize=8,
        loc="best",
        frameon=True,
    )

    plt.tight_layout()

    for extension in ["png", "tif", "jpg", "eps"]:
        plt.savefig(
            output_dir / f"{output_prefix}_coefficient_path.{extension}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.close()

    print("Generating CV-MSE curve figure...")

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        cv_df["alpha_lambda"],
        cv_df["mean_cv_mse"],
        yerr=cv_df["std_cv_mse"],
        fmt="-o",
        markersize=3,
        linewidth=1.2,
        capsize=2,
    )

    plt.axvline(
        alpha_optimum,
        linestyle="--",
        linewidth=1.5,
        label=f"Optimum λ = {alpha_optimum:.4g}",
    )

    plt.xscale("log")
    plt.gca().invert_xaxis()

    plt.xlabel("Regularization parameter λ (log scale)")
    plt.ylabel("Mean cross-validated MSE")

    plt.title("K-fold cross-validation curve for LASSO")

    plt.legend(frameon=True)

    plt.tight_layout()

    for extension in ["png", "tif", "jpg", "eps"]:
        plt.savefig(
            output_dir / f"{output_prefix}_cv_mse_curve.{extension}",
            dpi=600,
            bbox_inches="tight",
        )

    plt.close()

    print("\n================ LASSO CV SUMMARY ================")
    print(summary_df.to_string(index=False))

    print("\n=========== NON-ZERO COEFFICIENTS ===========")
    print(nonzero_df.to_string(index=False))

    print("\nSaved outputs to:")
    print(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run LASSO CV diagnostics and coefficient-path analysis."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to GLM-ready CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/lasso_diagnostics",
        help="Directory for outputs.",
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
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=200000,
        help="Maximum LASSO iterations.",
    )

    parser.add_argument(
        "--n_alphas",
        type=int,
        default=200,
        help="Number of lambda values.",
    )

    parser.add_argument(
        "--output_prefix",
        default="lasso_cv",
        help="Prefix for output files.",
    )

    args = parser.parse_args()

    run_lasso_cv_analysis(
        input_file=args.input_file,
        output_dir=args.output_dir,
        target_column=args.target_column,
        predictors=args.predictors,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_alphas=args.n_alphas,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()