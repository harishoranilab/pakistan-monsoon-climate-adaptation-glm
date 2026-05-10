"""

Purpose
-------
Screen candidate predictors for a GLM rainfall model using:

1. Variance Inflation Factor (VIF)
2. LASSO regression with K-fold cross-validation

Outputs
-------
1. VIF table
2. LASSO coefficient table
3. Final suggested predictor list
4. Selected LASSO alpha value

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif_table(df, predictors):
    """Compute VIF values for candidate predictors."""

    x = df[predictors].copy()

    x_standardized = (x - x.mean()) / x.std(ddof=0)
    x_standardized = x_standardized.replace([np.inf, -np.inf], np.nan).dropna()

    vif_records = []

    for index, predictor in enumerate(predictors):
        vif_value = variance_inflation_factor(x_standardized.values, index)
        vif_records.append(
            {
                "predictor": predictor,
                "VIF": vif_value,
            }
        )

    vif_df = pd.DataFrame(vif_records)
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    return vif_df


def run_lasso_cv(df, target_column, predictors, cv_folds, random_state):
    """Run LASSO regression with cross-validation."""

    x = df[predictors].copy()
    y = df[target_column].copy()

    valid_data = pd.concat([y, x], axis=1)
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna()

    y_clean = valid_data[target_column].values
    x_clean = valid_data[predictors].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_clean)

    lasso = LassoCV(
        cv=cv_folds,
        random_state=random_state,
        max_iter=10000,
    )

    lasso.fit(x_scaled, y_clean)

    coef_df = pd.DataFrame(
        {
            "predictor": predictors,
            "lasso_coefficient": lasso.coef_,
            "absolute_coefficient": np.abs(lasso.coef_),
            "nonzero": lasso.coef_ != 0.0,
        }
    )

    coef_df = coef_df.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return lasso, coef_df


def predictor_screening_vif_lasso(
    input_file,
    output_dir,
    target_column,
    predictors,
    cv_folds,
    random_state,
    output_prefix,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file)

    required_columns = [target_column] + predictors
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print("\nRunning VIF analysis...")
    vif_df = compute_vif_table(df, predictors)

    vif_file = output_dir / f"{output_prefix}_vif_results.csv"
    vif_df.to_csv(vif_file, index=False)

    print(f"Saved VIF results: {vif_file}")
    print(vif_df)

    print("\nRunning LASSO cross-validation...")
    lasso_model, lasso_df = run_lasso_cv(
        df=df,
        target_column=target_column,
        predictors=predictors,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    lasso_file = output_dir / f"{output_prefix}_lasso_coefficients.csv"
    lasso_df.to_csv(lasso_file, index=False)

    print(f"Saved LASSO coefficients: {lasso_file}")
    print(lasso_df)

    final_predictors = lasso_df.loc[
        lasso_df["nonzero"],
        "predictor",
    ].tolist()

    final_predictors_file = output_dir / f"{output_prefix}_final_predictors.txt"

    with open(final_predictors_file, "w", encoding="utf-8") as file:
        file.write("\n".join(final_predictors))

    alpha_file = output_dir / f"{output_prefix}_selected_lasso_alpha.txt"

    with open(alpha_file, "w", encoding="utf-8") as file:
        file.write(f"Selected LASSO alpha: {lasso_model.alpha_}\n")

    print(f"\nSelected LASSO alpha: {lasso_model.alpha_}")
    print(f"Suggested final predictors: {final_predictors}")
    print(f"Saved final predictor list: {final_predictors_file}")
    print(f"Saved selected alpha: {alpha_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Screen GLM predictors using VIF and LASSO cross-validation."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to GLM-ready CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/screening",
        help="Directory for output files.",
    )

    parser.add_argument(
        "--target_column",
        required=True,
        help="Target column for LASSO regression.",
    )

    parser.add_argument(
        "--predictors",
        nargs="+",
        required=True,
        help="Candidate predictor column names.",
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for LASSO.",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--output_prefix",
        default="predictor_screening",
        help="Prefix for output files.",
    )

    args = parser.parse_args()

    predictor_screening_vif_lasso(
        input_file=args.input_file,
        output_dir=args.output_dir,
        target_column=args.target_column,
        predictors=args.predictors,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()