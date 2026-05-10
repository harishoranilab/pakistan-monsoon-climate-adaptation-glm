
"""
Purpose
-------
Perform Leave-One-Year-Out (LOYO) cross-validation for multiple
Gaussian GLM configurations and compare predictive skill.

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def parse_predictors(predictor_string):
    """Parse semicolon-separated predictor names."""

    predictors = [
        item.strip()
        for item in str(predictor_string).split(";")
        if item.strip()
    ]

    return predictors


def load_model_config(model_config_file):
    """Load model definitions from CSV."""

    config_df = pd.read_csv(model_config_file)

    required_columns = ["model_name", "predictors"]

    missing_columns = [
        column for column in required_columns
        if column not in config_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing config columns: {missing_columns}"
        )

    models = {}

    for _, row in config_df.iterrows():
        models[row["model_name"]] = parse_predictors(
            row["predictors"]
        )

    return models


def clean_train_test_data(
    train,
    test,
    target_anom,
    target_total,
    predictors,
):
    """Remove NaN and infinite values."""

    required_columns = [target_anom, target_total] + predictors

    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    train = train.dropna(subset=required_columns)
    test = test.dropna(subset=required_columns)

    return train, test


def compute_skill_metrics(observed, predicted, climatology):
    """Compute deterministic skill metrics."""

    rmse_model = np.sqrt(
        mean_squared_error(observed, predicted)
    )

    rmse_climatology = np.sqrt(
        mean_squared_error(observed, climatology)
    )

    mae_model = mean_absolute_error(observed, predicted)

    mae_climatology = mean_absolute_error(
        observed,
        climatology,
    )

    r2_model = r2_score(observed, predicted)

    corr_model = np.corrcoef(observed, predicted)[0, 1]

    skill_rmse = 1.0 - (
        rmse_model / rmse_climatology
    )

    return {
        "RMSE_model": rmse_model,
        "RMSE_climatology": rmse_climatology,
        "MAE_model": mae_model,
        "MAE_climatology": mae_climatology,
        "R2_model": r2_model,
        "Corr_model": corr_model,
        "Skill_RMSE": skill_rmse,
    }


def save_scatter_plot(
    cv_df,
    output_dir,
    region_name,
    model_name,
):
    """Save observed-vs-predicted scatter plot."""

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        cv_df["obs_total"],
        cv_df["pred_total"],
    )

    min_val = min(
        cv_df["obs_total"].min(),
        cv_df["pred_total"].min(),
    )

    max_val = max(
        cv_df["obs_total"].max(),
        cv_df["pred_total"].max(),
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel("Observed JJAS rainfall (mm)")
    ax.set_ylabel("LOYO predicted JJAS rainfall (mm)")

    ax.set_title(
        f"{region_name}: observed vs predicted ({model_name})"
    )

    fig.tight_layout()

    fig.savefig(
        output_dir /
        f"{region_name}_scatter_obs_vs_pred_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_timeseries_plot(
    cv_df,
    output_dir,
    region_name,
    model_name,
):
    """Save observed vs predicted time-series plot."""

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        cv_df["year"],
        cv_df["obs_total"],
        marker="o",
        label="Observed",
    )

    ax.plot(
        cv_df["year"],
        cv_df["pred_total"],
        marker="s",
        label="LOYO predicted",
    )

    ax.plot(
        cv_df["year"],
        cv_df["baseline_total"],
        linestyle=":",
        linewidth=1.5,
        label="Climatology",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("JJAS rainfall (mm)")

    ax.set_title(
        f"{region_name}: observed vs predicted ({model_name})"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir /
        f"{region_name}_timeseries_obs_vs_pred_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def run_loyo_cv_skill_comparison(
    input_file,
    output_dir,
    region_name,
    target_anom,
    target_total,
    baseline_start,
    baseline_end,
    model_config_file,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {input_file}")

    df = pd.read_csv(input_file)

    models = load_model_config(model_config_file)

    years = df["year"].values

    baseline_mask = (
        (df["year"] >= baseline_start) &
        (df["year"] <= baseline_end)
    )

    climatology_total = df.loc[
        baseline_mask,
        target_total,
    ].mean()

    skill_rows = []

    for model_name, predictors in models.items():

        print(f"\n=== Running LOYO CV for {model_name} ===")

        pred_total = []
        obs_total = []

        pred_anom = []
        obs_anom = []

        years_used = []

        for year in years:

            train = df[df["year"] != year].copy()
            test = df[df["year"] == year].copy()

            train, test = clean_train_test_data(
                train=train,
                test=test,
                target_anom=target_anom,
                target_total=target_total,
                predictors=predictors,
            )

            if train.empty:
                raise ValueError(
                    f"No valid training rows for year {year}"
                )

            if test.empty:
                raise ValueError(
                    f"No valid test row for year {year}"
                )

            y_train = train[target_anom].values.astype(float)

            x_train = sm.add_constant(
                train[predictors].astype(float),
                has_constant="add",
            )

            model = sm.GLM(
                y_train,
                x_train,
                family=sm.families.Gaussian(),
            )

            result = model.fit()

            x_test_raw = sm.add_constant(
                test[predictors].astype(float),
                has_constant="add",
            )

            x_test = x_test_raw[
                result.model.exog_names
            ]

            y_pred_anom = result.predict(x_test).iloc[0]

            pred_anom.append(float(y_pred_anom))

            obs_anom.append(
                float(test[target_anom].values[0])
            )

            y_pred_total = (
                y_pred_anom + climatology_total
            )

            pred_total.append(float(y_pred_total))

            obs_total.append(
                float(test[target_total].values[0])
            )

            years_used.append(int(year))

        cv_df = pd.DataFrame(
            {
                "year": years_used,
                "obs_anom": obs_anom,
                "pred_anom": pred_anom,
                "obs_total": obs_total,
                "pred_total": pred_total,
                "baseline_total": climatology_total,
            }
        ).sort_values("year")

        prediction_file = (
            output_dir /
            f"{region_name}_loyo_predictions_{model_name}.csv"
        )

        cv_df.to_csv(prediction_file, index=False)

        metrics = compute_skill_metrics(
            observed=cv_df["obs_total"],
            predicted=cv_df["pred_total"],
            climatology=cv_df["baseline_total"],
        )

        metrics["model"] = model_name
        metrics["predictors"] = "; ".join(predictors)

        skill_rows.append(metrics)

        print(metrics)

        save_scatter_plot(
            cv_df=cv_df,
            output_dir=output_dir,
            region_name=region_name,
            model_name=model_name,
        )

        save_timeseries_plot(
            cv_df=cv_df,
            output_dir=output_dir,
            region_name=region_name,
            model_name=model_name,
        )

    skill_table = pd.DataFrame(skill_rows)

    skill_output = (
        output_dir /
        f"{region_name}_loyo_skill_comparison.csv"
    )

    skill_table.to_csv(skill_output, index=False)

    print("\n=== Skill comparison table ===")
    print(skill_table)

    print(f"\nSaved skill comparison table: {skill_output}")

    return skill_table


def main():
    parser = argparse.ArgumentParser(
        description="Run LOYO GLM skill comparison."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="GLM-ready CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/loyo_skill",
        help="Output directory.",
    )

    parser.add_argument(
        "--region_name",
        required=True,
        help="Region name.",
    )

    parser.add_argument(
        "--target_anom",
        required=True,
        help="Anomaly target column.",
    )

    parser.add_argument(
        "--target_total",
        required=True,
        help="Total rainfall target column.",
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

    parser.add_argument(
        "--model_config_file",
        required=True,
        help="CSV file defining model names and predictors.",
    )

    args = parser.parse_args()

    run_loyo_cv_skill_comparison(
        input_file=args.input_file,
        output_dir=args.output_dir,
        region_name=args.region_name,
        target_anom=args.target_anom,
        target_total=args.target_total,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        model_config_file=args.model_config_file,
    )


if __name__ == "__main__":
    main()