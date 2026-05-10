"""
Purpose
-------
Bootstrap evaluation of GLM predictive skill and model improvement.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    """
    Root mean square error.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def r2_score_manual(y_true, y_pred):
    """
    Manual R² calculation.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (
            y_true -
            np.mean(y_true)
        ) ** 2
    )

    if ss_tot == 0:
        return np.nan

    return float(
        1 - (ss_res / ss_tot)
    )


def find_column(df, possible_names):
    """
    Find first matching column.
    """

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_map:

            return lower_map[name.lower()]

    return None


def detect_prediction_columns(df):
    """
    Automatically detect required columns.
    """

    year_col = find_column(
        df,
        [
            "year",
            "Year",
        ],
    )

    if year_col is None:
        raise ValueError(
            "Could not detect year column."
        )

    obs_col = find_column(
        df,
        [
            "obs_total",
            "observed_total",
            "observed",
            "obs",
            "actual",
            "obs_anom",
            "observed_anom",
        ],
    )

    pred_col = find_column(
        df,
        [
            "pred_total",
            "predicted_total",
            "predicted",
            "prediction",
            "pred",
            "pred_anom",
            "predicted_anom",
            "yhat",
        ],
    )

    if obs_col is None:
        raise ValueError(
            f"Could not detect observed column.\n"
            f"Columns: {df.columns.tolist()}"
        )

    if pred_col is None:
        raise ValueError(
            f"Could not detect predicted column.\n"
            f"Columns: {df.columns.tolist()}"
        )

    return year_col, obs_col, pred_col


def load_prediction_file(path):
    """
    Load and standardize prediction file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    print(f"\nReading file: {path}")

    year_col, obs_col, pred_col = (
        detect_prediction_columns(df)
    )

    out = df[
        [year_col, obs_col, pred_col]
    ].copy()

    out.columns = [
        "year",
        "observed",
        "predicted",
    ]

    for column in out.columns:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    out = out.dropna()

    out = out.sort_values(
        "year"
    ).reset_index(drop=True)

    if out.empty:
        raise ValueError(
            f"No usable rows after cleaning: {path}"
        )

    return out


def get_climatology_prediction(
    df,
    climatology_start,
    climatology_end,
):
    """
    Generate climatology benchmark prediction.
    """

    mask = (
        (
            df["year"] >= climatology_start
        ) &
        (
            df["year"] <= climatology_end
        )
    )

    if mask.sum() >= 5:

        climatology_mean = df.loc[
            mask,
            "observed",
        ].mean()

        climatology_source = (
            f"{climatology_start}"
            f"–"
            f"{climatology_end} mean"
        )

    else:

        climatology_mean = (
            df["observed"].mean()
        )

        climatology_source = (
            "full-period mean"
        )

    climatology_prediction = np.full(
        len(df),
        climatology_mean,
    )

    return (
        climatology_prediction,
        climatology_mean,
        climatology_source,
    )


# ============================================================
# BOOTSTRAP: SKILL VS CLIMATOLOGY
# ============================================================

def bootstrap_skill_vs_climatology(
    df,
    n_boot,
    climatology_start,
    climatology_end,
    rng,
):
    """
    Bootstrap RMSE skill against climatology.
    """

    y = df["observed"].to_numpy()

    yhat = df["predicted"].to_numpy()

    (
        yclim,
        climatology_mean,
        climatology_source,
    ) = get_climatology_prediction(
        df=df,
        climatology_start=climatology_start,
        climatology_end=climatology_end,
    )

    n = len(y)

    rmse_model = rmse(y, yhat)

    rmse_climatology = rmse(
        y,
        yclim,
    )

    rmse_skill = (
        1 -
        (
            rmse_model /
            rmse_climatology
        )
    )

    r2_model = r2_score_manual(
        y,
        yhat,
    )

    bootstrap_skill = []

    for _ in range(n_boot):

        index = rng.integers(
            0,
            n,
            size=n,
        )

        y_b = y[index]

        yhat_b = yhat[index]

        yclim_b = yclim[index]

        rmse_model_b = rmse(
            y_b,
            yhat_b,
        )

        rmse_climatology_b = rmse(
            y_b,
            yclim_b,
        )

        if rmse_climatology_b == 0:
            continue

        bootstrap_skill.append(
            1 -
            (
                rmse_model_b /
                rmse_climatology_b
            )
        )

    bootstrap_skill = np.asarray(
        bootstrap_skill
    )

    ci_low = np.percentile(
        bootstrap_skill,
        2.5,
    )

    ci_high = np.percentile(
        bootstrap_skill,
        97.5,
    )

    p_boot = np.mean(
        bootstrap_skill <= 0
    )

    return {
        "n_years":
            n,

        "climatology_mean":
            climatology_mean,

        "climatology_source":
            climatology_source,

        "RMSE_model":
            rmse_model,

        "RMSE_climatology":
            rmse_climatology,

        "RMSE_skill":
            rmse_skill,

        "RMSE_skill_CI_2.5":
            ci_low,

        "RMSE_skill_CI_97.5":
            ci_high,

        "bootstrap_p_skill_le_0":
            p_boot,

        "significant_skill_vs_climatology":
            "Yes"
            if ci_low > 0
            else "No",

        "R2_model":
            r2_model,
    }


# ============================================================
# BOOTSTRAP: MODEL 2 VS MODEL 1
# ============================================================

def bootstrap_model_comparison(
    df_model1,
    df_model2,
    n_boot,
    rng,
):
    """
    Bootstrap added value of Model 2 over Model 1.
    """

    merged = pd.merge(
        df_model1[
            [
                "year",
                "observed",
                "predicted",
            ]
        ],
        df_model2[
            [
                "year",
                "observed",
                "predicted",
            ]
        ],
        on="year",
        suffixes=(
            "_Model1",
            "_Model2",
        ),
    )

    if merged.empty:
        raise ValueError(
            "No overlapping years."
        )

    y = merged[
        "observed_Model2"
    ].to_numpy()

    yhat_model1 = merged[
        "predicted_Model1"
    ].to_numpy()

    yhat_model2 = merged[
        "predicted_Model2"
    ].to_numpy()

    n = len(y)

    rmse_model1 = rmse(
        y,
        yhat_model1,
    )

    rmse_model2 = rmse(
        y,
        yhat_model2,
    )

    delta_rmse = (
        rmse_model1 -
        rmse_model2
    )

    bootstrap_delta = []

    for _ in range(n_boot):

        index = rng.integers(
            0,
            n,
            size=n,
        )

        y_b = y[index]

        model1_b = yhat_model1[index]

        model2_b = yhat_model2[index]

        bootstrap_delta.append(
            rmse(
                y_b,
                model1_b,
            ) -
            rmse(
                y_b,
                model2_b,
            )
        )

    bootstrap_delta = np.asarray(
        bootstrap_delta
    )

    ci_low = np.percentile(
        bootstrap_delta,
        2.5,
    )

    ci_high = np.percentile(
        bootstrap_delta,
        97.5,
    )

    p_boot = np.mean(
        bootstrap_delta <= 0
    )

    return {
        "RMSE_Model1":
            rmse_model1,

        "RMSE_Model2":
            rmse_model2,

        "Delta_RMSE_Model1_minus_Model2":
            delta_rmse,

        "Delta_RMSE_CI_2.5":
            ci_low,

        "Delta_RMSE_CI_97.5":
            ci_high,

        "bootstrap_p_Model2_not_better":
            p_boot,

        "significant_Model2_improvement":
            "Yes"
            if ci_low > 0
            else "No",
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_bootstrap_skill_analysis(
    config_file,
    output_dir,
    n_boot,
    random_seed,
    climatology_start,
    climatology_end,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = pd.read_csv(
        config_file
    )

    required_columns = [
        "region",
        "model1_file",
        "model2_file",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in config.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing config columns: "
            f"{missing_columns}"
        )

    rng = np.random.default_rng(
        random_seed
    )

    skill_rows = []

    comparison_rows = []

    for _, row in config.iterrows():

        region = row["region"]

        print("\n" + "=" * 70)
        print(f"Processing region: {region}")
        print("=" * 70)

        model1_file = row["model1_file"]

        model2_file = row["model2_file"]

        df_model1 = load_prediction_file(
            model1_file
        )

        df_model2 = load_prediction_file(
            model2_file
        )

        result_model1 = (
            bootstrap_skill_vs_climatology(
                df=df_model1,
                n_boot=n_boot,
                climatology_start=climatology_start,
                climatology_end=climatology_end,
                rng=rng,
            )
        )

        result_model1["region"] = region
        result_model1["model"] = "Model1"

        skill_rows.append(
            result_model1
        )

        result_model2 = (
            bootstrap_skill_vs_climatology(
                df=df_model2,
                n_boot=n_boot,
                climatology_start=climatology_start,
                climatology_end=climatology_end,
                rng=rng,
            )
        )

        result_model2["region"] = region
        result_model2["model"] = "Model2"

        skill_rows.append(
            result_model2
        )

        comparison_result = (
            bootstrap_model_comparison(
                df_model1=df_model1,
                df_model2=df_model2,
                n_boot=n_boot,
                rng=rng,
            )
        )

        comparison_result["region"] = (
            region
        )

        comparison_rows.append(
            comparison_result
        )

    skill_df = pd.DataFrame(
        skill_rows
    )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # ========================================================
    # ROUNDING
    # ========================================================

    skill_df_rounded = skill_df.copy()

    comparison_df_rounded = (
        comparison_df.copy()
    )

    for column in skill_df_rounded.select_dtypes(
        include=[np.number]
    ).columns:

        if column != "n_years":

            skill_df_rounded[column] = (
                skill_df_rounded[column].round(4)
            )

    for column in comparison_df_rounded.select_dtypes(
        include=[np.number]
    ).columns:

        comparison_df_rounded[column] = (
            comparison_df_rounded[column].round(4)
        )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    skill_csv = (
        output_dir /
        "bootstrap_skill_vs_climatology.csv"
    )

    comparison_csv = (
        output_dir /
        "bootstrap_model_comparison.csv"
    )

    excel_output = (
        output_dir /
        "bootstrap_skill_analysis.xlsx"
    )

    skill_df_rounded.to_csv(
        skill_csv,
        index=False,
    )

    comparison_df_rounded.to_csv(
        comparison_csv,
        index=False,
    )

    method_note = pd.DataFrame(
        {
            "Item": [
                "Bootstrap resamples",
                "Random seed",
                "Primary skill metric",
                "Skill interpretation",
                "Significance criterion",
                "Added-value metric",
            ],

            "Value": [
                n_boot,
                random_seed,
                "RMSE skill",
                (
                    "Positive RMSE skill "
                    "indicates improvement "
                    "over climatology"
                ),
                (
                    "95% bootstrap CI "
                    "excludes zero"
                ),
                (
                    "Delta RMSE = "
                    "RMSE_Model1 - "
                    "RMSE_Model2"
                ),
            ],
        }
    )

    with pd.ExcelWriter(
        excel_output,
        engine="openpyxl",
    ) as writer:

        skill_df_rounded.to_excel(
            writer,
            sheet_name="Skill_vs_climatology",
            index=False,
        )

        comparison_df_rounded.to_excel(
            writer,
            sheet_name="Model_comparison",
            index=False,
        )

        method_note.to_excel(
            writer,
            sheet_name="Method_note",
            index=False,
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n================ "
        "BOOTSTRAP SKILL VS CLIMATOLOGY "
        "================"
    )

    print(
        skill_df_rounded.to_string(
            index=False
        )
    )

    print(
        "\n================ "
        "BOOTSTRAP MODEL COMPARISON "
        "================"
    )

    print(
        comparison_df_rounded.to_string(
            index=False
        )
    )

    print("\nSaved files:")

    print(skill_csv)
    print(comparison_csv)
    print(excel_output)

    return (
        skill_df_rounded,
        comparison_df_rounded,
    )


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap RMSE skill analysis and "
            "model comparison."
        )
    )

    parser.add_argument(
        "--config_file",
        required=True,
        help=(
            "CSV file containing "
            "region, model1_file, "
            "and model2_file."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/bootstrap_skill",
        help="Output directory.",
    )

    parser.add_argument(
        "--n_boot",
        type=int,
        default=10000,
        help="Number of bootstrap resamples.",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--climatology_start",
        type=int,
        default=1981,
        help="Climatology start year.",
    )

    parser.add_argument(
        "--climatology_end",
        type=int,
        default=2010,
        help="Climatology end year.",
    )

    args = parser.parse_args()

    run_bootstrap_skill_analysis(
        config_file=args.config_file,
        output_dir=args.output_dir,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
        climatology_start=args.climatology_start,
        climatology_end=args.climatology_end,
    )


if __name__ == "__main__":
    main()