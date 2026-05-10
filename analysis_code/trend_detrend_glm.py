"""
Purpose
-------
1. Linear trend analysis
2. Non-stationarity assessment
3. Detrended GLM sensitivity testing
4. Original vs detrended coefficient comparison
5. LOYO cross-validation skill comparison
6. Region-specific interpretation summaries
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# DISPLAY LABELS
# ============================================================

DISPLAY_NAMES = {
    "R_JJAS": "Seasonal rainfall",
    "ERA5_tp": "ERA5 precipitation",
    "u850": "850-hPa zonal wind",
    "v850": "850-hPa meridional wind",
    "q850": "850-hPa specific humidity",
    "tcwv": "Total column water vapor",
    "t2m": "2-m temperature",
    "msl": "Mean sea-level pressure",
    "nino34": "Niño-3.4",
    "dmi": "Dipole Mode Index",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def significance_marker(p_value):

    if pd.isna(p_value):
        return ""

    if p_value < 0.05:
        return "*"

    if p_value < 0.10:
        return "°"

    return ""


def standardize_full(x):

    mean = np.nanmean(x, axis=0)

    std = np.nanstd(x, axis=0, ddof=1)

    std = np.where(std == 0, 1.0, std)

    return (x - mean) / std


def standardize_train_test(x_train, x_test):

    mean = np.nanmean(x_train, axis=0)

    std = np.nanstd(x_train, axis=0, ddof=1)

    std = np.where(std == 0, 1.0, std)

    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std

    return x_train_z, x_test_z


def linear_detrend(y, year):

    x = sm.add_constant(year)

    model = sm.OLS(y, x, missing="drop").fit()

    fitted = model.predict(x)

    residual = y - fitted

    return residual


def parse_semicolon_list(text):

    return [
        item.strip()
        for item in str(text).split(";")
        if item.strip()
    ]


# ============================================================
# CONFIG LOADING
# ============================================================

def load_config(config_file):

    config_df = pd.read_csv(config_file)

    required_columns = [
        "region",
        "file",
        "target",
        "predictors",
        "trend_variables",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in config_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing config columns: {missing_columns}"
        )

    return config_df


# ============================================================
# TREND ANALYSIS
# ============================================================

def compute_linear_trends(
    region,
    df,
    trend_variables,
):

    rows = []

    year = df["year"].values.astype(float)

    for variable in trend_variables:

        if variable not in df.columns:
            continue

        y = pd.to_numeric(
            df[variable],
            errors="coerce",
        ).values.astype(float)

        valid = np.isfinite(y)

        if valid.sum() < 10:
            continue

        x = sm.add_constant(year[valid])

        model = sm.OLS(
            y[valid],
            x,
        ).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 1},
        )

        slope = model.params[1]
        slope_decade = slope * 10.0

        ci = model.conf_int(alpha=0.05)

        ci_low = ci[1, 0] * 10.0
        ci_high = ci[1, 1] * 10.0

        rows.append(
            {
                "Region": region,
                "Variable": variable,
                "Display_name": DISPLAY_NAMES.get(variable, variable),
                "Trend_per_decade": slope_decade,
                "SE_per_decade": model.bse[1] * 10.0,
                "CI95_lower": ci_low,
                "CI95_upper": ci_high,
                "p_value": model.pvalues[1],
                "Significance": significance_marker(
                    model.pvalues[1]
                ),
                "N_years": valid.sum(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# GLM FITTING
# ============================================================

def fit_glm_coefficients(
    region,
    df,
    target,
    predictors,
    detrended=False,
):

    work = df.copy()

    required_columns = ["year", target] + predictors

    work = work[required_columns].copy()

    for column in required_columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    year = work["year"].values.astype(float)

    y = work[target].values.astype(float)

    x = work[predictors].values.astype(float)

    if detrended:

        y = linear_detrend(y, year)

        detrended_predictors = []

        for predictor_index in range(x.shape[1]):

            detrended_predictors.append(
                linear_detrend(
                    x[:, predictor_index],
                    year,
                )
            )

        x = np.column_stack(
            detrended_predictors
        )

    x_standardized = standardize_full(x)

    x_design = sm.add_constant(
        x_standardized,
        has_constant="add",
    )

    model = sm.GLM(
        y,
        x_design,
        family=sm.families.Gaussian(),
    ).fit()

    rows = []

    conf_int = model.conf_int()

    for predictor_index, predictor in enumerate(predictors):

        rows.append(
            {
                "Region": region,
                "Model_version":
                    "Detrended M2"
                    if detrended
                    else "Non-detrended M2",

                "Predictor": predictor,

                "Display_name":
                    DISPLAY_NAMES.get(
                        predictor,
                        predictor,
                    ),

                "Coefficient":
                    model.params[predictor_index + 1],

                "Standard_error":
                    model.bse[predictor_index + 1],

                "z_or_t_value":
                    model.tvalues[predictor_index + 1],

                "p_value":
                    model.pvalues[predictor_index + 1],

                "CI95_lower":
                    conf_int[predictor_index + 1, 0],

                "CI95_upper":
                    conf_int[predictor_index + 1, 1],

                "Significance":
                    significance_marker(
                        model.pvalues[predictor_index + 1]
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LOYO CROSS-VALIDATION
# ============================================================

def loyo_skill(
    region,
    df,
    target,
    predictors,
    detrended=False,
):

    work = df.copy()

    required_columns = ["year", target] + predictors

    work = work[required_columns].copy()

    for column in required_columns:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work = work.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    years = work["year"].values.astype(int)

    y_original = work[target].values.astype(float)

    x_original = work[predictors].values.astype(float)

    if detrended:

        y_all = linear_detrend(
            y_original,
            years.astype(float),
        )

        detrended_predictors = []

        for predictor_index in range(x_original.shape[1]):

            detrended_predictors.append(
                linear_detrend(
                    x_original[:, predictor_index],
                    years.astype(float),
                )
            )

        x_all = np.column_stack(
            detrended_predictors
        )

    else:

        y_all = y_original.copy()

        x_all = x_original.copy()

    predictions = []
    observations = []
    climatology_predictions = []

    for test_year in years:

        train_mask = years != test_year
        test_mask = years == test_year

        y_train = y_all[train_mask]
        y_test = y_all[test_mask]

        x_train = x_all[train_mask]
        x_test = x_all[test_mask]

        x_train_z, x_test_z = standardize_train_test(
            x_train,
            x_test,
        )

        x_train_design = sm.add_constant(
            x_train_z,
            has_constant="add",
        )

        x_test_design = sm.add_constant(
            x_test_z,
            has_constant="add",
        )

        model = sm.GLM(
            y_train,
            x_train_design,
            family=sm.families.Gaussian(),
        ).fit()

        prediction = model.predict(
            x_test_design
        )[0]

        climatology = np.nanmean(y_train)

        predictions.append(prediction)
        observations.append(y_test[0])
        climatology_predictions.append(climatology)

    predictions = np.array(predictions)
    observations = np.array(observations)
    climatology_predictions = np.array(
        climatology_predictions
    )

    rmse_model = np.sqrt(
        mean_squared_error(
            observations,
            predictions,
        )
    )

    rmse_climatology = np.sqrt(
        mean_squared_error(
            observations,
            climatology_predictions,
        )
    )

    rmse_skill = (
        1.0 -
        (rmse_model / rmse_climatology)
    )

    return {
        "Region": region,
        "Model_version":
            "Detrended M2"
            if detrended
            else "Non-detrended M2",

        "CV_R2":
            r2_score(
                observations,
                predictions,
            ),

        "Correlation":
            np.corrcoef(
                observations,
                predictions,
            )[0, 1],

        "RMSE":
            rmse_model,

        "MAE":
            mean_absolute_error(
                observations,
                predictions,
            ),

        "Climatology_RMSE":
            rmse_climatology,

        "RMSE_skill":
            rmse_skill,

        "N_years":
            len(observations),
    }


# ============================================================
# COEFFICIENT COMPARISON
# ============================================================

def build_coefficient_comparison(coef_df):

    non = coef_df[
        coef_df["Model_version"] ==
        "Non-detrended M2"
    ].copy()

    det = coef_df[
        coef_df["Model_version"] ==
        "Detrended M2"
    ].copy()

    non = non.rename(
        columns={
            "Coefficient":
                "Beta_non_detrended",

            "p_value":
                "p_non_detrended",
        }
    )

    det = det.rename(
        columns={
            "Coefficient":
                "Beta_detrended",

            "p_value":
                "p_detrended",
        }
    )

    merged = pd.merge(
        non,
        det,
        on=[
            "Region",
            "Predictor",
        ],
        how="outer",
        suffixes=("_non", "_det"),
    )

    merged["Sign_preserved"] = np.where(
        np.sign(
            merged["Beta_non_detrended"]
        ) ==
        np.sign(
            merged["Beta_detrended"]
        ),
        "Yes",
        "No",
    )

    merged["Significance_preserved"] = np.where(
        (
            merged["p_non_detrended"] < 0.05
        ) ==
        (
            merged["p_detrended"] < 0.05
        ),
        "Yes",
        "No",
    )

    return merged


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Trend and detrended GLM sensitivity analysis."
        )
    )

    parser.add_argument(
        "--config_file",
        required=True,
        help="Region configuration CSV file.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/trend_detrended_analysis",
        help="Directory for outputs.",
    )

    parser.add_argument(
        "--start_year",
        type=int,
        default=1981,
        help="Analysis start year.",
    )

    parser.add_argument(
        "--end_year",
        type=int,
        default=2020,
        help="Analysis end year.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_df = load_config(
        args.config_file
    )

    all_trends = []
    all_coefficients = []
    all_skills = []

    for _, row in config_df.iterrows():

        region = row["region"]

        input_file = row["file"]

        target = row["target"]

        predictors = parse_semicolon_list(
            row["predictors"]
        )

        trend_variables = parse_semicolon_list(
            row["trend_variables"]
        )

        print(f"\n=== Processing region: {region} ===")

        df = pd.read_csv(input_file)

        df = df[
            (
                df["year"] >= args.start_year
            ) &
            (
                df["year"] <= args.end_year
            )
        ].copy()

        trend_df = compute_linear_trends(
            region=region,
            df=df,
            trend_variables=trend_variables,
        )

        all_trends.append(trend_df)

        coef_non = fit_glm_coefficients(
            region=region,
            df=df,
            target=target,
            predictors=predictors,
            detrended=False,
        )

        coef_det = fit_glm_coefficients(
            region=region,
            df=df,
            target=target,
            predictors=predictors,
            detrended=True,
        )

        all_coefficients.extend(
            [coef_non, coef_det]
        )

        skill_non = loyo_skill(
            region=region,
            df=df,
            target=target,
            predictors=predictors,
            detrended=False,
        )

        skill_det = loyo_skill(
            region=region,
            df=df,
            target=target,
            predictors=predictors,
            detrended=True,
        )

        all_skills.extend(
            [skill_non, skill_det]
        )

    trend_output = pd.concat(
        all_trends,
        ignore_index=True,
    )

    coefficient_output = pd.concat(
        all_coefficients,
        ignore_index=True,
    )

    skill_output = pd.DataFrame(
        all_skills
    )

    coefficient_comparison = (
        build_coefficient_comparison(
            coefficient_output
        )
    )

    trend_csv = (
        output_dir /
        "table_A_linear_trends.csv"
    )

    coefficient_csv = (
        output_dir /
        "table_B_coefficients.csv"
    )

    coefficient_comparison_csv = (
        output_dir /
        "table_B_comparison.csv"
    )

    skill_csv = (
        output_dir /
        "table_C_skill_comparison.csv"
    )

    trend_output.to_csv(
        trend_csv,
        index=False,
    )

    coefficient_output.to_csv(
        coefficient_csv,
        index=False,
    )

    coefficient_comparison.to_csv(
        coefficient_comparison_csv,
        index=False,
    )

    skill_output.to_csv(
        skill_csv,
        index=False,
    )

    excel_output = (
        output_dir /
        "trend_detrended_glm_results.xlsx"
    )

    with pd.ExcelWriter(
        excel_output,
        engine="openpyxl",
    ) as writer:

        trend_output.to_excel(
            writer,
            sheet_name="Linear_trends",
            index=False,
        )

        coefficient_output.to_excel(
            writer,
            sheet_name="GLM_coefficients",
            index=False,
        )

        coefficient_comparison.to_excel(
            writer,
            sheet_name="Coefficient_comparison",
            index=False,
        )

        skill_output.to_excel(
            writer,
            sheet_name="LOYO_skill",
            index=False,
        )

    print("\n======================================")
    print("Trend and detrended analysis complete")
    print("======================================")

    print(f"Saved: {trend_csv}")
    print(f"Saved: {coefficient_csv}")
    print(f"Saved: {coefficient_comparison_csv}")
    print(f"Saved: {skill_csv}")
    print(f"Saved: {excel_output}")


if __name__ == "__main__":
    main()