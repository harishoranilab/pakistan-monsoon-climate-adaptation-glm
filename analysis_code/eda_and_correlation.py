
"""

Purpose
-------
Run basic exploratory data analysis for a GLM-ready seasonal climate dataset.

Outputs
-------
1. Summary statistics CSV
2. Correlation matrix CSV
3. Correlation heatmap PNG
4. Scatter plots between target and selected predictors

"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def run_eda_and_correlation(
    input_file,
    output_dir,
    target_column,
    predictors,
    figure_prefix,
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

    print("\nDataset columns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())

    summary = df[required_columns].describe()
    summary_file = output_dir / f"{figure_prefix}_summary_statistics.csv"
    summary.to_csv(summary_file)

    print(f"\nSaved summary statistics: {summary_file}")

    corr = df[required_columns].corr()
    corr_file = output_dir / f"{figure_prefix}_correlation_matrix.csv"
    corr.to_csv(corr_file)

    print(f"Saved correlation matrix: {corr_file}")
    print(corr)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, interpolation="nearest")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation coefficient")

    ax.set_title("Correlation matrix")

    fig.tight_layout()

    heatmap_file = output_dir / f"{figure_prefix}_correlation_heatmap.png"
    fig.savefig(heatmap_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap: {heatmap_file}")

    for predictor in predictors:
        fig, ax = plt.subplots(figsize=(6, 5))

        ax.scatter(df[predictor], df[target_column])

        ax.set_xlabel(predictor)
        ax.set_ylabel(target_column)
        ax.set_title(f"{target_column} vs {predictor}")

        fig.tight_layout()

        safe_predictor_name = predictor.replace("/", "_").replace(" ", "_")
        scatter_file = output_dir / f"{figure_prefix}_scatter_{target_column}_vs_{safe_predictor_name}.png"

        fig.savefig(scatter_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("Saved scatter plots.")


def main():
    parser = argparse.ArgumentParser(
        description="Run EDA, correlation analysis, and scatter plots for GLM climate data."
    )

    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to GLM-ready CSV dataset.",
    )

    parser.add_argument(
        "--output_dir",
        default="outputs/eda",
        help="Directory for EDA outputs.",
    )

    parser.add_argument(
        "--target_column",
        required=True,
        help="Target/predictand column name.",
    )

    parser.add_argument(
        "--predictors",
        nargs="+",
        required=True,
        help="Predictor column names.",
    )

    parser.add_argument(
        "--figure_prefix",
        default="glm_eda",
        help="Prefix for output files.",
    )

    args = parser.parse_args()

    run_eda_and_correlation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        target_column=args.target_column,
        predictors=args.predictors,
        figure_prefix=args.figure_prefix,
    )


if __name__ == "__main__":
    main()