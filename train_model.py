"""Train the House Price Prediction By SRM regression models.

Place Kaggle's train.csv in this project folder or in the data/ folder, then run:
    python train_model.py

The script fits Linear Regression and Random Forest Regressor models on log1p(SalePrice),
evaluates predictions in the original SalePrice units, and saves artifacts under outputs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots without requiring a graphical desktop.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

PROJECT_NAME = "House Price Prediction By SRM"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Core requested features plus four high-impact additions:
# OverallQual captures overall material/finish quality; GarageCars captures garage capacity;
# TotalBsmtSF captures basement area; FullBath captures full bathrooms.
NUMERIC_FEATURES = [
    "GrLivArea",
    "TotRmsAbvGrd",
    "BedroomAbvGr",
    "YearBuilt",
    "OverallQual",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
]
CATEGORICAL_FEATURES = ["Neighborhood"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "SalePrice"


def find_dataset() -> Path:
    """Find train.csv in either of the two documented locations."""
    candidates = [BASE_DIR / "train.csv", BASE_DIR / "data" / "train.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    expected = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find Kaggle train.csv for {PROJECT_NAME}.\n"
        f"Download the 'House Prices: Advanced Regression Techniques' training file and place it at:\n"
        f"{expected}\n"
        "Then run: python train_model.py"
    )


def make_preprocessor() -> ColumnTransformer:
    """Build missing-value handling and categorical encoding for each model."""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(regressor) -> Pipeline:
    """Create the common preprocessing + regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("regressor", regressor),
        ]
    )


def save_price_distribution(data: pd.DataFrame) -> None:
    """Save the SalePrice distribution before and after the log transformation."""
    prices = data[TARGET_COLUMN].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(prices, bins=35, alpha=0.8, edgecolor="black", linewidth=0.4)
    axes[0].set_title("SalePrice Distribution")
    axes[0].set_xlabel("SalePrice (dataset currency; treated as USD)")
    axes[0].set_ylabel("Number of houses")

    axes[1].hist(np.log1p(prices), bins=35, alpha=0.8, edgecolor="black", linewidth=0.4)
    axes[1].set_title("log1p(SalePrice) Distribution")
    axes[1].set_xlabel("log1p(SalePrice)")
    axes[1].set_ylabel("Number of houses")

    fig.suptitle(f"{PROJECT_NAME}: Price Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "price_distribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(data: pd.DataFrame) -> None:
    """Save a readable correlation heatmap for selected numeric features and target."""
    columns = NUMERIC_FEATURES + [TARGET_COLUMN]
    correlation = data[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title(f"{PROJECT_NAME}: Selected-Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    """Save actual versus predicted prices for the selected best model."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolors="none")
    minimum = float(min(np.min(y_true), np.min(y_pred)))
    maximum = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([minimum, maximum], [minimum, maximum], linestyle="--", linewidth=1.5)
    ax.set_xlabel("Actual SalePrice ($)")
    ax.set_ylabel("Predicted SalePrice ($)")
    ax.set_title(f"{PROJECT_NAME}: Actual vs Predicted\nBest model: {model_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_random_forest_feature_importance(rf_pipeline: Pipeline) -> None:
    """Save Random Forest feature importances after preprocessing/one-hot encoding."""
    fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
    fitted_regressor = rf_pipeline.named_steps["regressor"]
    names = fitted_preprocessor.get_feature_names_out()
    importances = fitted_regressor.feature_importances_

    importance_df = pd.DataFrame({"feature": names, "importance": importances})
    importance_df["feature"] = importance_df["feature"].str.replace(
        r"^(num|cat)__", "", regex=True
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    top_features = importance_df.head(20).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_xlabel("Random Forest feature importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"{PROJECT_NAME}: Random Forest Feature Importance (Top 20)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Load the Kaggle data, train models, evaluate, and save the best model."""
    try:
        data_path = find_dataset()
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"\n=== {PROJECT_NAME} ===")
    print(f"Loading dataset: {data_path}")
    try:
        data = pd.read_csv(data_path)
    except Exception as error:  # Provide a useful message for a bad or unreadable CSV.
        print(f"Could not read {data_path}: {error}", file=sys.stderr)
        return 1

    print("\n--- Exploratory Data Analysis Summary ---")
    print(f"Original dataset shape (rows, columns): {data.shape}")

    if TARGET_COLUMN not in data.columns:
        print(
            f"The dataset must contain target column '{TARGET_COLUMN}'. "
            "Please use Kaggle's train.csv, not test.csv.",
            file=sys.stderr,
        )
        return 1

    missing_counts = data.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if missing_counts.empty:
        print("Missing values: none")
    else:
        print("Columns with missing values (top 20):")
        print(missing_counts.head(20).to_string())
        if len(missing_counts) > 20:
            print(f"... plus {len(missing_counts) - 20} other columns with missing values")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        print(
            "Required columns missing from train.csv: " + ", ".join(missing_columns),
            file=sys.stderr,
        )
        return 1

    # Convert numeric inputs safely; non-numeric entries become NaN and are imputed in-pipeline.
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    rows_before_target_cleanup = len(data)
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    data = data[data[TARGET_COLUMN] > 0].copy()
    rows_removed = rows_before_target_cleanup - len(data)
    if rows_removed:
        print(f"Rows removed because SalePrice was missing or non-positive: {rows_removed}")
    if data.empty:
        print("No rows with a positive SalePrice remain; training cannot continue.", file=sys.stderr)
        return 1

    target_skew = float(data[TARGET_COLUMN].skew())
    print(f"Rows available after target cleanup: {len(data)}")
    print(f"SalePrice skewness before log transform: {target_skew:.4f}")
    print(f"Selected input features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")
    print(
        "Additional features used: OverallQual (quality), GarageCars (garage capacity), "
        "TotalBsmtSF (basement area), and FullBath (full bathrooms)."
    )

    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].astype(float).copy()

    # The test split is held out before fitting any preprocessing/model steps.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
    y_train_log = np.log1p(y_train)

    models = {
        "Linear Regression": make_pipeline(LinearRegression()),
        "Random Forest Regressor": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    comparison_rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    for model_name, pipeline in models.items():
        print(f"\nTraining {model_name}...")
        pipeline.fit(X_train, y_train_log)
        # Models predict log1p(SalePrice); expm1 converts predictions back to dollar units.
        predicted_log = pipeline.predict(X_test)
        predicted_prices = np.maximum(0.0, np.expm1(predicted_log))
        predictions[model_name] = predicted_prices

        mae = float(mean_absolute_error(y_test, predicted_prices))
        rmse = float(np.sqrt(mean_squared_error(y_test, predicted_prices)))
        r2 = float(r2_score(y_test, predicted_prices))
        comparison_rows.append(
            {
                "model": model_name,
                "mae_dollars": mae,
                "rmse_dollars": rmse,
                "r2": r2,
            }
        )

    comparison_rows.sort(key=lambda row: float(row["rmse_dollars"]))
    best_result = comparison_rows[0]
    best_model_name = str(best_result["model"])
    best_pipeline = models[best_model_name]

    print("\n--- Model Comparison (errors in SalePrice dollars) ---")
    display_rows = [
        {
            "Model": row["model"],
            "MAE ($)": f"${float(row['mae_dollars']):,.2f}",
            "RMSE ($)": f"${float(row['rmse_dollars']):,.2f}",
            "R²": f"{float(row['r2']):.4f}",
        }
        for row in comparison_rows
    ]
    print(pd.DataFrame(display_rows).to_string(index=False))
    print(f"\nSelected best model by lowest test RMSE: {best_model_name}")

    # Preserve the observed Neighborhood choices for validation and the Stage 2 dropdown.
    neighborhoods = sorted(
        data["Neighborhood"].dropna().astype(str).unique().tolist()
    )
    if not neighborhoods:
        print("No non-missing Neighborhood values were found.", file=sys.stderr)
        return 1

    model_bundle = {
        "project_name": PROJECT_NAME,
        "model_name": best_model_name,
        "pipeline": best_pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "neighborhoods": neighborhoods,
        "target_column": TARGET_COLUMN,
        "target_transform": "log1p; convert model predictions with expm1",
    }
    joblib.dump(model_bundle, OUTPUT_DIR / "best_model.joblib")

    metrics_payload = {
        "project_name": PROJECT_NAME,
        "dataset": "Kaggle: House Prices - Advanced Regression Techniques (train.csv)",
        "target": TARGET_COLUMN,
        "target_transform": "log1p during training; expm1 for predictions and evaluation",
        "currency_note": "SalePrice and error metrics are reported in dollars ($).",
        "split": {"test_size": 0.20, "random_state": 42},
        "best_model": best_model_name,
        "features": FEATURE_COLUMNS,
        "neighborhoods": neighborhoods,
        "metrics": comparison_rows,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    with (OUTPUT_DIR / "neighborhoods.json").open("w", encoding="utf-8") as file:
        json.dump(neighborhoods, file, indent=2)

    # Save the requested EDA and model diagnostic plots.
    save_price_distribution(data)
    save_correlation_heatmap(data)
    save_actual_vs_predicted(y_test, predictions[best_model_name], best_model_name)
    save_random_forest_feature_importance(models["Random Forest Regressor"])

    print(f"\nArtifacts saved to: {OUTPUT_DIR}")
    print("  - best_model.joblib")
    print("  - metrics.json")
    print("  - neighborhoods.json")
    print("  - price_distribution.png")
    print("  - correlation_heatmap.png")
    print("  - actual_vs_predicted.png")
    print("  - feature_importance.png and feature_importance.csv")
    print(f"\n{PROJECT_NAME} training completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
