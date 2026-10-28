"""Flask API for House Price Prediction By SRM.

Train the model first with `python train_model.py`, then start this API with:
    python app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_NAME = "House Price Prediction By SRM"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

# Flask's import name stays __name__ so local file/resource lookup works reliably.
# PROJECT_NAME is the user-facing name used by the House Price Prediction By SRM API.
app = Flask(__name__)
app.config["PROJECT_NAME"] = PROJECT_NAME
CORS(app)  # Allows the Stage 2 static frontend to call this API from another local port.

INTEGER_FIELDS = {
    "TotRmsAbvGrd",
    "BedroomAbvGr",
    "YearBuilt",
    "OverallQual",
    "GarageCars",
    "FullBath",
}

# Broad, sanity-checking ranges. These are validation limits, not claims about typical homes.
NUMERIC_LIMITS = {
    "GrLivArea": (1, 50000),
    "TotRmsAbvGrd": (1, 30),
    "BedroomAbvGr": (0, 20),
    "YearBuilt": (1700, 2100),
    "OverallQual": (1, 10),
    "GarageCars": (0, 10),
    "TotalBsmtSF": (0, 50000),
    "FullBath": (0, 10),
}

MODEL_BUNDLE: dict[str, Any] | None = None
MODEL_LOAD_ERROR: str | None = None
METRICS_DATA: dict[str, Any] | None = None
METRICS_LOAD_ERROR: str | None = None


def load_artifacts() -> None:
    """Load training artifacts but keep the API startable when training has not run yet."""
    global MODEL_BUNDLE, MODEL_LOAD_ERROR, METRICS_DATA, METRICS_LOAD_ERROR

    if not MODEL_PATH.is_file():
        MODEL_LOAD_ERROR = (
            f"Trained model was not found at {MODEL_PATH}. "
            "Place Kaggle train.csv in the project folder and run `python train_model.py` first."
        )
    else:
        try:
            MODEL_BUNDLE = joblib.load(MODEL_PATH)
            MODEL_LOAD_ERROR = None
        except Exception as error:
            MODEL_LOAD_ERROR = f"Could not load the trained model: {error}"

    if not METRICS_PATH.is_file():
        METRICS_LOAD_ERROR = (
            f"Metrics file was not found at {METRICS_PATH}. Run `python train_model.py` first."
        )
    else:
        try:
            import json

            with METRICS_PATH.open("r", encoding="utf-8") as file:
                METRICS_DATA = json.load(file)
            METRICS_LOAD_ERROR = None
        except Exception as error:
            METRICS_LOAD_ERROR = f"Could not load the metrics file: {error}"


load_artifacts()


def error_response(message: str, status_code: int):
    """Return a consistently structured API error response."""
    return jsonify({"project_name": PROJECT_NAME, "error": message}), status_code


def validate_payload(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a JSON payload and return a single-row feature dictionary."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object with the required house features."
    if MODEL_BUNDLE is None:
        return None, MODEL_LOAD_ERROR or "The trained model is not available."

    required_fields = MODEL_BUNDLE["feature_columns"]
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        return None, "Missing required field(s): " + ", ".join(missing_fields) + "."

    validated: dict[str, Any] = {}
    for field in MODEL_BUNDLE["numeric_features"]:
        raw_value = payload[field]
        if isinstance(raw_value, bool) or raw_value is None or raw_value == "":
            return None, f"'{field}' must be a valid number."
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None, f"'{field}' must be a valid number."
        if not np.isfinite(value):
            return None, f"'{field}' must be a finite number."

        minimum, maximum = NUMERIC_LIMITS[field]
        if value < minimum or value > maximum:
            return None, f"'{field}' must be between {minimum} and {maximum}."
        if field in INTEGER_FIELDS and not value.is_integer():
            return None, f"'{field}' must be a whole number."
        validated[field] = int(value) if field in INTEGER_FIELDS else value

    neighborhood = payload["Neighborhood"]
    if not isinstance(neighborhood, str) or not neighborhood.strip():
        return None, "'Neighborhood' must be a non-empty string."
    allowed_neighborhoods = MODEL_BUNDLE["neighborhoods"]
    if neighborhood not in allowed_neighborhoods:
        return None, (
            "'Neighborhood' is not recognized. Choose one of the values returned by "
            "GET /metrics in the 'neighborhoods' field."
        )
    validated["Neighborhood"] = neighborhood
    return validated, None


@app.get("/health")
def health():
    """Report API availability without exposing local file contents."""
    model_ready = MODEL_BUNDLE is not None
    metrics_ready = METRICS_DATA is not None
    return jsonify(
        {
            "project_name": PROJECT_NAME,
            "status": "ready" if model_ready else "model_not_trained",
            "model_ready": model_ready,
            "metrics_ready": metrics_ready,
            "message": "API is running." if model_ready else (MODEL_LOAD_ERROR or "Model unavailable."),
        }
    ), (200 if model_ready else 503)


@app.post("/predict")
def predict():
    """Predict SalePrice from a JSON object containing all selected input features."""
    if MODEL_BUNDLE is None:
        return error_response(MODEL_LOAD_ERROR or "The trained model is not available.", 503)

    payload = request.get_json(silent=True)
    if payload is None:
        return error_response("Request body must contain valid JSON with Content-Type: application/json.", 400)

    validated, validation_error = validate_payload(payload)
    if validation_error:
        return error_response(validation_error, 400)

    try:
        input_frame = pd.DataFrame([validated], columns=MODEL_BUNDLE["feature_columns"])
        predicted_log_price = float(MODEL_BUNDLE["pipeline"].predict(input_frame)[0])
        predicted_price = float(np.maximum(0.0, np.expm1(predicted_log_price)))
        if not np.isfinite(predicted_price):
            return error_response("The model returned an invalid prediction. Check the input values.", 500)

        return jsonify(
            {
                "project_name": PROJECT_NAME,
                "predicted_price": round(predicted_price, 2),
                "currency": "USD",
                "model_used": MODEL_BUNDLE["model_name"],
                "message": "Prediction generated successfully.",
            }
        ), 200
    except Exception as error:
        app.logger.exception("Prediction failed for %s", PROJECT_NAME)
        return error_response(f"Prediction failed. Check the input values. Details: {error}", 500)


@app.get("/metrics")
def metrics():
    """Return the real model comparison results produced by train_model.py."""
    if METRICS_DATA is None:
        return error_response(METRICS_LOAD_ERROR or "Metrics are not available.", 503)
    return jsonify(METRICS_DATA), 200


if __name__ == "__main__":
    print(f"Starting {PROJECT_NAME} API on http://127.0.0.1:5000")
    if MODEL_BUNDLE is None:
        print(f"NOTICE: {MODEL_LOAD_ERROR}")
    app.run(host="127.0.0.1", port=5000, debug=False)
