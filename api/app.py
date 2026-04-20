"""
Lightweight Flask API backed by a precomputed lookup table.

The GBT model has only 4 discrete inputs (hour, day, month, region), so every
possible prediction is materialized ahead of time into models/predictions.json.
This lets the API run on tiny hardware (e.g. Render's free tier) with no
PySpark / JVM dependency and sub-millisecond response times.

To regenerate predictions.json after retraining the model, run:
    python scripts/precompute.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREDICTIONS_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "models", "predictions.json")
)


def _load_predictions() -> dict[str, Any]:
    if not os.path.isfile(_PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"predictions.json not found at {_PREDICTIONS_PATH}. "
            "Run `python scripts/precompute.py` locally (with the Spark model) "
            "and commit the generated file."
        )
    with open(_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "predictions" not in payload:
        raise ValueError("predictions.json is malformed (missing 'predictions' key).")
    return payload


_PAYLOAD = _load_predictions()
_LOOKUP: dict[str, float] = _PAYLOAD["predictions"]
_REGIONS: list[str] = sorted(_PAYLOAD.get("regions", []))
_REGION_INDEX: dict[str, int] = dict(_PAYLOAD.get("region_index", {}))
logger.info(
    "Loaded %d predictions covering %d regions from %s",
    len(_LOOKUP),
    len(_REGIONS),
    _PREDICTIONS_PATH,
)

app = Flask(__name__)


@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


def _bad_request(message: str):
    return jsonify({"error": message}), 400


def _coerce_int_field(name: str, value: Any, min_v: int, max_v: int):
    if isinstance(value, bool):
        return None, f"{name} must be an integer, not a boolean."
    if isinstance(value, float):
        if not value.is_integer():
            return None, f"{name} must be a whole number."
        value = int(value)
    if not isinstance(value, int):
        return None, f"{name} must be an integer."
    if not (min_v <= value <= max_v):
        return None, f"{name} must be an integer between {min_v} and {max_v}."
    return value, None


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "Smart Energy Consumption Predictor",
            "endpoints": ["/health", "/regions", "/predict"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": "GBTRegressionModel (precomputed lookup)",
            "features": _PAYLOAD.get("features", ["hour", "day", "month", "region"]),
            "regions": _REGIONS,
            "predictions_loaded": len(_LOOKUP),
        }
    )


@app.route("/regions", methods=["GET"])
def regions():
    return jsonify({"mapping": _REGION_INDEX})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    if not request.is_json:
        return _bad_request("Content-Type must be application/json.")

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return _bad_request("Expected JSON body with hour, day, month, and region.")

    hour = data.get("hour")
    day = data.get("day")
    month = data.get("month")
    region = data.get("region")

    missing = [
        n for n, v in (("hour", hour), ("day", day), ("month", month), ("region", region))
        if v is None
    ]
    if missing:
        return _bad_request(f"Missing required field(s): {', '.join(missing)}")

    hour, err = _coerce_int_field("hour", hour, 0, 23)
    if err:
        return _bad_request(err)
    day, err = _coerce_int_field("day", day, 1, 7)
    if err:
        return _bad_request(err)
    month, err = _coerce_int_field("month", month, 1, 12)
    if err:
        return _bad_request(err)
    if not isinstance(region, str):
        return _bad_request("region must be a string.")
    region = region.strip().upper()
    if region not in _REGION_INDEX:
        return _bad_request("region must be one of: " + ", ".join(_REGIONS))

    key = f"{hour}-{day}-{month}-{region}"
    value = _LOOKUP.get(key)
    if value is None:
        # Shouldn't happen if precompute covered the full grid, but stay graceful.
        logger.error("Lookup miss for key %s", key)
        return jsonify({"error": "No precomputed prediction for that combination."}), 500

    logger.info(
        "predict hour=%s day=%s month=%s region=%s -> %s",
        hour, day, month, region, value,
    )
    return jsonify({"prediction": float(value)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
