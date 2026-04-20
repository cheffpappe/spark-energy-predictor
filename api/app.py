"""
Flask API for PySpark GBT energy consumption prediction.
Compatible with frontend POST to /predict with JSON { hour, day, month, region }.
"""

import json
import logging
import os
import sys
import traceback

# --- Windows: JVM gateway can fail if TEMP / project paths contain spaces ---
def _spark_work_dir() -> str:
    if sys.platform != "win32":
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spark_work")
    # Prefer a path without spaces (avoids broken command lines for the Java gateway)
    candidates = [
        r"C:\SparkEnergyTmp",
        os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "Temp", "SparkEnergy"),
    ]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except OSError:
            continue
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spark_work")


_SPARK_DIR = _spark_work_dir()
# Windows: fixed Hadoop shim path (must match where winutils.exe is installed; independent of _SPARK_DIR fallback)
_WIN_HADOOP_HOME = r"C:\SparkEnergyTmp\hadoop"
_WINUTILS_EXE = os.path.join(_WIN_HADOOP_HOME, "bin", "winutils.exe")

if sys.platform == "win32":
    # Force JVM / Spark temp away from %USERPROFILE%\AppData\... (spaces in path break the gateway)
    os.environ["TEMP"] = _SPARK_DIR
    os.environ["TMP"] = _SPARK_DIR
    os.environ["TMPDIR"] = _SPARK_DIR
    if os.path.isfile(_WINUTILS_EXE):
        os.environ["HADOOP_HOME"] = _WIN_HADOOP_HOME
        _bin = os.path.join(_WIN_HADOOP_HOME, "bin")
        os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")

from flask import Flask, jsonify, request
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressionModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Region string → numeric index (must match training pipeline) ---
# Canonical fallback mapping (used when models/region_mapping.json is missing).
_FALLBACK_REGION_TO_INDEX = {
    "PJME": 0,
    "PJMW": 1,
    "DAYTON": 2,
    "AEP": 3,
    "DUQ": 4,
    "DOM": 5,
    "COMED": 6,
    "FE": 7,
    "NI": 8,
    "DEOK": 9,
    "EKPC": 10,
}

_REGION_MAPPING_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "region_mapping.json")
)


def _load_region_mapping():
    """Load models/region_mapping.json if it exists; otherwise use the fallback."""
    if os.path.isfile(_REGION_MAPPING_PATH):
        try:
            with open(_REGION_MAPPING_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict) or not raw:
                raise ValueError("region_mapping.json must be a non-empty object.")
            mapping = {}
            for k, v in raw.items():
                if not isinstance(k, str):
                    raise ValueError("region_mapping.json keys must be strings.")
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError("region_mapping.json values must be integers.")
                mapping[k.strip().upper()] = int(v)
            logging.getLogger(__name__).info(
                "Loaded region mapping from %s (%d regions).",
                _REGION_MAPPING_PATH,
                len(mapping),
            )
            return mapping
        except Exception:
            logging.getLogger(__name__).exception(
                "Failed to parse %s, falling back to hardcoded mapping.",
                _REGION_MAPPING_PATH,
            )
    return dict(_FALLBACK_REGION_TO_INDEX)


REGION_TO_INDEX = _load_region_mapping()

# --- Flask ---
app = Flask(__name__)


def _add_cors_headers(response):
    """Allow browser frontend to call this API from another origin/port."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.after_request
def after_request(response):
    return _add_cors_headers(response)


# --- Spark: single session for the whole process (not per request) ---
def _create_spark():
    tmp_opt = "-Djava.io.tmpdir=" + _SPARK_DIR.replace("\\", "/")
    if sys.platform == "win32" and os.path.isfile(_WINUTILS_EXE):
        _lib = os.path.join(_WIN_HADOOP_HOME, "bin").replace("\\", "/")
        tmp_opt += " -Djava.library.path=" + _lib
    # local[1] + low parallelism: stable single-row inference on Windows / Python 3.13
    b = (
        SparkSession.builder.appName("EnergyPredictAPI")
        .master("local[1]")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.sql.codegen.wholeStage", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.local.dir", _SPARK_DIR)
        .config("spark.driver.extraJavaOptions", tmp_opt)
    )
    if sys.platform == "win32" and os.path.isfile(_WINUTILS_EXE):
        b = b.config("spark.hadoop.hadoop.home.dir", _WIN_HADOOP_HOME)
    sess = b.getOrCreate()
    sess.sparkContext.setLogLevel("WARN")
    return sess


try:
    spark = _create_spark()
except Exception:
    logger.exception("Failed to start SparkSession")
    print(
        "\n[Spark] Could not start Spark. Install JDK 8+ (11+ recommended), set JAVA_HOME, "
        "and ensure temp dir is writable: %s\n" % _SPARK_DIR
    )
    raise

# Model path: ../models/best_pjm_model relative to this file (api/app.py)
_MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "best_pjm_model")
)
if not os.path.isdir(_MODEL_DIR):
    raise FileNotFoundError(
        f"Model directory not found at {_MODEL_DIR}. "
        "Expected Spark ML GBTRegressionModel folder."
    )
model = GBTRegressionModel.load(_MODEL_DIR)
logger.info("Loaded GBT model from %s", _MODEL_DIR)

# Reusable assembler: EXACT feature order as training
_assembler = VectorAssembler(
    inputCols=["hour", "day", "month", "region_index"],
    outputCol="features",
)


def _bad_request(message):
    return jsonify({"error": message}), 400


def _server_error(message):
    return jsonify({"error": message}), 500


def _coerce_int_field(name, value, min_v, max_v):
    """Parse JSON number as int; reject bool and non-whole floats."""
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


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model": "GBTRegressionModel",
            "features": ["hour", "day", "month", "region_index"],
            "regions": sorted(REGION_TO_INDEX.keys()),
        }
    )


@app.route("/regions", methods=["GET"])
def regions():
    return jsonify({"mapping": REGION_TO_INDEX})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # CORS preflight
    if request.method == "OPTIONS":
        return "", 204

    try:
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
            name
            for name, val in (("hour", hour), ("day", day), ("month", month), ("region", region))
            if val is None
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
        if region not in REGION_TO_INDEX:
            return _bad_request(
                "region must be one of: "
                + ", ".join(sorted(REGION_TO_INDEX.keys()))
            )

        region_index = REGION_TO_INDEX[region]

        logger.info(
            "Incoming /predict: hour=%s day=%s month=%s region=%s region_index=%s",
            hour,
            day,
            month,
            region,
            region_index,
        )
        print(
            f"[predict] hour={hour} day={day} month={month} region={region} region_index={region_index}"
        )

        # Single-row DataFrame built from JVM literals (avoids Python worker path used by
        # createDataFrame from Python rows — needed for Python 3.13 + PySpark on Windows).
        df = (
            spark.range(1)
            .select(
                F.lit(hour).cast("int").alias("hour"),
                F.lit(day).cast("int").alias("day"),
                F.lit(month).cast("int").alias("month"),
                F.lit(region_index).cast("int").alias("region_index"),
            )
        )
        df = _assembler.transform(df).coalesce(1)
        scored = model.transform(df).select("prediction")

        jrow = scored._jdf.first()
        out = float(jrow.getDouble(0))
        logger.info("Prediction value: %s", out)
        print(f"[predict] prediction={out}")

        return jsonify({"prediction": out})

    except ValueError as e:
        return _bad_request(str(e))
    except Exception:
        traceback.print_exc()
        return _server_error("Prediction failed. Check server logs for details.")


if __name__ == "__main__":
    # use_reloader=False: Flask debug reloader spawns a second process and a second SparkSession (breaks PySpark).
    app.run(debug=True, use_reloader=False)
