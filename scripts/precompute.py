"""
One-time precompute: materialize predictions for every possible
(hour, day, month, region) input and save them as a lookup JSON.

Produces models/predictions.json (flat dict keyed "hour-day-month-region").
After running this once locally, the Flask API no longer needs PySpark.

Run from the project root:
    python scripts/precompute.py

Requires the original PySpark app to still be present at api/app_spark_legacy.py
(it boots the SparkSession and loads the GBT model).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_legacy_app_path = os.path.join(ROOT, "api", "app_spark_legacy.py")
if not os.path.isfile(_legacy_app_path):
    raise SystemExit(
        "Expected api/app_spark_legacy.py (the original PySpark app.py renamed). "
        "If you already removed it, restore it before running this script."
    )

import importlib.util

spec = importlib.util.spec_from_file_location("app_spark_legacy", _legacy_app_path)
legacy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(legacy)  # boots SparkSession + loads the GBT model

from pyspark.sql.types import IntegerType, StringType, StructField, StructType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("precompute")

HOURS = list(range(24))
DAYS = list(range(1, 8))
MONTHS = list(range(1, 13))
REGIONS = sorted(legacy.REGION_TO_INDEX.keys())

OUT_PATH = os.path.join(ROOT, "models", "predictions.json")


def main() -> None:
    start = time.time()
    combos = [
        (h, d, m, legacy.REGION_TO_INDEX[r], r)
        for r in REGIONS
        for m in MONTHS
        for d in DAYS
        for h in HOURS
    ]
    total = len(combos)
    logger.info("Scoring %d combos via CSV + JVM read...", total)

    spark = legacy.spark

    # Write combos to a temp CSV, then read back through the JVM — this avoids the
    # Python worker path that breaks with Python 3.13 + PySpark on Windows.
    tmp_csv = None
    try:
        fd, tmp_csv = tempfile.mkstemp(suffix=".csv", prefix="precompute_", dir=legacy._SPARK_DIR)
        os.close(fd)
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hour", "day", "month", "region_index", "region"])
            for h, d, m, ri, r in combos:
                w.writerow([h, d, m, ri, r])

        schema = StructType([
            StructField("hour", IntegerType(), False),
            StructField("day", IntegerType(), False),
            StructField("month", IntegerType(), False),
            StructField("region_index", IntegerType(), False),
            StructField("region", StringType(), False),
        ])

        # Spark on Windows wants file:/// URIs with forward slashes.
        uri = "file:///" + tmp_csv.replace("\\", "/")
        df = spark.read.option("header", "true").schema(schema).csv(uri).coalesce(1)
        df = legacy._assembler.transform(df)
        scored = legacy.model.transform(df).select("hour", "day", "month", "region", "prediction")

        # Collect via JVM-side iterator to avoid Python worker serialization.
        jrows = scored._jdf.collectAsList()
        lookup: dict[str, float] = {}
        for jrow in jrows:
            h = jrow.getInt(0)
            d = jrow.getInt(1)
            m = jrow.getInt(2)
            r = jrow.getString(3)
            p = float(jrow.getDouble(4))
            lookup[f"{h}-{d}-{m}-{r}"] = round(p, 4)
    finally:
        if tmp_csv and os.path.isfile(tmp_csv):
            try:
                os.remove(tmp_csv)
            except OSError:
                pass

    if len(lookup) != total:
        raise RuntimeError(
            f"Expected {total} predictions but got {len(lookup)}. "
            "Check for duplicate keys or missing regions."
        )

    payload = {
        "schema_version": 1,
        "features": ["hour", "day", "month", "region"],
        "hour_range": [0, 23],
        "day_range": [1, 7],
        "month_range": [1, 12],
        "regions": REGIONS,
        "region_index": {r: legacy.REGION_TO_INDEX[r] for r in REGIONS},
        "count": len(lookup),
        "predictions": lookup,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = os.path.getsize(OUT_PATH) / 1024
    elapsed = time.time() - start
    logger.info(
        "Wrote %d predictions to %s (%.1f KB) in %.1fs.",
        len(lookup),
        OUT_PATH,
        size_kb,
        elapsed,
    )


if __name__ == "__main__":
    main()
