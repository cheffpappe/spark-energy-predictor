# Smart Energy Consumption Predictor

Flask + PySpark GBT regression API with a static HTML frontend.
Predicts hourly PJM energy load given `hour`, `day`, `month`, and `region`.

## Project layout

```
api/app.py           Flask API wrapping a PySpark GBTRegressionModel
models/              Trained Spark ML model + region_mapping.json
web/index.html       Static frontend (vanilla JS, no build step)
Dockerfile           Java 17 + Python 3 + gunicorn for Render deploys
render.yaml          Render Blueprint: API (Docker) + static site
requirements.txt     Python deps (flask, pyspark, numpy)
```

## Local development

Requires **JDK 11+** and **Python 3.10+**.

```bash
pip install -r requirements.txt
python api/app.py
```

Then open `web/index.html` directly in a browser (it will call `http://127.0.0.1:5000/predict`).

## Deploy on Render

This repo ships with a Docker-based setup because PySpark requires a JVM
that Render's native Python runtime does not provide.

1. Push this repo to GitHub (already done if you are reading this).
2. In Render, click **New → Blueprint** and select this repo. Render will
   pick up `render.yaml` and create two services:
   - `spark-energy-api` — the Flask + Spark backend (Docker, Standard plan).
   - `spark-energy-web` — the static frontend.
3. Once the API is live, copy its URL (e.g. `https://spark-energy-api.onrender.com`)
   and paste it into `web/index.html` as:
   ```html
   <meta name="api-base-url" content="https://spark-energy-api.onrender.com">
   ```
   Commit and push — the static site will redeploy.

### Why the Standard plan

A `local[1]` SparkSession uses roughly 500 MB – 1 GB of RAM at idle, so the
free (512 MB) tier will OOM. Expect a 30–60 s cold start while the JVM and
Spark context boot.

## API

| Method | Path       | Description                            |
| ------ | ---------- | -------------------------------------- |
| GET    | `/health`  | Service health and supported regions.  |
| GET    | `/regions` | Region → index mapping used by model.  |
| POST   | `/predict` | Body: `{hour, day, month, region}` JSON. |

Example:

```bash
curl -X POST https://spark-energy-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"hour": 14, "day": 3, "month": 7, "region": "PJME"}'
```
