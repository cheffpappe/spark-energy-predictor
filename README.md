# Smart Energy Consumption Predictor

Flask API + static HTML frontend that predicts hourly PJM energy load from
`hour`, `day`, `month`, and `region`.

The underlying model is a PySpark GBT regressor, but because the model's
inputs are all discrete and low-cardinality (24 × 7 × 12 × 11 = 22,176
combinations), we precompute every possible prediction once and ship the
result as a JSON lookup table. The deployed API is a tiny, dependency-light
Flask app — no Spark, no JVM — so it runs comfortably on free-tier hosts
like Render.

## Project layout

```
api/app.py                 Flask API backed by models/predictions.json
api/app_spark_legacy.py    Original PySpark Flask app (kept for precompute only)
scripts/precompute.py      Regenerates models/predictions.json via Spark
models/best_pjm_model/     Trained Spark ML GBT model
models/predictions.json    Precomputed lookup (checked in, ~1 MB)
models/region_mapping.json Region -> index map used by training
web/index.html             Static frontend (vanilla JS, no build step)
render.yaml                Render Blueprint: free-tier API + static site
requirements.txt           Runtime deps: flask + gunicorn
requirements-precompute.txt Extra deps needed only to regenerate predictions
```

## Local development (API)

```bash
pip install -r requirements.txt
python api/app.py
# API now at http://127.0.0.1:5000
```

Open `web/index.html` directly in a browser — it detects localhost and
calls `http://127.0.0.1:5000/predict`.

## Regenerating the lookup table

Only needed if you retrain the model.

```bash
pip install -r requirements-precompute.txt   # installs pyspark (needs Java)
python scripts/precompute.py                  # writes models/predictions.json
```

Commit the regenerated `models/predictions.json` and push.

## Deploy on Render (free tier)

1. Push this repo to GitHub (already done).
2. Render dashboard → **New → Blueprint** → select this repo.
3. Render reads `render.yaml` and creates:
   - `spark-energy-api` — Flask API on the **Free** plan.
   - `spark-energy-web` — Static site (always free).
4. After the API is live, copy its URL and paste into `web/index.html`:
   ```html
   <meta name="api-base-url" content="https://spark-energy-api.onrender.com">
   ```
   Commit and push — the static site auto-redeploys.

**Free-tier caveat:** Render spins down idle free services, so the first
request after 15 minutes of inactivity takes ~30 seconds while the container
wakes. Subsequent requests are fast.

## API

| Method | Path       | Description                                |
| ------ | ---------- | ------------------------------------------ |
| GET    | `/health`  | Service health, supported regions.         |
| GET    | `/regions` | `region -> index` mapping used by model.   |
| POST   | `/predict` | Body: `{hour, day, month, region}` JSON.   |

```bash
curl -X POST https://spark-energy-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"hour": 14, "day": 3, "month": 7, "region": "PJME"}'
```
