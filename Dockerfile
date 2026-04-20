FROM eclipse-temurin:17-jre-jammy

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        procps \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/opt/java/openjdk \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install --break-system-packages -r requirements.txt gunicorn

COPY api ./api
COPY models ./models

ENV PORT=10000
EXPOSE 10000

# Single worker: each gunicorn worker would spawn its own SparkSession and fight over JVM temp dirs.
CMD ["sh", "-c", "gunicorn --chdir api --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 180 --graceful-timeout 30 app:app"]
