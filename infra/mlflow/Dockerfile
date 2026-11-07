# Servidor de MLflow con la comparación de modelos y el modelo registrado.
# Se construye desde la raíz del repositorio:
#   docker build -f infra/mlflow/Dockerfile -t dengue-mlflow .
FROM python:3.13-slim

# MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false evita los procesos de trabajos en segundo plano
# (funciones de IA generativa que este proyecto no usa): ahorra ~1,4 GB de RAM.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

WORKDIR /srv
COPY requirements/mlflow.txt requirements/mlflow.txt
RUN pip install --root-user-action=ignore -r requirements/mlflow.txt

RUN useradd --create-home --uid 10002 mlflow && mkdir /datos && chown mlflow /datos
COPY --chown=mlflow infra/mlflow/registrar_experimentos.py infra/mlflow/iniciar.sh ./
COPY --chown=mlflow modelos/ modelos/
USER mlflow

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"

CMD ["sh", "/srv/iniciar.sh"]
