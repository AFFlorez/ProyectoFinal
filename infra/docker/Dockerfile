# Imagen de la app de pronóstico de dengue (API + interfaz web).
# Se construye desde la raíz del repositorio:
#   docker build -f infra/docker/Dockerfile -t dengue-app .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PUERTO=8000 \
    WORKERS=1

WORKDIR /srv
COPY requirements/app.txt requirements/app.txt
RUN pip install --root-user-action=ignore -r requirements/app.txt

RUN useradd --create-home --uid 10001 dengue
COPY --chown=dengue app/ app/
COPY --chown=dengue modelos/ modelos/
COPY --chown=dengue data/DatosProcesados/ data/DatosProcesados/
USER dengue

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PUERTO\"]}/api/salud', timeout=4)"

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PUERTO} --workers ${WORKERS} --proxy-headers"]
