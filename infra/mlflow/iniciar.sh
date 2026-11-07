#!/bin/sh
# Inicia el servidor de MLflow y, la primera vez, carga los experimentos del proyecto.
# El registro vive dentro del contenedor: al reconstruirlo (dengue-desplegar) se vuelve a cargar.
set -e
mkdir -p /datos/artefactos
mlflow server --host 0.0.0.0 --port 5000 --workers "${MLFLOW_WORKERS:-1}" \
  --backend-store-uri sqlite:////datos/mlflow.db \
  --artifacts-destination /datos/artefactos --serve-artifacts \
  --allowed-hosts "${MLFLOW_HOSTS_PERMITIDOS:-*}" &
SERVIDOR=$!

until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2)" 2>/dev/null; do
  kill -0 "$SERVIDOR" 2>/dev/null || exit 1
  sleep 1
done
if [ ! -f /datos/.registrado ]; then
  python /srv/registrar_experimentos.py --tracking http://127.0.0.1:5000 --modelos /srv/modelos
  touch /datos/.registrado
fi
wait "$SERVIDOR"
