#!/usr/bin/env bash
# Crea o actualiza la pila de CloudFormation y muestra la URL de la aplicación.
#
# Uso (terminal del AWS Academy Learner Lab o cualquier equipo con AWS CLI configurado):
#   ./infra/scripts/desplegar.sh                          # valores por defecto
#   ./infra/scripts/desplegar.sh Rama=main TipoInstancia=t3.large
#   PILA=dengue-pruebas ./infra/scripts/desplegar.sh
set -euo pipefail

PILA="${PILA:-dengue-app}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
PLANTILLA="$RAIZ/infra/cloudformation/dengue-app.yaml"

if ! aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  echo "No hay credenciales de AWS válidas. En Learner Lab: Start Lab > AWS Details > AWS CLI > Show," >&2
  echo "y copia el contenido en ~/.aws/credentials (las credenciales cambian en cada sesión)." >&2
  exit 1
fi

aws cloudformation validate-template --region "$REGION" --template-body "file://$PLANTILLA" >/dev/null
echo "Desplegando la pila '$PILA' en $REGION. La primera vez tarda entre 8 y 15 minutos (instala Docker y TensorFlow)..."

argumentos=(cloudformation deploy --region "$REGION" --stack-name "$PILA" --template-file "$PLANTILLA" --no-fail-on-empty-changeset)
if [ "$#" -gt 0 ]; then argumentos+=(--parameter-overrides "$@"); fi
if ! aws "${argumentos[@]}"; then
  echo "Falló el despliegue. Eventos recientes:" >&2
  aws cloudformation describe-stack-events --region "$REGION" --stack-name "$PILA" --max-items 15 \
    --query "StackEvents[?contains(ResourceStatus, 'FAILED')].[LogicalResourceId,ResourceStatusReason]" --output table >&2 || true
  exit 1
fi

aws cloudformation describe-stacks --region "$REGION" --stack-name "$PILA" \
  --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output table
URL="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$PILA" \
  --query "Stacks[0].Outputs[?OutputKey=='UrlAplicacion'].OutputValue" --output text)"
echo "Comprobando ${URL}api/salud ..."
curl -sf --max-time 10 "${URL}api/salud" && echo && echo "Listo: $URL"
