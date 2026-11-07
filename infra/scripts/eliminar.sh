#!/usr/bin/env bash
# Elimina la pila (instancia, grupo de seguridad e IP elástica). Uso: ./infra/scripts/eliminar.sh
set -euo pipefail
PILA="${PILA:-dengue-app}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
read -r -p "Se eliminará la pila '$PILA' en $REGION y todos sus recursos. ¿Continuar? [s/N] " respuesta
[[ "$respuesta" =~ ^[sS]$ ]] || { echo "Cancelado."; exit 0; }
aws cloudformation delete-stack --region "$REGION" --stack-name "$PILA"
aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$PILA"
echo "Pila '$PILA' eliminada."
