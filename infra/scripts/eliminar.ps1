# Elimina la pila (instancia, grupo de seguridad e IP elástica). Uso: .\infra\scripts\eliminar.ps1
param([string]$Pila = "dengue-app", [string]$Region = "us-east-1")
$ErrorActionPreference = "Stop"
$respuesta = Read-Host "Se eliminará la pila '$Pila' en $Region y todos sus recursos. ¿Continuar? [s/N]"
if ($respuesta -notmatch '^[sS]$') { Write-Host "Cancelado."; exit 0 }
aws cloudformation delete-stack --region $Region --stack-name $Pila
aws cloudformation wait stack-delete-complete --region $Region --stack-name $Pila
Write-Host "Pila '$Pila' eliminada."
