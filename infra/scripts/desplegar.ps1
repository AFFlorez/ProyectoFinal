<#
Crea o actualiza la pila de CloudFormation desde Windows (requiere AWS CLI v2).

Uso:
  .\infra\scripts\desplegar.ps1
  .\infra\scripts\desplegar.ps1 -Parametros "Rama=main","TipoInstancia=t3.large"
  .\infra\scripts\desplegar.ps1 -Pila dengue-pruebas
Antes: Learner Lab > Start Lab > AWS Details > AWS CLI > Show, y pegar el bloque en
%USERPROFILE%\.aws\credentials (las credenciales cambian en cada sesión).
#>
param(
    [string]$Pila = "dengue-app",
    [string]$Region = "us-east-1",
    [string[]]$Parametros = @()
)
$ErrorActionPreference = "Stop"
$plantilla = Join-Path $PSScriptRoot "..\cloudformation\dengue-app.yaml" | Resolve-Path

aws sts get-caller-identity --region $Region | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No hay credenciales de AWS válidas (AWS Details > AWS CLI en Learner Lab)." }

Write-Host "Desplegando la pila '$Pila' en $Region. La primera vez tarda entre 8 y 15 minutos..."
$argumentos = @("cloudformation", "deploy", "--region", $Region, "--stack-name", $Pila,
    "--template-file", "$plantilla", "--no-fail-on-empty-changeset")
if ($Parametros.Count -gt 0) { $argumentos += @("--parameter-overrides") + $Parametros }
aws @argumentos
if ($LASTEXITCODE -ne 0) {
    aws cloudformation describe-stack-events --region $Region --stack-name $Pila --max-items 15 `
        --query "StackEvents[?contains(ResourceStatus, 'FAILED')].[LogicalResourceId,ResourceStatusReason]" --output table
    throw "Falló el despliegue."
}
aws cloudformation describe-stacks --region $Region --stack-name $Pila `
    --query "Stacks[0].Outputs[].[OutputKey,OutputValue]" --output table
$url = aws cloudformation describe-stacks --region $Region --stack-name $Pila `
    --query "Stacks[0].Outputs[?OutputKey=='UrlAplicacion'].OutputValue" --output text
Write-Host "Listo: $url"
Start-Process $url
