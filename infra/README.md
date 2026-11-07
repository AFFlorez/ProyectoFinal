# Guía de despliegue y operación

Esta guía explica cómo se reproduce y opera el despliegue descrito en el [README principal](../README.md#solución-desplegada). Todo se crea con la plantilla [cloudformation/dengue-app.yaml](cloudformation/dengue-app.yaml).

| Carpeta | Contenido |
| --- | --- |
| `cloudformation/` | Plantilla de infraestructura como código (EC2, grupo de seguridad, IP elástica) |
| `docker/` | Imagen de la aplicación (`Dockerfile`) y `docker-compose.yml` para ejecutar todo en local |
| `mlflow/` | Imagen de MLflow y script que registra los experimentos y el modelo |
| `scripts/` | `desplegar` y `eliminar` la pila, en bash y PowerShell |

## Requisitos

- Sesión del AWS Academy Learner Lab iniciada (*Start Lab*, indicador en verde), región `us-east-1`.
- Repositorio público, o un token de solo lectura en el parámetro `TokenGitHub`.
- La rama que se despliega (`Rama`, por defecto `main`) debe contener `data/DatosProcesados/semanal_*.csv`. Sin esos archivos, la app solo ofrece el modo de datos propios.

## Desplegar

**Desde la terminal del Learner Lab** (ya tiene credenciales de AWS):

```bash
git clone https://github.com/AFFlorez/ProyectoFinal.git
cd ProyectoFinal
bash infra/scripts/desplegar.sh
```

El script valida la plantilla, crea o actualiza la pila `dengue-app` y muestra las URL. La primera vez tarda entre 10 y 15 minutos (instala Docker y construye las imágenes de TensorFlow y MLflow). Para cambiar parámetros: `bash infra/scripts/desplegar.sh Rama=main HabilitarMlflow=false`.

**Desde la consola de AWS:** CloudFormation → *Create stack* → *Upload a template file* → `infra/cloudformation/dengue-app.yaml` → nombre `dengue-app` → *Submit*. Cuando el estado sea `CREATE_COMPLETE`, las URL aparecen en la pestaña *Outputs*.

**Desde Windows:** copiar las credenciales de *AWS Details → AWS CLI → Show* en `%USERPROFILE%\.aws\credentials` y ejecutar `.\infra\scripts\desplegar.ps1`.

Salidas de la pila:

| Salida | Contenido |
| --- | --- |
| `UrlAplicacion` | Interfaz web (`http://<IP>/`) |
| `UrlDocumentacionApi` | Swagger de la API (`http://<IP>/docs`) |
| `UrlMlflow` | MLflow (`http://<IP>:5000/`) |
| `ComandoSsh` | Acceso a la instancia con `labsuser.pem` |

## Parámetros de la plantilla

| Parámetro | Por defecto | Uso |
| --- | --- | --- |
| `RepositorioUrl`, `Rama` | este repositorio, `main` | Código que se despliega |
| `TokenGitHub` | vacío | Solo si el repositorio es privado |
| `ModeloS3Uri` | vacío | `s3://bucket/prefijo/` con el modelo, si no se versiona en Git |
| `TipoInstancia` | `t3.medium` | `t3.small`, `t3.medium` o `t3.large` |
| `TamanoDiscoGb` | 30 | Disco gp3 |
| `PerfilInstancia`, `ParClaves` | `LabInstanceProfile`, `vockey` | Recursos que ya trae el Learner Lab |
| `UsarIpElastica` | `true` | URL fija entre sesiones del laboratorio |
| `PuertoPublico`, `CidrPermitido`, `CidrSsh` | 80, `0.0.0.0/0`, `0.0.0.0/0` | Red y acceso |
| `TituloApp`, `PercentilModerado`, `PercentilAlto`, `Workers`, `NivelLog` | ver plantilla | Parámetros de la aplicación |
| `HabilitarMlflow`, `PuertoMlflow` | `true`, 5000 | Servidor MLflow en la misma instancia |

## Ciclo de vida en el Learner Lab

| Situación | Qué hacer |
| --- | --- |
| Termina la sesión de 4 horas | Nada. Al volver a *Start Lab*, la instancia se enciende, Docker arranca los contenedores y la IP elástica mantiene las URL. Hay que esperar 1-2 minutos. |
| Cambios en la rama | En la instancia: `sudo dengue-desplegar` (descarga la rama, reconstruye las imágenes y reinicia los contenedores). |
| Cambiar un parámetro de la app | Volver a ejecutar `desplegar.sh Parametro=valor`, o editar `/etc/dengue-app/app.env` y ejecutar `sudo dengue-desplegar`. |
| *Reset* del laboratorio o pila borrada | `bash infra/scripts/desplegar.sh`: todo se recrea desde GitHub. |
| Eliminar todo | `bash infra/scripts/eliminar.sh` |

Antes de la evaluación conviene iniciar el laboratorio unos minutos antes y comprobar `http://<IP>/api/salud`.

## Operación y diagnóstico

| Necesidad | Cómo |
| --- | --- |
| Estado del servicio | `GET /api/salud`. Docker revisa ese endpoint cada 30 s (`sudo docker ps` muestra `healthy`). |
| Registros | `sudo docker logs -f dengue-app` (cada llamada a la API con su tiempo en ms) y `sudo docker logs dengue-mlflow` |
| Instalación | `sudo tail -100 /var/log/dengue-app-instalacion.log` |
| Acceso a la instancia | `ssh -i labsuser.pem ec2-user@<IP>` o *EC2 → Connect → EC2 Instance Connect* |

Si la pila falla, CloudFormation la revierte. Las causas más comunes son un repositorio privado sin token o una rama que no existe. Si MLflow falla, la aplicación sigue en servicio y el aviso queda en el registro de instalación.

## Ejecución local

```bash
cp .env.example .env    # opcional
docker compose -f infra/docker/docker-compose.yml up -d --build
# Aplicación: http://localhost:8000 · API: http://localhost:8000/docs · MLflow: http://localhost:5000
```

Pruebas automáticas:

```bash
docker build -f infra/docker/Dockerfile -t dengue-app .
docker run --rm -u 0 -v "$PWD:/srv" -w /srv --entrypoint sh dengue-app -c "pip install -r requirements/dev.txt && python -m pytest -q tests"
```

## Gestión del modelo

El notebook (punto 20.4) exporta el modelo elegido a `modelos/`, y `python notebooks/exportar_experimentos.py` actualiza `modelos/experimentos/`, que es lo que se carga en MLflow. Para publicar un modelo nuevo: reemplazar esas carpetas, ejecutar las pruebas, hacer commit y ejecutar `sudo dengue-desplegar` en la instancia.

Si el modelo no se versiona en Git: `aws s3 cp --recursive modelos/ s3://<bucket>/modelos/` y desplegar con `ModeloS3Uri=s3://<bucket>/modelos/`. También se puede indicar `MODELO_URL` con un `.zip` accesible por https.

## Costo aproximado

Con precios de lista en us-east-1 (pueden variar): EC2 t3.medium ≈ USD 0,04 por hora encendida, disco de 30 GB ≈ USD 2,4 al mes e IP elástica ≈ USD 3,6 al mes. Con el laboratorio encendido unas 4 horas al día, el total es de unos USD 10-11 al mes, dentro del crédito de USD 50 del Learner Lab.
