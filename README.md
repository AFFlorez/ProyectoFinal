# Pronóstico semanal de dengue en el Valle del Cauca

Aplicación web y API que estiman los **casos de dengue de la semana t+4** (cuatro semanas adelante) en los cinco municipios del Valle del Cauca con más casos: **Cali, Guadalajara de Buga, Palmira, Tuluá y Yumbo**. El modelo usa las **últimas 20 semanas de casos notificados y clima** (temperatura media, máxima y mínima, y precipitación) para dar a los equipos de vigilancia en salud pública un margen de un mes para planear las acciones de prevención y control.

- **Problema:** el dengue es endémico en el Valle del Cauca y presenta brotes estacionales. Los casos notificados en SIVIGILA muestran lo que ya ocurrió; un pronóstico con semanas de anticipación ayuda a preparar campañas, insumos y atención.
- **Solución:** una red **LSTM global** (64 unidades + Dense 16, 19.569 parámetros), elegida entre **160 modelos** (regresión lineal, Random Forest, XGBoost y redes LSTM globales y locales con historias de 8 a 24 semanas) por tener el menor MAE de validación. Se despliega en AWS como una aplicación web con interfaz para el usuario final y una API documentada.

| | Enlace |
| --- | --- |
| Aplicación desplegada | `http://<IP-ELÁSTICA>/` (salida `UrlAplicacion` de la pila de CloudFormation) |
| Documentación de la API (Swagger) | `http://<IP-ELÁSTICA>/docs` |
| Modelado y experimentos | [notebooks/README.md](notebooks/README.md) · [notebooks/ProyectoFinal.ipynb](notebooks/ProyectoFinal.ipynb) |
| Paquete del modelo | [modelos/](modelos/) |
| Plantilla de infraestructura | [infra/cloudformation/dengue-app.yaml](infra/cloudformation/dengue-app.yaml) |
| Datos de casos | [Portal SIVIGILA – Instituto Nacional de Salud](https://portalsivigila.ins.gov.co/) (evento 210, dengue) |

## Arquitectura

![Arquitectura de despliegue](docs/arquitectura.svg)

Una sola plantilla de **CloudFormation** crea un grupo de seguridad, una instancia **EC2** (Amazon Linux 2023) y una **IP elástica**. Al arrancar, la instancia instala Docker, clona este repositorio, construye la imagen y levanta el contenedor. CloudFormation espera la señal `cfn-signal` de la instancia, que solo se envía cuando `/api/salud` responde. Así, cuando la pila termina en `CREATE_COMPLETE`, la aplicación ya está funcionando.

Se eligió EC2 + Docker porque funciona dentro de las restricciones del **AWS Academy Learner Lab** (no permite crear roles IAM; se usa `LabInstanceProfile`). El modelo es pequeño (271 KB), así que no hace falta GPU, SageMaker ni un servicio externo. La IP elástica conserva la misma URL cuando el laboratorio se apaga y se vuelve a encender, y el contenedor tiene `restart: unless-stopped`, así que la aplicación vuelve sola.

## Estructura del repositorio

```text
ProyectoFinal/
├── README.md                    Este documento
├── .env.example                 Variables de configuración de la app (copiar a .env)
│
├── notebooks/                   Exploración, entrenamiento y experimentación
│   ├── ProyectoFinal.ipynb      Notebook principal (puntos 1-20)
│   ├── *.py                     Módulos que usa el notebook y análisis complementarios
│   ├── Iniciar-Notebook.cmd     Abre JupyterLab con el entorno .venv (Windows)
│   └── README.md                Detalle del modelado y resultados
│
├── data/                        Datos (ver data/README.md)
│   ├── DataRaw/                 Casos SIVIGILA reducidos al Valle del Cauca, 2007-2025
│   ├── DatosProcesados/         Serie semanal de los 5 municipios: casos + clima
│   └── MunicipiosValledelCauca.csv
│
├── modelos/                     Modelo final exportado por el notebook (punto 20.4)
│   ├── modelo.keras             Red LSTM entrenada (271 KB)
│   ├── preprocesamiento.joblib  Escaladores y codificador ajustados en entrenamiento
│   ├── configuracion.json       Arquitectura, variables, métricas y versiones
│   └── sha256.json              Huellas para verificar la integridad
│
├── app/                         Aplicación desplegada (FastAPI)
│   ├── main.py                  API REST e interfaz web
│   ├── pronosticador.py         Carga del modelo, transformaciones e inferencia
│   ├── historico.py             Serie 2007-2025 y pronósticos precalculados
│   ├── calendario.py            Calendario epidemiológico (semanas de domingo a sábado)
│   ├── config.py                Parámetros leídos de variables de entorno
│   └── static/                  Interfaz web (HTML, CSS, JavaScript + Chart.js)
│
├── infra/                       Despliegue
│   ├── cloudformation/          Plantilla de infraestructura como código (AWS)
│   ├── docker/                  Dockerfile y docker-compose.yml
│   └── scripts/                 desplegar / eliminar (bash y PowerShell)
│
├── requirements/                Dependencias de Python, separadas por uso
│   ├── app.txt                  Aplicación desplegada (Linux, Python 3.13)
│   ├── dev.txt                  app.txt + herramientas de pruebas
│   ├── notebook.txt             Librerías del notebook
│   └── notebook-lock.txt        Versiones exactas del entorno del notebook
│
├── tests/                       Pruebas automáticas (reproducción del notebook y API)
└── docs/                        Diagrama de arquitectura, artículos y entregables
```

Los resultados de entrenamiento (`ResultadosModelos/`), el clima original (`data/ShapeMunicipios/`) y la copia completa de SIVIGILA (`data/DataRawOriginal/`) se guardan solo en local y Git los ignora.

## Uso de la aplicación

La interfaz tiene cuatro pestañas:

1. **Pronóstico.** Se elige el municipio y la semana de emisión, es decir, la última semana epidemiológica con datos observados. La app muestra los casos estimados para cuatro semanas después, el **nivel de alerta** (bajo, moderado o alto), los casos reales si esa semana ya ocurrió y dos gráficas: los casos de entrada con el pronóstico, y el clima de esas semanas. También muestra una tabla con los cinco municipios para la misma semana.
2. **Desempeño histórico.** Compara, semana a semana, el pronóstico con los casos reales en un rango de años, con MAE, RMSE y sesgo separados por entrenamiento, validación y prueba.
3. **Usar mis datos.** Se descarga la plantilla CSV, se completan al menos 20 semanas consecutivas por municipio y se sube el archivo. Acepta CSV con coma o con punto y coma y coma decimal (formato de Excel en español).
4. **Acerca del modelo.** Explica qué hace el modelo, cómo interpretar la alerta, sus métricas y sus limitaciones.

**Nivel de alerta.** Compara el pronóstico con la distribución de los casos semanales del municipio en 2007-2021 (los datos de entrenamiento). Se marca moderado desde el percentil 75 y alto desde el percentil 90. Ambos percentiles se configuran con variables de entorno.

### API

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/api/salud` | Estado del servicio y tiempos de carga |
| GET | `/api/modelo` | Arquitectura, variables y métricas del modelo |
| GET | `/api/configuracion` | Parámetros activos |
| GET | `/api/municipios` | Municipios, umbrales de alerta y semanas disponibles |
| GET | `/api/pronostico?codigo=76001&ano=2024&semana=10` | Pronóstico t+4 emitido en una semana del histórico (sin `ano`/`semana`: la última) |
| GET | `/api/resumen?ano=2024&semana=10` | Los cinco municipios para una semana de emisión |
| GET | `/api/desempeno?codigo=76001&desde=2022&hasta=2025` | Pronósticos frente a casos reales y métricas |
| GET | `/api/plantilla.csv` | Plantilla de entrada con 20 semanas de ejemplo |
| POST | `/api/pronostico/archivo` | Pronóstico a partir de un CSV (multipart, campo `archivo`) |
| POST | `/api/pronostico/semanas` | Pronóstico a partir de semanas en JSON |

**Ejemplo con CSV.** La plantilla trae una fila por municipio y semana:

```csv
COD_DANE,MUNICIPIO,FECHA_INICIO,CASOS,TEMP_MEDIA_C,TEMP_MAX_C,TEMP_MIN_C,PRECIPITACION_MM
76001,Cali,2021-08-15,112,17.76,22.32,14.71,36.9
...
76001,Cali,2021-12-26,78,17.93,21.71,15.04,59.63
```

```bash
curl -F archivo=@plantilla_pronostico_dengue.csv http://<IP-ELÁSTICA>/api/pronostico/archivo
```

Respuesta (resumida). Con las 20 semanas que terminan el 1 de enero de 2022, el modelo estima unos 74 casos en Cali para la semana del 23 al 29 de enero de 2022:

```json
{"resultados": [{
  "municipio": {"codigo": "76001", "nombre": "Cali"},
  "emision":  {"ano": 2021, "semana": 52, "fecha_inicio": "2021-12-26", "fecha_fin": "2022-01-01"},
  "objetivo": {"ano": 2022, "semana": 4,  "fecha_inicio": "2022-01-23", "fecha_fin": "2022-01-29"},
  "horizonte_semanas": 4,
  "casos_predichos": 74.24,
  "casos_predichos_redondeados": 74,
  "nivel_alerta": "bajo",
  "umbrales_alerta": {"moderado": 172.8, "alto": 391.1},
  "casos_reales": null,
  "historia": ["...20 semanas de entrada..."]
}]}
```

Los errores de entrada devuelven `422` con un mensaje en español, por ejemplo: `"76001 (Cali): se requieren 20 semanas; hay 1."`

## Despliegue en AWS (Learner Lab)

### Requisitos

- Sesión del **AWS Academy Learner Lab** iniciada (`Start Lab`, indicador en verde). Región `us-east-1`.
- El repositorio debe ser **público**. Si es privado, hay que pasar un token de solo lectura en el parámetro `TokenGitHub`.
- La rama que se despliega (`Rama`, por defecto `main`) debe contener los archivos `data/DatosProcesados/semanal_*.csv`. Sin ella, la app funciona solo en el modo "Usar mis datos".

### Opción A: terminal del Learner Lab (recomendada)

La terminal que aparece en la página del laboratorio ya tiene credenciales de AWS configuradas:

```bash
git clone https://github.com/AFFlorez/ProyectoFinal.git
cd ProyectoFinal
bash infra/scripts/desplegar.sh
```

El script valida la plantilla, crea o actualiza la pila `dengue-app`, espera a que termine (de 8 a 15 minutos la primera vez) y muestra la URL. Se pueden cambiar parámetros, por ejemplo: `bash infra/scripts/desplegar.sh Rama=despliegue-aws TipoInstancia=t3.large`.

### Opción B: consola de CloudFormation

1. AWS Console → CloudFormation → *Create stack* → *With new resources*.
2. *Upload a template file* → `infra/cloudformation/dengue-app.yaml`.
3. Nombre de la pila: `dengue-app`. Revisar los parámetros (los valores por defecto sirven) → *Submit*.
4. Cuando el estado sea `CREATE_COMPLETE`, abrir la URL de la pestaña *Outputs* → `UrlAplicacion`.

### Opción C: desde Windows (PowerShell + AWS CLI)

Copiar las credenciales de *AWS Details → AWS CLI → Show* en `%USERPROFILE%\.aws\credentials` y ejecutar:

```powershell
.\infra\scripts\desplegar.ps1
```

### Qué pasa cuando el laboratorio se apaga

| Situación | Qué hacer |
| --- | --- |
| Termina la sesión de 4 horas (*End Lab* o tiempo agotado) | Nada. La instancia se detiene; al volver a hacer *Start Lab* se enciende, Docker arranca el contenedor y la URL (IP elástica) es la misma. Esperar 1-2 minutos. |
| Se publicaron cambios en la rama | `ssh -i labsuser.pem ec2-user@<IP>` y luego `sudo dengue-desplegar`. Otra opción es *EC2 → Connect → EC2 Instance Connect*. |
| Se cambió un parámetro de la app (percentiles, título, workers) | Volver a ejecutar `desplegar.sh Parametro=valor`, o editar `/etc/dengue-app/app.env` y ejecutar `sudo dengue-desplegar`. |
| *Reset* del laboratorio o se borró la pila | Volver a ejecutar `bash infra/scripts/desplegar.sh`. Todo se recrea desde GitHub. |
| Eliminar todo | `bash infra/scripts/eliminar.sh` (o borrar la pila desde la consola). |

**Antes de la evaluación:** iniciar el laboratorio unos minutos antes de la hora asignada y comprobar `http://<IP>/api/salud`.

**Diagnóstico:** en la instancia, `sudo tail -100 /var/log/dengue-app-instalacion.log` (instalación) y `sudo docker logs dengue-app` (aplicación). Si la pila falla, CloudFormation la revierte. La causa más común es que el repositorio sea privado o que la rama no exista.

## Ejecución local

Con Docker, desde la raíz del repositorio:

```bash
cp .env.example .env        # opcional: ajustar parámetros
docker compose -f infra/docker/docker-compose.yml up -d --build
# Interfaz: http://localhost:8000   API: http://localhost:8000/docs
```

Sin Docker (Python 3.13, Linux o macOS; en Windows conviene usar Docker o WSL):

```bash
python -m venv .venv-app && source .venv-app/bin/activate
pip install -r requirements/app.txt
uvicorn app.main:app --port 8000
```

El entorno del notebook es distinto; su instalación está en [notebooks/README.md](notebooks/README.md#entorno).

## Parametrización

Todos los parámetros se leen de variables de entorno. Están documentados en [.env.example](.env.example). En AWS se fijan con los parámetros de la plantilla y quedan en `/etc/dengue-app/app.env`.

| Variable | Por defecto | Uso |
| --- | --- | --- |
| `APP_TITULO` | Pronóstico semanal de dengue · Valle del Cauca | Título de la interfaz |
| `MODELO_DIR` | `modelos` | Carpeta del paquete del modelo |
| `MODELO_URL` | vacío | URL https de un `.zip` con el modelo, si no está en el repositorio |
| `VERIFICAR_SHA256` | `true` | Rechaza un modelo o preprocesamiento alterado |
| `DATOS_DIR` | `data/DatosProcesados` | Carpeta de `semanal_*.csv` |
| `ALERTA_PERCENTIL_MODERADO` / `ALERTA_PERCENTIL_ALTO` | `75` / `90` | Umbrales del nivel de alerta |
| `MAX_SEMANAS_ENTRADA` | `2000` | Límite de filas por solicitud con datos propios |
| `WORKERS` | `1` (`2` en AWS) | Procesos de uvicorn |
| `LOG_LEVEL` | `INFO` | Nivel de registro |
| `CORS_ORIGENES` | `*` | Dominios que pueden consumir la API |

Parámetros de la plantilla de CloudFormation: `RepositorioUrl`, `Rama`, `TokenGitHub`, `ModeloS3Uri`, `TipoInstancia` (t3.small, t3.medium o t3.large), `TamanoDiscoGb`, `PerfilInstancia`, `ParClaves`, `UsarIpElastica`, `PuertoPublico`, `CidrPermitido`, `CidrSsh`, `TituloApp`, `PercentilModerado`, `PercentilAlto`, `Workers` y `NivelLog`.

## Consideraciones del modelo

| Aspecto | Valor |
| --- | --- |
| Arquitectura | LSTM 64 → Dense 16 → salida lineal, con one-hot del municipio |
| Parámetros / tamaño | 19.569 / 271 KB (`modelo.keras`) |
| Entrada | 20 semanas × 7 variables: casos, 3 temperaturas, precipitación, seno y coseno de la semana |
| Salida | `log1p(casos)` estandarizado por municipio, convertido a casos sin redondear |
| Hardware | Solo CPU; ≈ 290 MB de RAM por proceso; imagen Docker ≈ 2,5 GB (TensorFlow CPU) |
| Carga | ≈ 5 s (importar TensorFlow y cargar el modelo) |
| Histórico | ≈ 4.900 pronósticos precalculados en un solo lote en ≈ 4 s al iniciar; cada consulta tarda 20-40 ms |
| Datos propios | ≈ 0,2 s por archivo de cinco municipios |
| Integridad | Al iniciar se verifica el SHA-256 de `modelo.keras` y `preprocesamiento.joblib` contra `sha256.json` |

**Estrategia de despliegue.** El modelo cabe en el repositorio, así que se versiona con el código. Las inferencias del histórico se calculan una sola vez al iniciar el servicio (en un lote vectorizado). Por eso las consultas no ejecutan la red y responden en milisegundos, incluso con varios usuarios. TensorFlow se importa de forma diferida dentro del pronosticador. Si el modelo creciera, se puede sacar del repositorio y entregarlo desde S3 (`ModeloS3Uri`) o desde una URL (`MODELO_URL`) sin cambiar el código.

**Gestión del modelo.** El notebook (punto 20.4) exporta el modelo elegido a `modelos/`. Para publicar un modelo nuevo se reemplaza esa carpeta, se ejecutan las pruebas, se hace commit y se ejecuta `sudo dengue-desplegar` en la instancia. Si el paquete se guarda en S3: `aws s3 cp --recursive modelos/ s3://<bucket>/modelos/` y desplegar con `ModeloS3Uri=s3://<bucket>/modelos/`.

**Desempeño y limitaciones.** El MAE es de 11,4 casos en validación (2022-2023) y de 38,1 en prueba (2024-2025). En 2024-2025 hubo un brote mayor que todos los de entrenamiento, y el modelo tiende a subestimar los picos (sesgo medio negativo). No imputa datos faltantes y solo cubre los cinco municipios entrenados. Es una herramienta de apoyo, no reemplaza la vigilancia epidemiológica. El análisis completo está en [notebooks/README.md](notebooks/README.md).

## Datos

El detalle está en [data/README.md](data/README.md). En resumen:

- `data/DataRaw/`: casos individuales de SIVIGILA (evento 210) reducidos al Valle del Cauca y a las cinco columnas que se usan.
- `data/DatosProcesados/semanal_*.csv`: serie semanal por municipio con casos y clima, generada en el punto 11 del notebook. Es lo único que la aplicación necesita del histórico (≈ 5.000 filas, menos de 1 MB).
- Los archivos de clima originales (`data/ShapeMunicipios/CLIMA_DENGUE_VALLE/`) y los datos completos se conservan fuera de Git por su tamaño. Para regenerar la serie semanal hay que ubicarlos en esa carpeta y ejecutar el notebook hasta el punto 11.

## Pruebas

```bash
docker build -f infra/docker/Dockerfile -t dengue-app .
docker run --rm -u 0 -v "$PWD:/srv" -w /srv --entrypoint sh dengue-app -c "pip install -r requirements/dev.txt && python -m pytest -q tests"
```

Las pruebas comprueban que la app **reproduce exactamente** el tensor de entrada y las predicciones que exportó el notebook (`modelos/entradas_ejemplo_validacion.npz` y `predicciones_ejemplo_validacion.csv`), el calendario epidemiológico, la validación de entradas y los endpoints de la API.

## Credenciales y seguridad

- La aplicación es de acceso público y no requiere usuario ni contraseña.
- El repositorio no contiene credenciales. Las credenciales de AWS del Learner Lab son temporales y se configuran fuera del repositorio (`~/.aws/credentials`).
- El contenedor se ejecuta con un usuario sin privilegios. La plantilla solo abre el puerto de la app y, de forma opcional, SSH con el par de claves `vockey` del laboratorio.
