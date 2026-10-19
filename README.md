# Proyecto final: pronóstico semanal de dengue

El notebook principal es **ProyectoFinal.ipynb**. El objetivo es pronosticar los casos registrados de la semana **t+4**, para Cali, Guadalajara de Buga, Palmira, Tuluá y Yumbo, con historias de **12, 16 y 20 semanas**.

## Abrir el proyecto

Ejecuta `Iniciar-Notebook.cmd` y conserva abierta su terminal. En Jupyter o VS Code selecciona el kernel **Python (ProyectoFinal)**, correspondiente a `.venv/Scripts/python.exe`.

```powershell
.\Iniciar-Notebook.cmd
```

Las dependencias están en `requirements.txt` y las versiones instaladas en `requirements-lock.txt`. Para recrear el entorno utiliza una versión de Python compatible con ese archivo (el proyecto usa Python 3.13), crea `.venv`, instala las dependencias y registra el kernel:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name proyecto-final --display-name "Python (ProyectoFinal)"
```

## Orden actual del notebook

| Punto | Contenido | Datos de modelado |
| --- | --- | --- |
| 1–12 | Carga, municipios, series semanales y transformaciones. La preparación original conserva las tres particiones en archivos separados. | Escaladores ajustados solo con entrenamiento. |
| 13 | Construcción de secuencias con los datos del punto 12. Sin modelos ni métricas. | Train y validation. |
| 14 | Entrenamiento de regresión lineal, Random Forest y XGBoost para las tres historias: nueve ajustes nuevos. | Train; parada de XGBoost con validation. |
| 15 | Métricas y gráficas de los baselines que se acaban de entrenar. | Comparación en validación. |
| 16 | Definición de las estructuras LSTM y GRU. | Sin entrenamiento todavía. |
| 17 | Entrenamiento de 41 redes globales y 15 locales. | Ajuste con train, checkpoints con validation. |
| 18 | Comparación completa y detalle municipal. | Validación. |
| 19 | Selección provisional, discusión y limitaciones. | Validación. |
| **20** | **Evaluación final, gráficas de los tres periodos y exportación del modelo elegido.** | Test 2024–2025, después de fijar la selección. |

**El notebook entrena los modelos desde cero al ejecutar sus celdas.** Las llamadas a `fit` están en los puntos 14 y 17. Las predicciones y métricas se calculan después de esos ajustes. Las exportaciones son salidas de esta corrida; no se necesitan modelos ni resultados anteriores para iniciarla.

La preparación inicial carga y transforma las tres particiones por separado. La exploración predictiva utiliza solamente entrenamiento y validación. El test 2024–2025 ya se observó históricamente; mover su bloque al final no lo vuelve a convertir en una prueba ciega.

## Datos y soporte común

- `DataRaw/`: CSV anuales reducidos al Valle del Cauca (76), con las cinco columnas de vigilancia utilizadas. El notebook mantiene la limpieza de INI_SIN y la selección municipal.
- `DataRawOriginal/`: copia íntegra de los 19 Excel originales. Se conserva localmente y está excluida de Git.
- `ShapeMunicipios/CLIMA_DENGUE_VALLE/`: precipitación semanal y temperaturas media, máxima y mínima.
- `DatosProcesados/semanal_training.csv` y `semanal_validation.csv`: semanas originales de entrenamiento 2007–2021 y validación 2022–2023.
- Los modelos utilizan 3.800 ventanas de entrenamiento (760 por municipio) y 505 de validación (101 por municipio), idénticas entre historias.
- Cada ventana avanza una semana. Para pronosticar t+4 utiliza únicamente información hasta t; el objetivo es una semana específica, no la suma de casos del mes. La validación empieza con emisiones desde el cierre del entrenamiento.
- Las entradas incluyen casos, tres temperaturas, precipitación y seno/coseno de la semana. Las redes globales y baselines agregan el one-hot municipal; las locales mantienen su municipio fijo.
- Se conserva log1p/estandarización municipal para casos y los escaladores de clima de entrenamiento. No se introduce clima futuro ni se reajustan escaladores con validación.

## Archivos de casos reducidos

Los archivos de trabajo se llaman `DataRaw/Datos_AAAA_210.csv`. Conservan `COD_DPTO_O`, `COD_MUN_O`, `SEMANA`, `ANO` e `INI_SIN`, con todos los municipios del departamento 76. La reducción no elimina fechas faltantes, municipios sin correspondencia ni registros repetidos; esas comprobaciones siguen en el notebook.

`datos_dengue.py` centraliza la lectura, conserva los códigos como texto y prefiere los CSV reducidos. También admite los Excel con la hoja `Datos`. El notebook trabaja con `DataRaw`; no abre automáticamente la copia completa de `DataRawOriginal`.

La conversión puede reproducirse con:

```powershell
.venv\Scripts\python.exe preparar_datos_dengue.py
```

La primera ejecución copia y verifica los originales antes de sustituir los archivos de trabajo. Las siguientes comprueban el manifiesto y los hashes sin volver a convertirlos. `DataRaw/manifiesto_reduccion.json` y `DataRaw/resumen_reduccion.csv` registran tamaños, filas y verificaciones por año. Git permite incluir esos CSV reducidos y sus registros; los Excel completos de `DataRawOriginal` y cualquier otro archivo de `DataRaw` quedan excluidos.

La gráfica inicial muestra ahora únicamente Valle del Cauca, porque los CSV reducidos ya no contienen otros departamentos. Los conjuntos municipales y las series utilizadas por los modelos deben conservar los mismos valores.

## Exploración ampliada

La matriz prueba LSTM y GRU con una capa de 32 unidades y Dense16 como referencia; sin Dense intermedia; Dropout de 0,20 y 0,50; dos capas 32→16 con Dropout de 0,20; y tres capas 32→16→8 con Dropout de 0,20. Cada estructura se prueba con 12, 16 y 20 semanas.

También se entrenan las referencias LSTM de 64 unidades y de dos capas de 32, ambas con historia de 16 semanas, y una LSTM local por municipio e historia. Añadimos además LSTM16 + Dense8 sin Dropout para las historias de 12, 16 y 20 semanas. Son **56 ajustes de redes**: 41 globales y 15 locales. Las cinco redes locales de cada historia forman un enfoque para la comparación. Junto con los nueve baselines, quedan **53 enfoques** en la tabla de validación. Se mantiene una semilla y configuración por receta, sin búsqueda exhaustiva.

Las redes utilizan semilla 42, Adam con tasa inicial de 0,001, lotes de 32, hasta 100 épocas y `shuffle=False`. EarlyStopping conserva los pesos con menor MSE de validación y detiene el ajuste tras 10 épocas sin mejora. ReduceLROnPlateau reduce la tasa tras 5 épocas sin mejora. Una menor pérdida transformada no garantiza un menor MAE en casos.

El punto 14 crea una carpeta nueva en `ResultadosModelos/entrenamiento_en_notebook/ejecucion_*`. Allí se guardan configuración, escaladores y codificador actuales, modelos entrenados, historiales, predicciones de train/validation, métricas generales y municipales y figuras, incluida la comparación de LSTM16 + Dense8 con LSTM32 + Dense16 por historia. El ranking general se ordena por MAE de validación sin redondear y conserva el mínimo de RMSE como referencia. La discusión y las conclusiones están escritas directamente en celdas Markdown a partir de los resultados guardados; deben revisarse al cambiar la corrida o la selección.

`modelado_notebook.py` contiene las utilidades para construir ventanas, definir redes, invertir transformaciones, calcular métricas, dibujar y exportar. El notebook contiene y ejecuta los ajustes predictivos. `exploracion_temporal.py` se conserva como historial y no se importa en el flujo actual.

Desde el punto 18.2 se elige explícitamente **LSTM32 + Dense16 + Dropout de 0,20, con historia de 20 semanas**. La arquitectura se escoge por la preferencia de trabajar con regularización; entre sus tres historias, 20 semanas obtuvo los menores MAE y RMSE de validación. El mínimo general de MAE sigue identificado como referencia. Se conserva la selección previa en `seleccion_hasta_18_1.json` y la decisión actual, con su motivo, en `seleccion_validacion.json`. Esta preferencia no se presenta como una demostración de mejor generalización.

El punto 18.2 utiliza `analisis_validacion.py` para ampliar la lectura de las predicciones ya calculadas: compara el modelo elegido con el mínimo de MAE, la versión sin Dropout de la misma historia, las familias, los municipios y el mejor baseline general. También analiza los años de validación y los mayores errores y conserva el detalle municipal de las seis mejores configuraciones. Las tablas, las figuras municipales y las series del candidato se guardan en `analisis_validacion/` dentro de la carpeta de la corrida. Este análisis no entrena modelos ni consulta evaluación. Con las variables de la sesión disponibles, se pueden actualizar los puntos 18.2 y 19 después de ejecutar el punto 2, que recarga las utilidades.

## Revisar o repetir

Para retomar desde los conjuntos de casos ya limpios, ejecuta los puntos **1, 2 y 9.2**, y continúa desde el **10**. El punto **9.1** guarda los archivos después de la limpieza; no hace falta ejecutarlo cuando solo quieres cargar `training.csv`, `validation.csv` y `test.csv`. La carga comprueba los periodos y conserva los códigos municipales como texto, sin sobrescribir los archivos. El punto 11 reutiliza esos conjuntos para obtener los conteos semanales.

Esta ruta evita repetir la limpieza; los modelos se entrenan cuando llegas a los puntos 14 y 17. No hay una opción activada por defecto que recupere resultados históricos en lugar de entrenar. Después de entrenar, cambia `HISTORIA_PARA_CURVAS` a 12, 16 o 20 para revisar otra historia sin repetir el ajuste.

Para repetir la comparación, reinicia el kernel y ejecuta el notebook en orden, desde el inicio o retomando los conjuntos limpios como se indica arriba. Cada corrida crea su propia carpeta y conserva las anteriores. Las estructuras están definidas en `modelado_notebook.py`; si se cambia la matriz también deben revisarse las comprobaciones que describen su tamaño.

El punto 20 está habilitado para la evaluación solicitada. No lo ejecutes durante la exploración de arquitecturas: primero fija la selección con validación. Este bloque utiliza el candidato registrado en 18.2 y las referencias de menor MAE por familia de la misma ejecución. Guarda las métricas generales, municipales y anuales, las gráficas de entrenamiento, validación y evaluación y el modelo elegido. No se ajustan modelos ni escaladores con evaluación.

La evaluación realizada está en `ResultadosModelos/entrenamiento_en_notebook/ejecucion_93868d0d/evaluacion_al_cierre/ejecucion_3cb61062/`. La LSTM elegida obtuvo MAE 41.166 y RMSE 106.534 en 510 objetivos de test; la regresión lineal de 12 semanas obtuvo los menores errores entre las siete referencias evaluadas (31.405 y 79.162). La selección previa se conserva. El análisis del punto 20 está en Markdown y reconoce que este test ya se había consultado históricamente.

Dentro de esa carpeta, `modelo_elegido/modelo.keras` contiene la arquitectura y los pesos de LSTM32 + Dense16 + Dropout 0,20 e historia de 20 semanas. Lo acompañan `preprocesamiento.joblib`, `configuracion.json`, las dependencias, un ejemplo de validación y sus predicciones, instrucciones de carga y hashes SHA-256. Se comprobó que el archivo exportado reproduce las predicciones originales para un ejemplo de cada municipio.

## Historial conservado

Los respaldos del notebook y las comparaciones históricas se conservan en `ResultadosModelos/respaldos_notebook/`, `ResultadosModelos/evaluacion_final_t4/` y `ResultadosModelos/gru_xgboost_t4/`, junto con las otras exportaciones anteriores.

Esos resultados quedan fuera de la exploración y de la selección actuales. Las exportaciones anteriores se conservaron.
