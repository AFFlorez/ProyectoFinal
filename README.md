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
| 17 | Entrenamiento de 38 redes globales y 15 locales. | Ajuste con train, checkpoints con validation. |
| 18 | Comparación completa y detalle municipal. | Validación. |
| 19 | Selección provisional, discusión y limitaciones. | Validación. |
| **20** | **Evaluación y análisis final al cerrar la selección. Desactivado por defecto.** | Solo se abre evaluación al activarlo explícitamente. |

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

También se entrenan las referencias LSTM de 64 unidades y de dos capas de 32, ambas con historia de 16 semanas, y una LSTM local por municipio e historia. Son **53 ajustes de redes**: 38 globales y 15 locales. Las cinco redes locales de cada historia forman un enfoque para la comparación. Junto con los nueve baselines, quedan **50 enfoques** en la tabla de validación. Se mantiene una semilla y configuración por receta, sin búsqueda exhaustiva.

Las redes utilizan semilla 42, Adam con tasa inicial de 0,001, lotes de 32, hasta 100 épocas y `shuffle=False`. EarlyStopping conserva los pesos con menor MSE de validación y detiene el ajuste tras 10 épocas sin mejora. ReduceLROnPlateau reduce la tasa tras 5 épocas sin mejora. Una menor pérdida transformada no garantiza un menor MAE en casos.

El punto 14 crea una carpeta nueva en `ResultadosModelos/entrenamiento_en_notebook/ejecucion_*`. Allí se guardan configuración, escaladores y codificador actuales, modelos entrenados, historiales, predicciones de train/validation, métricas generales y municipales, figuras y discusión. El criterio principal es MAE de validación sin redondear; RMSE se informa por separado. Las conclusiones se generan con las métricas de la sesión, sin un ganador ni errores fijados de antemano.

`modelado_notebook.py` contiene las utilidades para construir ventanas, definir redes, invertir transformaciones, calcular métricas, dibujar y exportar. El notebook contiene y ejecuta los ajustes predictivos. `exploracion_temporal.py` se conserva como historial y no se importa en el flujo actual.

## Revisar o repetir

Para retomar desde los conjuntos de casos ya limpios, ejecuta los puntos **1, 2 y 9.2**, y continúa desde el **10**. El punto **9.1** guarda los archivos después de la limpieza; no hace falta ejecutarlo cuando solo quieres cargar `training.csv`, `validation.csv` y `test.csv`. La carga comprueba los periodos y conserva los códigos municipales como texto, sin sobrescribir los archivos. El punto 11 reutiliza esos conjuntos para obtener los conteos semanales.

Esta ruta evita repetir la limpieza; los modelos se entrenan cuando llegas a los puntos 14 y 17. No hay una opción activada por defecto que recupere resultados históricos en lugar de entrenar. Después de entrenar, cambia `HISTORIA_PARA_CURVAS` a 12, 16 o 20 para revisar otra historia sin repetir el ajuste.

Para repetir la comparación, reinicia el kernel y ejecuta el notebook en orden, desde el inicio o retomando los conjuntos limpios como se indica arriba. Cada corrida crea su propia carpeta y conserva las anteriores. Las estructuras están definidas en `modelado_notebook.py`; si se cambia la matriz también deben revisarse las comprobaciones que describen su tamaño.

Mantén `EJECUTAR_EVALUACION_FINAL = False` mientras siga abierta la exploración. La selección y discusión de desarrollo son provisionales. Solo el último bloque utiliza los modelos de la ejecución actual para comparar el candidato y las referencias elegidos por validación con el conjunto de evaluación. Allí quedan también las gráficas de entrenamiento, validación y evaluación y el análisis final. No se ajustan modelos ni escaladores con evaluación.

## Historial conservado

Los respaldos del notebook y las comparaciones históricas se conservan en `ResultadosModelos/respaldos_notebook/`, `ResultadosModelos/evaluacion_final_t4/` y `ResultadosModelos/gru_xgboost_t4/`, junto con las otras exportaciones anteriores.

Esos resultados quedan fuera de la exploración y de la selección actuales. Las exportaciones anteriores se conservaron.
