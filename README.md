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
| 13 | Protocolo y recuperación verificada de secuencias. | Train y validation. |
| 14 | Comparación inicial de historias como antecedente. | Validación. |
| 15 | Regresión lineal, Random Forest y XGBoost para las tres historias. | Train y validation. |
| 16 | Referencias LSTM globales, locales y GRU ya entrenadas. | Validación. |
| 17 | Matriz de seis estructuras × dos familias × tres historias; curvas. | Ajuste con train, checkpoints con validation. |
| 18 | Comparación completa y detalle municipal. | Validación. |
| 19 | Selección provisional, discusión y limitaciones. | Validación. |
| **20** | **Evaluación y análisis final al cerrar la selección. Desactivado por defecto.** | Solo se abre evaluación al activarlo explícitamente. |

**La exploración actual no leyó el test.** El test 2024–2025 ya se observó históricamente; mover su bloque al final no lo vuelve a convertir en una prueba ciega. La preparación original de archivos no se utiliza para ajustar o comparar los modelos con sus resultados de test.

## Datos y soporte común

- `DataRaw/`: CSV anuales reducidos al Valle del Cauca (76), con las cinco columnas de vigilancia utilizadas. El notebook mantiene la limpieza de INI_SIN y la selección municipal.
- `DataRawOriginal/`: copia íntegra de los 19 Excel originales. Se conserva localmente y está excluida de Git.
- `ShapeMunicipios/CLIMA_DENGUE_VALLE/`: precipitación semanal y temperaturas media, máxima y mínima.
- `DatosProcesados/semanal_training.csv` y `semanal_validation.csv`: semanas originales de entrenamiento 2007–2021 y validación 2022–2023.
- Los modelos utilizan 3.800 ventanas de entrenamiento (760 por municipio) y 505 de validación (101 por municipio), idénticas entre historias.
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

Hay 36 combinaciones recurrentes y 50 configuraciones en la comparación completa al añadir los baselines y referencias anteriores. Se entrenaron **31 redes y dos XGBoost nuevos**; los demás ajustes se reutilizaron. Se mantienen una semilla y configuración por receta, sin búsqueda exhaustiva.

Los resultados actuales están en [exploración de entrenamiento y validación](ResultadosModelos/exploracion_temporal_validacion/ejecucion_291cbf82/README.md): modelos, predicciones de train/validation, métricas generales y municipales, diez figuras, configuración previa, selección provisional y comprobaciones. El criterio principal es MAE de validación sin redondear; RMSE se informa por separado.

## Revisar o repetir

Por defecto `REENTRENAR_EXPLORACION = False` recupera los ajustes ejecutados. No hace falta reentrenar para revisar tablas o figuras. Cambia `HISTORIA_PARA_CURVAS` a 12, 16 o 20 y ejecuta esa celda para ver las curvas de otra historia.

Para otra ejecución modifica las recetas del módulo `exploracion_temporal.py` y activa el reentrenamiento. El bloque crea una carpeta nueva, conserva los resultados previos y utiliza solo entrenamiento y validación. La matriz actual está fijada y verificada; si se cambia su diseño también deberán revisarse las comprobaciones que describen su tamaño.

Mantén `EJECUTAR_EVALUACION_FINAL = False` mientras siga abierta la exploración. La selección y discusión de desarrollo son provisionales. El último bloque está preparado para comparar el candidato congelado y las mejores referencias por familia, con el análisis y las gráficas finales, una vez que se cierre esa selección. Esa nueva ejecución no se realizó en esta exploración.

## Historial conservado

- [Notebook anterior a la reorganización](ResultadosModelos/respaldos_notebook/antes_reorganizar_desarrollo_evaluacion.ipynb).
- [Evaluación histórica de LSTM y baselines](ResultadosModelos/evaluacion_final_t4/ejecucion_7456e638/README.md).
- [Comparación histórica con GRU y XGBoost](ResultadosModelos/gru_xgboost_t4/ejecucion_04dafbf5/README.md).

Esos resultados quedan fuera de la exploración y de la selección actuales. Las exportaciones anteriores se conservaron.
