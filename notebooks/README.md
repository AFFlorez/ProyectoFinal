# Pronóstico semanal de dengue

El proyecto compara modelos para pronosticar los casos de una semana específica, **t+4**, en Cali, Guadalajara de Buga, Palmira, Tuluá y Yumbo. Las historias son de **8, 12, 16, 20 y 24 semanas**. El modelo seleccionado es el que obtiene el **menor MAE de validación sin redondear**; no se mantiene una preferencia previa por arquitectura o Dropout.

## Datos y separación temporal

- Entrenamiento: 2007–2021.
- Validación: 2022–2023, utilizada para comparar configuraciones y escoger el modelo.
- Test: 2024–2025, evaluado al cerrar la selección. Este periodo ya se había observado en etapas anteriores; la evaluación actual no se presenta como una prueba ciega.

Cada ventana avanza una semana y utiliza casos y clima observados hasta t. Todas las historias comparten **3.780 objetivos de entrenamiento y 505 de validación**. Cada red local usa 756 y 101, respectivamente. El soporte común parte de la historia más larga, de 24 semanas. Las ventanas se solapan, por lo que no son observaciones independientes.

Las siete variables temporales son casos, temperaturas media/máxima/mínima, precipitación y seno/coseno de la semana epidemiológica. Los modelos globales agregan el one-hot municipal; los locales tienen un municipio fijo. Los escaladores se ajustan únicamente con entrenamiento. Las métricas se calculan en casos, deshaciendo la transformación de salida con el escalador de cada municipio, sin redondear ni recortar los pronósticos.

## Modelos comparados

| Grupo | Configuraciones | Ajustes predictivos |
| --- | --- | ---: |
| Baselines | Regresión lineal, Random Forest y XGBoost, con las cinco historias | 15 |
| LSTM globales | Catorce estructuras, con las cinco historias | 70 |
| LSTM locales | Tres estructuras × cinco historias × cinco municipios | 75 |
| Total | 100 enfoques al reunir las cinco redes de cada enfoque local | **160** |

Las estructuras globales incluyen la referencia LSTM32 + Dense16, la variante sin Dense intermedia, Dropout de 0,20 y 0,50, dos y tres capas, LSTM16 + Dense8, seis opciones LSTM64 (Dense16, Dense8 y sin Dense intermedia, con Dropout de 0 y 0,20) y dos capas de 32 con Dense16. Todas conservan una salida lineal de una neurona y reciben el one-hot municipal.

Los locales comparan LSTM32 + Dense16, LSTM16 + Dense8 y LSTM8 + Dense4, con **5.665, 1.681 y 553 parámetros por red**, sin Dropout. Cada combinación de estructura e historia reúne cinco redes independientes.

Usamos semilla 42. Las LSTM se ajustan con Adam, tasa inicial de 0,001, lotes de 32, hasta 200 épocas y orden sin barajar. EarlyStopping recupera los pesos con menor MSE de validación y detiene el ajuste tras veinte épocas sin mejora; la tasa se reduce tras cinco épocas sin mejora. La comparación posterior utiliza MAE y RMSE en casos, no la pérdida transformada de entrenamiento.

Los baselines conservan sus configuraciones iniciales: regresión lineal predeterminada, Random Forest de 300 árboles con profundidad máxima 12 y mínimo tres ejemplos por hoja, y XGBoost de hasta 800 rondas, profundidad 3, tasa 0,03 y parada anticipada de 50 rondas con validación. No se hace una búsqueda exhaustiva de hiperparámetros.

## Orden del notebook

| Punto | Contenido |
| --- | --- |
| 1–12 | Lectura, limpieza, selección municipal, series semanales y transformaciones |
| 13 | Secuencias de las cinco historias, sin entrenar modelos |
| 14–15 | Entrenamiento y validación de los 15 baselines |
| 16–17 | Definición y entrenamiento de las 145 redes LSTM |
| 18 | Ranking de los 100 enfoques, detalle del seleccionado y comparación local |
| 19 | Discusión de validación y limitaciones, escrita en Markdown |
| 20 | Test, comparación de los tres periodos, exportación y análisis final |

Una ejecución completa entrena todos los modelos desde cero. Las llamadas a entrenamiento están en los puntos 14 y 17. Los modelos y las métricas de corridas anteriores no son requisitos para iniciar una nueva. Las conclusiones están en celdas Markdown y deben revisarse cuando cambien los resultados.

Para retomar desde los conjuntos limpios, ejecuta **1, 2 y 9.2** y continúa desde el **10**. El punto 9.1 guarda los archivos después de limpiar; 9.2 los carga sin repetir ese trabajo. HISTORIA_PARA_CURVAS permite revisar cualquiera de las cinco historias después de entrenar.

## Archivos del proyecto

El notebook y sus módulos están en `notebooks/`, pero trabajan desde la raíz del repositorio: la primera celda cambia el directorio de trabajo a la raíz. Los datos están en `data/` (ver [data/README.md](../data/README.md)). Los scripts de esta carpeta se ejecutan desde la raíz, por ejemplo `.venv\Scripts\python.exe notebooks\analisis_validacion.py`.

- `data/DataRaw/`: CSV anuales reducidos al departamento 76 y a las cinco columnas utilizadas: COD_DPTO_O, COD_MUN_O, SEMANA, ANO e INI_SIN.
- `data/DataRawOriginal/`: copia de los archivos completos, conservada localmente y excluida de Git.
- `data/ShapeMunicipios/CLIMA_DENGUE_VALLE/`: clima semanal por municipio. Se conserva fuera de Git por su tamaño.
- `data/DatosProcesados/`: series semanales separadas en entrenamiento, validación y test. Solo `semanal_*.csv` (cinco municipios, menos de 1 MB) se versiona, porque la aplicación desplegada la usa como histórico.
- `notebooks/modelado_notebook.py`: construcción de ventanas, arquitecturas, métricas, gráficas y exportación.
- `notebooks/analisis_validacion.py`: diagnóstico del modelo seleccionado a partir de sus predicciones de validación.
- `notebooks/inferencia_dengue.py`: versión anterior de la inferencia (historia de 16 semanas). La aplicación usa `app/pronosticador.py`, adaptado al modelo final de 20 semanas.
- `modelos/`: paquete del modelo elegido que se incluye en Git, con pesos, escaladores, configuración y ejemplo de carga. El punto 20.4 actualiza esta carpeta.
- `ResultadosModelos/entrenamiento_en_notebook/ejecucion_*/`: modelos, configuración, escaladores, historiales, predicciones, tablas, figuras y verificaciones de cada corrida.

Dentro de cada corrida, `analisis_locales/` conserva la comparación municipal y por tamaño. `evaluacion_al_cierre/` contiene el test y su carpeta `modelo_elegido/`, con los pesos, el preprocesamiento, la configuración, un ejemplo comprobado y un script de carga. Si el elegido es local, la exportación incluye sus cinco redes. Las corridas completas se mantienen fuera de Git. El paquete elegido se copia a `modelos/`, cuyos archivos están permitidos en el `.gitignore`. Para comprobar su carga desde la raíz del proyecto, ejecuta `.venv\Scripts\python.exe modelos\predecir_ejemplo.py`.

Los CSV reducidos conservan los registros del Valle del Cauca; la limpieza de INI_SIN y de municipios sigue en el notebook. `datos_dengue.py` centraliza su lectura. `preparar_datos_dengue.py` permite repetir la reducción y verifica la copia original y el manifiesto antes de sustituir archivos de trabajo.

## Entorno

El proyecto utiliza Python 3.13 y el entorno `.venv`. Las dependencias están en `requirements/notebook.txt` y las versiones instaladas en `requirements/notebook-lock.txt`. La aplicación desplegada usa otro archivo, `requirements/app.txt`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements
otebook-lock.txt
.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name proyecto-final --display-name "Python (ProyectoFinal)"
.
otebooks\Iniciar-Notebook.cmd
```

Los respaldos del notebook y los resultados de etapas anteriores se conservan en `ResultadosModelos/`; no participan en la selección de la nueva corrida.


## Resultados de la nueva corrida

El menor MAE de validación fue el de **LSTM 64 · Dense16 · D0,0 · 20s**, que queda seleccionado. Todas las configuraciones se entrenaron desde cero con historias de 8, 12, 16, 20 y 24 semanas.

| Conjunto | Objetivos | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Entrenamiento | 3.780 | 7.743 | 21.898 |
| Validación | 505 | 11.377 | 26.837 |
| Test | 510 | 38.139 | 95.413 |

La corrida está en `ResultadosModelos/entrenamiento_en_notebook/ejecucion_5a3d407a`. La evaluación y el modelo exportado están en `ResultadosModelos/entrenamiento_en_notebook/ejecucion_5a3d407a/evaluacion_al_cierre/ejecucion_ecefe053`. VERIFICACION_ENTRENAMIENTO.json registra las comprobaciones de datos, métricas y selección.

La referencia local con menor MAE de validación fue **LSTM local 16 · Dense8 (5 redes) · 24s**: MAE 12.872 y RMSE 30.108. El análisis municipal, las curvas y la comparación de tamaños están en el punto 18.3 del notebook.

Entre las cinco referencias evaluadas, el menor MAE de test fue el de **Regresión lineal · 12s**, con MAE 31.285 y RMSE 78.708. La selección se mantiene fijada con validación; el periodo de test ya era conocido y no se utiliza para volver a escoger.
