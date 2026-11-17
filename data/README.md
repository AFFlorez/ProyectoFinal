# Datos

| Carpeta o archivo | Contenido | En Git |
| --- | --- | --- |
| `DataRaw/Datos_AAAA_210.csv` | Casos individuales de dengue (evento 210) del SIVIGILA, 2007-2025, reducidos al Valle del Cauca y a las columnas COD_DPTO_O, COD_MUN_O, SEMANA, ANO e INI_SIN | Sí |
| `DataRaw/manifiesto_reduccion.json`, `resumen_reduccion.csv` | Registro de la reducción hecha con `notebooks/preparar_datos_dengue.py` | Sí |
| `MunicipiosValledelCauca.csv` | Catálogo de códigos y nombres de los municipios | Sí |
| `DatosProcesados/semanal_training.csv`, `semanal_validation.csv`, `semanal_test.csv` | Serie semanal de los cinco municipios: casos, temperatura media, máxima y mínima, y precipitación. La genera el punto 11 del notebook y la usa la aplicación desplegada | Sí |
| `training.csv`, `validation.csv`, `test.csv` | Casos limpios por periodo (punto 9 del notebook) | No (se regeneran) |
| `clima_google_earth_engine.js` | Código de Google Earth Engine que calcula el clima semanal de los cinco municipios a partir de CHIRPS (precipitación) y ERA5-Land (temperaturas) y exporta un CSV por año | Sí |
| `ShapeMunicipios/CLIMA_DENGUE_VALLE/` | Clima semanal de los cinco municipios, `CLIMA_VALLE_SIVIGILA_AAAA.csv`: precipitación de CHIRPS, temperaturas de ERA5 y días con dato de cada fuente (`DIAS_CHIRPS`, `DIAS_ERA5`) | Sí |
| `DataRawOriginal/` | Excel completos descargados del SIVIGILA | No (tamaño) |

Fuentes:

- Casos: [Portal SIVIGILA – Instituto Nacional de Salud](https://portalsivigila.ins.gov.co/).
- Precipitación: [CHIRPS Daily v2.0](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY?hl=es-419) (`UCSB-CHG/CHIRPS/DAILY`), suma semanal promediada sobre el municipio a 5.566 m.
- Temperatura: [ERA5-Land Daily Aggregated](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR?hl=es-419) (`ECMWF/ERA5_LAND/DAILY_AGGR`): media semanal de `temperature_2m`, máximo de `temperature_2m_max` y mínimo de `temperature_2m_min`, en °C, promediados sobre el municipio a 11.132 m.
- Límites municipales: Marco Geoestadístico Nacional 2018 del DANE, cargado como *asset* en Google Earth Engine (la ruta del *asset* en el código es la de la cuenta que hizo la extracción; hay que cambiarla por la propia).

Con estos archivos, la serie semanal se regenera ejecutando el notebook hasta el punto 11. Las pruebas de `tests/test_historico.py` comprueban que esa serie reproduce las métricas oficiales del modelo.
