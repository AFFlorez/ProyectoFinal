"""Inferencia del paquete exportado de dengue. No ajusta modelos ni escaladores."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


class PronosticadorDengue:
    """Recibe 16 semanas completas de un municipio y pronostica su semana t+4.

    Los casos y el clima de la última semana deben estar disponibles al emitir.
    La salida principal conserva los valores continuos del modelo, sin recorte.
    """

    def __init__(self, carpeta):
        self.carpeta = Path(carpeta)
        self.config = json.loads((self.carpeta / 'configuracion.json').read_text(encoding='utf-8'))
        self.prep = joblib.load(self.carpeta / 'preprocesamiento.joblib')
        self.lstm = tf.keras.models.load_model(self.carpeta / 'modelo_lstm_global_t4.keras', compile=False)
        self.baselines = {
            'Regresión lineal': joblib.load(self.carpeta / 'regresion_lineal_t4.joblib'),
            'Random Forest': joblib.load(self.carpeta / 'random_forest_t4.joblib'),
        }
        self.columnas = self.config['columnas_temporales']

    def transformar_semanas(self, semanas, semanas_por_ano=None):
        """Aplicar parámetros congelados a un panel semanal, sin modificarlo."""
        d = semanas.copy(deep=True).reset_index(drop=True)
        clima = ['TEMP_MEDIA_C', 'TEMP_MAX_C', 'TEMP_MIN_C', 'PRECIPITACION_MM']
        requeridas = ['COD_DANE', 'ANO', 'SEMANA_EPI', 'FECHA_INICIO', 'FECHA_FIN', 'CASOS', *clima]
        ausentes = sorted(set(requeridas) - set(d.columns))
        if ausentes:
            raise ValueError(f'Faltan columnas: {ausentes}')
        if d.empty:
            raise ValueError('Se requieren semanas históricas.')
        d['COD_DANE'] = d['COD_DANE'].astype('string')
        if not d['COD_DANE'].isin(self.config['municipios']).all():
            raise ValueError('Hay municipios no incluidos en el entrenamiento.')
        if d[requeridas].isna().any().any():
            raise ValueError('Hay datos faltantes; no se imputa durante inferencia.')
        for c in ['FECHA_INICIO', 'FECHA_FIN']:
            d[c] = pd.to_datetime(d[c], errors='raise')
        for c in ['ANO', 'SEMANA_EPI', 'CASOS', *clima]:
            d[c] = pd.to_numeric(d[c], errors='raise')
        if not np.isfinite(d[['ANO', 'SEMANA_EPI', 'CASOS', *clima]].to_numpy(float)).all():
            raise ValueError('Hay valores infinitos o no numéricos.')
        if not np.equal(d[['ANO', 'SEMANA_EPI', 'CASOS']], np.floor(d[['ANO', 'SEMANA_EPI', 'CASOS']])).all().all():
            raise ValueError('Año, semana y casos deben ser enteros.')
        if d[['CASOS', 'PRECIPITACION_MM']].lt(0).any().any():
            raise ValueError('Casos y precipitación no pueden ser negativos.')
        if not (d['TEMP_MIN_C'].le(d['TEMP_MEDIA_C']) & d['TEMP_MEDIA_C'].le(d['TEMP_MAX_C'])).all():
            raise ValueError('Las temperaturas mínima, media y máxima son inconsistentes.')
        if d.duplicated(['COD_DANE', 'FECHA_INICIO']).any() or d.duplicated(['COD_DANE', 'ANO', 'SEMANA_EPI']).any():
            raise ValueError('Hay semanas repetidas.')
        if not (d['FECHA_FIN'] - d['FECHA_INICIO']).dt.days.eq(6).all():
            raise ValueError('Cada semana debe abarcar siete días.')
        calendario = {int(k): int(v) for k, v in self.config['semanas_por_ano'].items()}
        if semanas_por_ano:
            calendario.update({int(k): int(v) for k, v in semanas_por_ano.items()})
        if not set(calendario.values()).issubset({52, 53}):
            raise ValueError('El calendario debe indicar 52 o 53 semanas por año.')
        denominador = d['ANO'].map(calendario)
        if denominador.isna().any():
            raise ValueError('Falta el calendario epidemiológico de un año. Proporcionar semanas_por_ano verificado; no se extrapola.')
        if not d['SEMANA_EPI'].between(1, denominador).all():
            raise ValueError('Semana fuera del calendario suministrado.')
        # Verificar también que año, semana y fechas representen el mismo calendario.
        calendario_guardado = pd.read_csv(self.carpeta / 'calendario_semanal.csv', parse_dates=['FECHA_INICIO', 'FECHA_FIN'])
        conocidas = d.loc[d['ANO'].isin(calendario_guardado['ANO'])]
        cotejo = conocidas.merge(calendario_guardado, on=['ANO', 'SEMANA_EPI'], how='left', validate='many_to_one', suffixes=('', '_calendario'))
        for c in ['FECHA_INICIO', 'FECHA_FIN']:
            if not cotejo[c].eq(cotejo[c + '_calendario']).all():
                raise ValueError('Las fechas no coinciden con el calendario epidemiológico exportado.')
        d['CASOS_std'] = np.nan
        for codigo, g in d.groupby('COD_DANE', observed=True):
            d.loc[g.index, 'CASOS_std'] = self.prep['escaladores_casos'][codigo].transform(np.log1p(g[['CASOS']].to_numpy(float))).ravel()
        temps = ['TEMP_MEDIA_C', 'TEMP_MAX_C', 'TEMP_MIN_C']
        d[[c + '_std' for c in temps]] = self.prep['escaladores_clima']['temperaturas'].transform(d[temps].to_numpy(float))
        d['PRECIPITACION_MM_std'] = self.prep['escaladores_clima']['precipitacion'].transform(np.log1p(d[['PRECIPITACION_MM']].to_numpy(float))).ravel()
        fase = 2 * np.pi * (d['SEMANA_EPI'].to_numpy(float) - 1) / denominador.to_numpy(float)
        d['semana_sin'], d['semana_cos'] = np.sin(fase), np.cos(fase)
        return d

    def invertir(self, valores, codigos):
        z = np.asarray(valores, dtype=float).reshape(-1, 1)
        codigos = pd.Series(np.asarray(codigos).astype(str))
        if len(z) != len(codigos) or not np.isfinite(z).all():
            raise ValueError('Predicciones y códigos incompatibles o no finitos.')
        salida = np.empty(len(z), dtype=float)
        for codigo, posiciones in codigos.groupby(codigos).indices.items():
            salida[posiciones] = np.expm1(self.prep['escaladores_casos'][codigo].inverse_transform(z[posiciones])).ravel()
        if not np.isfinite(salida).all():
            raise ValueError('Desbordamiento al invertir las predicciones.')
        return salida

    def predecir_tensores(self, X, codigos):
        """Pronósticos estandarizados; mismo orden de filas que las entradas."""
        X = np.asarray(X, dtype=np.float32)
        codigos = pd.Series(np.asarray(codigos).astype(str), name='COD_DANE', dtype='string')
        if X.shape != (len(codigos), 16, 7) or not len(X) or not np.isfinite(X).all():
            raise ValueError('Se espera un tensor finito (N, 16, 7), con N positivo.')
        aux = self.prep['codificador_municipios'].transform(codigos.to_frame())
        if hasattr(aux, 'toarray'):
            aux = aux.toarray()
        batches = []
        for i in range(0, len(X), 32):
            batches.append(self.lstm({'historia': X[i:i+32], 'municipio': aux[i:i+32]}, training=False).numpy())
        matriz = np.concatenate([X.reshape(len(X), -1).astype(float), aux], axis=1)
        return {
            'LSTM global': np.concatenate(batches).astype(float),
            **{nombre: np.asarray(m.predict(matriz), dtype=float).reshape(-1, 1) for nombre, m in self.baselines.items()},
        }

    def pronosticar(self, historia, semanas_por_ano=None):
        """Una predicción por municipio usando sus últimas 16 semanas disponibles.

        Para años nuevos debe suministrarse el número de semanas epidemiológicas
        verificado (52/53). El calendario histórico exportado se valida por fechas.
        """
        datos = self.transformar_semanas(historia, semanas_por_ano)
        X, filas = [], []
        for codigo, g in datos.groupby('COD_DANE', observed=True, sort=True):
            g = g.sort_values('FECHA_INICIO').tail(16)
            if len(g) != 16 or not g['FECHA_INICIO'].diff().dropna().dt.days.eq(7).all():
                raise ValueError(f'{codigo}: se requieren 16 semanas consecutivas.')
            X.append(g[self.columnas].to_numpy(np.float32))
            ultima = g.iloc[-1]
            filas.append({'COD_DANE': codigo, 'MUNICIPIO': self.config['municipios'][codigo],
                          'FECHA_EMISION': ultima['FECHA_FIN'],
                          'FECHA_OBJETIVO': ultima['FECHA_INICIO'] + pd.Timedelta(weeks=4), 'HORIZONTE': 4})
        meta = pd.DataFrame(filas)
        tablas = []
        for nombre, z in self.predecir_tensores(np.stack(X), meta['COD_DANE']).items():
            tabla = meta.copy()
            tabla.insert(0, 'MODELO', nombre)
            tabla['CASOS_PREDICHOS'] = self.invertir(z, tabla['COD_DANE'])
            tabla['PRONOSTICO_NEGATIVO'] = tabla['CASOS_PREDICHOS'].lt(0)
            tablas.append(tabla)
        return pd.concat(tablas, ignore_index=True)
