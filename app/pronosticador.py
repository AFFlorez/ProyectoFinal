"""Carga del modelo seleccionado (modelos/) y pronóstico de los casos de la semana t+4.

Reproduce las transformaciones del notebook con los escaladores ajustados en
entrenamiento: no se reajusta nada durante la inferencia.
"""
import hashlib
import json
import logging
import time
import zipfile
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from statistics import NormalDist
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd

from app.calendario import semana_epidemiologica, semanas_del_ano

log = logging.getLogger(__name__)

TEMPERATURAS = ['TEMP_MEDIA_C', 'TEMP_MAX_C', 'TEMP_MIN_C']
CLIMA = [*TEMPERATURAS, 'PRECIPITACION_MM']
MUNICIPIOS = {
    '76001': 'Cali',
    '76111': 'Guadalajara de Buga',
    '76520': 'Palmira',
    '76834': 'Tuluá',
    '76892': 'Yumbo',
}
ARCHIVOS_VERIFICADOS = ['modelo.keras', 'preprocesamiento.joblib']
ARCHIVOS_REQUERIDOS = ['configuracion.json', *ARCHIVOS_VERIFICADOS]


class ErrorEntrada(ValueError):
    """Datos de entrada inválidos; el mensaje se muestra al usuario."""


def descargar_modelo(url, carpeta):
    """Descargar un .zip con el paquete del modelo cuando no está en el repositorio."""
    carpeta.mkdir(parents=True, exist_ok=True)
    log.info('Descargando el modelo desde %s', url.split('?')[0])
    with urlopen(url, timeout=120) as respuesta:
        contenido = respuesta.read()
    with zipfile.ZipFile(BytesIO(contenido)) as paquete:
        for nombre in paquete.namelist():
            destino = carpeta / Path(nombre).name
            if Path(nombre).name in {*ARCHIVOS_REQUERIDOS, 'sha256.json'}:
                destino.write_bytes(paquete.read(nombre))


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


class Pronosticador:
    def __init__(self, carpeta, verificar_integridad=True, url_modelo=''):
        inicio = time.perf_counter()
        self.carpeta = Path(carpeta)
        if url_modelo and not all((self.carpeta / a).is_file() for a in ARCHIVOS_REQUERIDOS):
            descargar_modelo(url_modelo, self.carpeta)
        faltantes = [a for a in ARCHIVOS_REQUERIDOS if not (self.carpeta / a).is_file()]
        if faltantes:
            raise FileNotFoundError(f'Faltan archivos del modelo en {self.carpeta}: {faltantes}. Ver README, sección "Gestión del modelo".')
        if verificar_integridad:
            self._verificar_integridad()

        self.config = json.loads((self.carpeta / 'configuracion.json').read_text(encoding='utf-8-sig'))
        if self.config['tipo'] != 'red':
            raise ValueError(f"La app sirve el modelo global LSTM exportado; tipo recibido: {self.config['tipo']}")
        self.prep = joblib.load(self.carpeta / 'preprocesamiento.joblib')
        self.historia = int(self.config['historia_semanas'])
        self.horizonte = int(self.config['horizonte_semanas'])
        self.columnas = list(self.config['columnas_temporales'])
        self.orden = [str(c) for c in self.config['orden_one_hot']]
        categorias = [str(c) for c in self.prep['codificador_municipios'].categories_[0]]
        if categorias != self.orden or set(self.orden) != set(MUNICIPIOS):
            raise ValueError('El codificador municipal no coincide con la configuración exportada.')

        import tensorflow as tf  # Importación diferida: solo al cargar el modelo.
        tf.get_logger().setLevel('ERROR')
        self.modelo = tf.keras.models.load_model(self.carpeta / self.config['archivos'], compile=False)
        self.parametros = int(self.modelo.count_params())
        self.predecir(np.zeros((1, self.historia, len(self.columnas)), np.float32), [self.orden[0]])
        self.segundos_carga = time.perf_counter() - inicio
        log.info('Modelo %s cargado en %.2f s (%d parámetros)', self.config['nombre'], self.segundos_carga, self.parametros)

    def _verificar_integridad(self):
        esperados = json.loads((self.carpeta / 'sha256.json').read_text(encoding='utf-8-sig'))
        for archivo in ARCHIVOS_VERIFICADOS:
            if sha256(self.carpeta / archivo) != esperados.get(archivo):
                raise ValueError(f'{archivo} no coincide con sha256.json: el paquete del modelo está alterado o incompleto.')

    # ---- Transformaciones -------------------------------------------------
    def transformar(self, semanas):
        """Matriz (filas, 7) en el orden de columnas_temporales, con los escaladores de entrenamiento."""
        d = semanas
        valores = {}
        casos_std = np.empty(len(d))
        for codigo, posiciones in d.groupby('COD_DANE', sort=False).indices.items():
            log_casos = np.log1p(d['CASOS'].to_numpy(float)[posiciones]).reshape(-1, 1)
            casos_std[posiciones] = self.prep['escaladores_casos'][codigo].transform(log_casos).ravel()
        valores['CASOS_std'] = casos_std
        temperaturas = self.prep['escaladores_clima']['temperaturas'].transform(d[TEMPERATURAS].to_numpy(float))
        for i, columna in enumerate(TEMPERATURAS):
            valores[f'{columna}_std'] = temperaturas[:, i]
        lluvia = np.log1p(d[['PRECIPITACION_MM']].to_numpy(float))
        valores['PRECIPITACION_MM_std'] = self.prep['escaladores_clima']['precipitacion'].transform(lluvia).ravel()
        # Semana 1 tiene fase cero; el denominador es el número de semanas de ese año.
        denominador = d['ANO'].map(semanas_del_ano).to_numpy(float)
        fase = 2 * np.pi * (d['SEMANA_EPI'].to_numpy(float) - 1) / denominador
        valores['semana_sin'], valores['semana_cos'] = np.sin(fase), np.cos(fase)
        return np.column_stack([valores[c] for c in self.columnas]).astype(np.float32)

    def predecir(self, X, codigos):
        """Casos continuos de t+4 para tensores (N, historia, 7); sin redondear ni recortar."""
        X = np.asarray(X, dtype=np.float32)
        codigos = np.asarray(codigos).astype(str)
        if X.ndim != 3 or X.shape[1:] != (self.historia, len(self.columnas)) or len(X) != len(codigos):
            raise ValueError(f'Se espera un tensor (N, {self.historia}, {len(self.columnas)}).')
        municipio = (codigos[:, None] == np.array(self.orden)[None, :]).astype(np.float32)
        z = np.concatenate([
            self.modelo({'historia': X[i:i + 512], 'municipio': municipio[i:i + 512]}, training=False).numpy()
            for i in range(0, len(X), 512)
        ]).astype(float)
        casos = np.empty(len(z))
        for codigo in np.unique(codigos):
            mascara = codigos == codigo
            casos[mascara] = np.expm1(self.prep['escaladores_casos'][codigo].inverse_transform(z[mascara])).ravel()
        return casos

    # ---- Interpretación ---------------------------------------------------
    def umbrales(self, codigo, percentil_moderado, percentil_alto):
        """Casos semanales que marcan alerta moderada/alta según la distribución de entrenamiento.

        El escalador guarda la media y desviación de log1p(casos) de 2007-2021; los
        percentiles se aproximan con una normal en esa escala logarítmica.
        """
        escalador = self.prep['escaladores_casos'][codigo]
        media, desviacion = float(escalador.mean_[0]), float(escalador.scale_[0])
        return {
            'moderado': float(np.expm1(media + NormalDist().inv_cdf(percentil_moderado / 100) * desviacion)),
            'alto': float(np.expm1(media + NormalDist().inv_cdf(percentil_alto / 100) * desviacion)),
        }

    @staticmethod
    def nivel_alerta(casos, umbrales):
        if casos >= umbrales['alto']:
            return 'alto'
        if casos >= umbrales['moderado']:
            return 'moderado'
        return 'bajo'

    # ---- Entradas del usuario --------------------------------------------
    def preparar_entrada(self, datos):
        """Validar semanas suministradas por el usuario y devolver las últimas `historia` por municipio."""
        requeridas = ['COD_DANE', 'FECHA_INICIO', 'CASOS', *CLIMA]
        d = datos.rename(columns=lambda c: str(c).strip().upper()).copy()
        ausentes = [c for c in requeridas if c not in d.columns]
        if ausentes:
            raise ErrorEntrada(f'Faltan columnas: {", ".join(ausentes)}. Descarga la plantilla para ver el formato.')
        d = d[requeridas]
        if d.empty:
            raise ErrorEntrada('El archivo no contiene semanas.')
        if d.isna().any().any():
            filas = (d.index[d.isna().any(axis=1)] + 2).tolist()[:5]
            raise ErrorEntrada(f'Hay celdas vacías (filas {filas}). No se imputan valores faltantes.')
        d['COD_DANE'] = d['COD_DANE'].astype(str).str.strip().str.replace(r'\.0+$', '', regex=True)
        desconocidos = sorted(set(d['COD_DANE']) - set(MUNICIPIOS))
        if desconocidos:
            raise ErrorEntrada(f'Municipios no incluidos en el modelo: {desconocidos}. Válidos: {", ".join(f"{k} ({v})" for k, v in MUNICIPIOS.items())}.')
        try:
            d['FECHA_INICIO'] = pd.to_datetime(d['FECHA_INICIO'], format='%Y-%m-%d')
        except (ValueError, TypeError) as error:
            raise ErrorEntrada('FECHA_INICIO debe tener formato AAAA-MM-DD (domingo que inicia la semana).') from error
        for columna in ['CASOS', *CLIMA]:
            d[columna] = pd.to_numeric(d[columna], errors='coerce')
        if d[['CASOS', *CLIMA]].isna().any().any() or not np.isfinite(d[['CASOS', *CLIMA]].to_numpy(float)).all():
            raise ErrorEntrada('CASOS y las variables de clima deben ser números.')
        if not np.equal(d['CASOS'], np.floor(d['CASOS'])).all():
            raise ErrorEntrada('CASOS debe ser un número entero de casos.')
        if d[['CASOS', 'PRECIPITACION_MM']].lt(0).any().any():
            raise ErrorEntrada('Casos y precipitación no pueden ser negativos.')
        if not (d['TEMP_MIN_C'].le(d['TEMP_MEDIA_C']) & d['TEMP_MEDIA_C'].le(d['TEMP_MAX_C'])).all():
            raise ErrorEntrada('En cada semana debe cumplirse TEMP_MIN_C ≤ TEMP_MEDIA_C ≤ TEMP_MAX_C.')
        if d.duplicated(['COD_DANE', 'FECHA_INICIO']).any():
            raise ErrorEntrada('Hay semanas repetidas para un mismo municipio.')
        try:
            epi = [semana_epidemiologica(f.date()) for f in d['FECHA_INICIO']]
        except ValueError as error:
            raise ErrorEntrada(str(error)) from error
        d['ANO'] = [a for a, _ in epi]
        d['SEMANA_EPI'] = [s for _, s in epi]
        d = d.sort_values(['COD_DANE', 'FECHA_INICIO']).groupby('COD_DANE', sort=True).tail(self.historia)
        for codigo, grupo in d.groupby('COD_DANE'):
            if len(grupo) < self.historia:
                raise ErrorEntrada(f'{codigo} ({MUNICIPIOS[codigo]}): se requieren {self.historia} semanas; hay {len(grupo)}.')
            if not grupo['FECHA_INICIO'].diff().dropna().dt.days.eq(7).all():
                raise ErrorEntrada(f'{codigo} ({MUNICIPIOS[codigo]}): las últimas {self.historia} semanas deben ser consecutivas.')
        return d.reset_index(drop=True)

    def pronosticar_entrada(self, datos):
        d = self.preparar_entrada(datos)
        matriz = self.transformar(d)
        codigos = sorted(d['COD_DANE'].unique())
        X = np.stack([matriz[d.index[d['COD_DANE'].eq(c)]] for c in codigos])
        casos = self.predecir(X, codigos)
        resultados = []
        for codigo, valor in zip(codigos, casos):
            ultima = d.loc[d['COD_DANE'].eq(codigo)].iloc[-1]
            objetivo = ultima['FECHA_INICIO'] + timedelta(weeks=self.horizonte)
            resultados.append({'codigo': codigo, 'valor': float(valor), 'ultima': ultima, 'objetivo': objetivo,
                               'historia': d.loc[d['COD_DANE'].eq(codigo)]})
        return resultados

    def ejemplo_entrada(self):
        """Reconstruir en unidades originales las entradas de ejemplo exportadas por el notebook."""
        referencia = pd.read_csv(self.carpeta / 'predicciones_ejemplo_validacion.csv', dtype={'COD_DANE': str}, encoding='utf-8-sig')
        with np.load(self.carpeta / 'entradas_ejemplo_validacion.npz') as ejemplo:
            X, codigos = ejemplo['historia'].astype(float), ejemplo['codigos'].astype(str)
        col = {c: i for i, c in enumerate(self.columnas)}
        filas = []
        for i, codigo in enumerate(codigos):
            objetivo = pd.Timestamp(referencia.loc[referencia['COD_DANE'].eq(codigo), 'FECHA_OBJETIVO'].iloc[0])
            ultima = objetivo - timedelta(weeks=self.horizonte)
            casos = np.expm1(self.prep['escaladores_casos'][codigo].inverse_transform(X[i][:, [col['CASOS_std']]])).ravel()
            temperaturas = self.prep['escaladores_clima']['temperaturas'].inverse_transform(
                X[i][:, [col[f'{t}_std'] for t in TEMPERATURAS]])
            lluvia = np.expm1(self.prep['escaladores_clima']['precipitacion'].inverse_transform(
                X[i][:, [col['PRECIPITACION_MM_std']]])).ravel()
            for k in range(self.historia):
                filas.append({
                    'COD_DANE': codigo, 'MUNICIPIO': MUNICIPIOS[codigo],
                    'FECHA_INICIO': (ultima - timedelta(weeks=self.historia - 1 - k)).strftime('%Y-%m-%d'),
                    'CASOS': int(round(casos[k])), 'TEMP_MEDIA_C': temperaturas[k, 0], 'TEMP_MAX_C': temperaturas[k, 1],
                    'TEMP_MIN_C': temperaturas[k, 2], 'PRECIPITACION_MM': max(float(lluvia[k]), 0.0),
                })
        return pd.DataFrame(filas), referencia
