"""Serie semanal 2007-2025 de los cinco municipios (data/DatosProcesados/semanal_*.csv, punto 11 del notebook).

Al iniciar se calculan, en un solo lote, los pronósticos de todas las semanas de
emisión posibles. Las consultas posteriores solo filtran esa tabla.
"""
import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from app.calendario import inicio_ano_epidemiologico, semana_epidemiologica
from app.pronosticador import CLIMA, MUNICIPIOS

log = logging.getLogger(__name__)

COLUMNAS = ['COD_DANE', 'ANO', 'SEMANA_EPI', 'FECHA_INICIO', 'FECHA_FIN', 'CASOS', *CLIMA]


# Primer año de cada periodo, en el orden del notebook.
PERIODOS = [(2007, 'Entrenamiento'), (2022, 'Validación'), (2024, 'Prueba'), (2026, 'Datos nuevos')]
# El notebook evalúa todas las historias sobre los mismos objetivos: los que admiten la más larga (24 semanas).
HISTORIA_SOPORTE_COMUN = 24
FUERA = 'Fuera de la evaluación'


def conjunto_de(ano_objetivo, fin_emision, con_soporte_comun):
    """Periodo de un pronóstico, con la misma regla del notebook (construir_secuencias).

    Quedan fuera de las métricas por periodo, como en el notebook, los objetivos sin el
    soporte común de 24 semanas y los que tienen la emisión antes del cierre del periodo
    anterior. Así los conteos coinciden con el informe: 3.780, 505 y 510.
    """
    inicio, nombre = [(a, n) for a, n in PERIODOS if a <= ano_objetivo][-1]
    cierre_anterior = pd.Timestamp(inicio_ano_epidemiologico(inicio) - timedelta(days=1))
    if not con_soporte_comun or (inicio > PERIODOS[0][0] and fin_emision < cierre_anterior):
        return FUERA
    return nombre


class Historico:
    def __init__(self, carpeta, pronosticador):
        self.disponible = False
        self.mensaje = ''
        archivos = sorted(carpeta.glob('semanal_*.csv')) if carpeta.is_dir() else []
        if not archivos:
            self.mensaje = (f'No se encontraron archivos semanal_*.csv en {carpeta.name}/. '
                            'El pronóstico con datos propios (CSV) sigue disponible.')
            log.warning(self.mensaje)
            return
        inicio = time.perf_counter()
        self.semanas = self._leer(archivos)
        self.predicciones = self._pronosticar_todo(pronosticador)
        self.disponible = True
        self.archivos = [a.name for a in archivos]
        log.info('Histórico cargado: %d semanas, %d pronósticos en %.2f s',
                 len(self.semanas), len(self.predicciones), time.perf_counter() - inicio)

    @staticmethod
    def _leer(archivos):
        d = pd.concat([pd.read_csv(a, dtype={'COD_DANE': str}, encoding='utf-8-sig') for a in archivos], ignore_index=True)
        ausentes = sorted(set(COLUMNAS) - set(d.columns))
        if ausentes:
            raise ValueError(f'Faltan columnas en la serie semanal: {ausentes}')
        d = d[COLUMNAS].copy()
        d['COD_DANE'] = d['COD_DANE'].str.strip()
        d = d.loc[d['COD_DANE'].isin(MUNICIPIOS)]
        for columna in ['FECHA_INICIO', 'FECHA_FIN']:
            d[columna] = pd.to_datetime(d[columna], format='%Y-%m-%d')
        d = d.sort_values(['COD_DANE', 'FECHA_INICIO']).reset_index(drop=True)
        if d.duplicated(['COD_DANE', 'FECHA_INICIO']).any():
            raise ValueError('La serie semanal tiene semanas repetidas.')
        fechas = d[['FECHA_INICIO', 'ANO', 'SEMANA_EPI']].drop_duplicates()
        calculado = [semana_epidemiologica(f.date()) for f in fechas['FECHA_INICIO']]
        if calculado != list(zip(fechas['ANO'].astype(int), fechas['SEMANA_EPI'].astype(int))):
            raise ValueError('El calendario de la serie semanal no coincide con el calendario epidemiológico.')
        for codigo, g in d.groupby('COD_DANE'):
            if not g['FECHA_INICIO'].diff().dropna().dt.days.eq(7).all():
                raise ValueError(f'{codigo}: la serie semanal tiene huecos.')
        return d

    def _pronosticar_todo(self, pronosticador):
        d = self.semanas
        h, horizonte = pronosticador.historia, pronosticador.horizonte
        matriz = pronosticador.transformar(d)
        completas = d[['CASOS', *CLIMA]].notna().all(axis=1).to_numpy()
        reales = d.set_index(['COD_DANE', 'FECHA_INICIO'])['CASOS']
        X, filas = [], []
        for codigo, posiciones in d.groupby('COD_DANE', sort=True).indices.items():
            for fin in range(h - 1, len(posiciones)):
                ventana = posiciones[fin - h + 1:fin + 1]
                if not completas[ventana].all():
                    continue
                X.append(matriz[ventana])
                ultima = d.iloc[posiciones[fin]]
                objetivo = ultima['FECHA_INICIO'] + timedelta(weeks=horizonte)
                ano_obj, semana_obj = semana_epidemiologica(objetivo.date())
                filas.append({
                    'COD_DANE': codigo, 'POSICION': int(posiciones[fin]),
                    'ANO': int(ultima['ANO']), 'SEMANA_EPI': int(ultima['SEMANA_EPI']),
                    'FECHA_INICIO': ultima['FECHA_INICIO'], 'FECHA_FIN': ultima['FECHA_FIN'],
                    'FECHA_OBJETIVO': objetivo, 'ANO_OBJETIVO': ano_obj, 'SEMANA_OBJETIVO': semana_obj,
                    'CASOS_REALES': reales.get((codigo, objetivo), np.nan),
                    'SOPORTE_COMUN': fin >= HISTORIA_SOPORTE_COMUN - 1,
                })
        tabla = pd.DataFrame(filas)
        tabla['CASOS_PREDICHOS'] = pronosticador.predecir(np.stack(X), tabla['COD_DANE'].to_numpy())
        tabla['CONJUNTO'] = [conjunto_de(a, f, s) for a, f, s in
                             zip(tabla['ANO_OBJETIVO'], tabla['FECHA_FIN'], tabla['SOPORTE_COMUN'])]
        return tabla

    # ---- Consultas ---------------------------------------------------------
    def emision(self, codigo, ano=None, semana=None):
        filas = self.predicciones.loc[self.predicciones['COD_DANE'].eq(codigo)]
        if ano is None or semana is None:
            return filas.iloc[-1]
        elegida = filas.loc[filas['ANO'].eq(ano) & filas['SEMANA_EPI'].eq(semana)]
        if elegida.empty:
            primera = filas.iloc[0]
            raise LookupError(f'No hay pronóstico para {ano}-S{semana}. La primera semana de emisión disponible es '
                              f'{primera["ANO"]}-S{primera["SEMANA_EPI"]} (se necesitan 20 semanas previas).')
        return elegida.iloc[0]

    def ventana(self, fila, historia, posteriores):
        d = self.semanas
        pos = int(fila['POSICION'])
        mismo = d['COD_DANE'].eq(fila['COD_DANE'])
        previa = d.iloc[pos - historia + 1:pos + 1]
        siguiente = d.iloc[pos + 1:pos + 1 + posteriores]
        return previa, siguiente.loc[mismo.iloc[pos + 1:pos + 1 + posteriores]]

    def rango(self, codigo):
        filas = self.predicciones.loc[self.predicciones['COD_DANE'].eq(codigo)]
        return filas.iloc[0], filas.iloc[-1]
