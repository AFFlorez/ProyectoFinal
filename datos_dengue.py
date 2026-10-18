"""Lectura común de los archivos anuales de vigilancia del proyecto."""
from pathlib import Path
import pandas as pd

COLUMNAS_CASOS = ['COD_DPTO_O', 'COD_MUN_O', 'SEMANA', 'ANO', 'INI_SIN']
DEPARTAMENTO = '76'


def codigo_departamento(serie):
    """Normalizar solo para seleccionar o comprobar el departamento."""
    return (
        serie.astype('string').str.strip()
        .str.replace(r'^="([^"]*)"$', r'\1', regex=True)
        .str.replace(r'\.0+$', '', regex=True)
        .replace('', pd.NA).str.zfill(2)
    )


def ruta_archivo_dengue(ano, carpeta='DataRaw'):
    """Preferir el CSV reducido; admitir el Excel si aún no se ha convertido."""
    carpeta = Path(carpeta)
    for extension in ['csv', 'xlsx']:
        ruta = carpeta / f'Datos_{int(ano)}_210.{extension}'
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        f'Falta Datos_{int(ano)}_210.csv en {carpeta}. '
        'Para preparar los archivos, ejecuta preparar_datos_dengue.py.'
    )


def leer_archivo_dengue(ruta, columnas=None):
    """Leer códigos y fechas como texto, conservando orden, nulos y duplicados."""
    ruta = Path(ruta)
    columnas = list(COLUMNAS_CASOS if columnas is None else columnas)
    if ruta.suffix.lower() == '.csv':
        datos = pd.read_csv(
            ruta, usecols=columnas, dtype='string', encoding='utf-8-sig',
            keep_default_na=False, na_values=[''],
        )
    elif ruta.suffix.lower() == '.xlsx':
        datos = pd.read_excel(
            ruta, sheet_name='Datos', usecols=columnas,
            dtype='string', engine='calamine',
        )
    else:
        raise ValueError(f'Formato no admitido: {ruta.suffix}')
    return datos.loc[:, columnas]
