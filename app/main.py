"""API e interfaz web del pronóstico semanal de dengue (FastAPI).

Documentación interactiva de la API en /docs. La interfaz para el usuario final
se sirve en / desde app/static.
"""
import io
import logging
import math
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.calendario import semana_epidemiologica
from app.config import cargar_configuracion
from app.historico import FUERA, PERIODOS, Historico
from app.pronosticador import CLIMA, MUNICIPIOS, ErrorEntrada, Pronosticador

VERSION = '1.0.0'
EQUIPO = [
    ('Andrés Felipe Flórez', 'af.florez@uniandes.edu.co'),
    ('Oscar Morillo', 'o.morillo@uniandes.edu.co'),
    ('Jared Foster', 'j.foster@uniandes.edu.co'),
    ('Jeferson David Vargas', 'jd.vargast1@uniandes.edu.co'),
]
ORDEN_CONJUNTOS = [nombre for _, nombre in PERIODOS] + [FUERA]
ESTATICOS = Path(__file__).resolve().parent / 'static'
config = cargar_configuracion()
logging.basicConfig(level=config.nivel_log, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('app')
estado = {}


@asynccontextmanager
async def ciclo_de_vida(_app):
    pronosticador = Pronosticador(config.carpeta_modelo, config.verificar_integridad, config.url_modelo)
    estado['pronosticador'] = pronosticador
    estado['historico'] = Historico(config.carpeta_datos, pronosticador)
    estado['inicio'] = time.time()
    yield
    estado.clear()


app = FastAPI(
    title='API · Pronóstico semanal de dengue',
    version=VERSION,
    description=(
        'Pronostica los casos de dengue de la **semana t+4** en cinco municipios del Valle del Cauca '
        '(Cali, Buga, Palmira, Tuluá y Yumbo) con una red LSTM global (64 unidades + Dense 16) que usa '
        'las últimas **20 semanas** de casos y clima. La interfaz web está en `/`.\n\n'
        'Proyecto final del curso *Proyecto: Despliegue de Soluciones*, Maestría en Inteligencia Artificial, '
        'Universidad de los Andes.\n\n**Equipo**\n\n'
        + '\n'.join(f'- {nombre} ({correo})' for nombre, correo in EQUIPO)
    ),
    lifespan=ciclo_de_vida,
)
app.add_middleware(CORSMiddleware, allow_origins=list(config.origenes_cors), allow_methods=['GET', 'POST'], allow_headers=['*'])


@app.middleware('http')
async def medir_tiempo(request: Request, siguiente):
    inicio = time.perf_counter()
    respuesta = await siguiente(request)
    ms = (time.perf_counter() - inicio) * 1000
    respuesta.headers['X-Tiempo-Respuesta-ms'] = f'{ms:.1f}'
    if not request.url.path.startswith('/api'):
        # La interfaz se revalida en cada visita: tras actualizar el despliegue nadie ve una versión vieja.
        respuesta.headers['Cache-Control'] = 'no-cache'
    if request.url.path.startswith('/api'):
        log.info('%s %s -> %s (%.1f ms)', request.method, request.url.path, respuesta.status_code, ms)
    return respuesta


@app.exception_handler(ErrorEntrada)
async def error_entrada(_request, error):
    return JSONResponse(status_code=422, content={'detail': str(error)})


# ---- Utilidades -----------------------------------------------------------
def _pron():
    return estado['pronosticador']


def _historico():
    historico = estado['historico']
    if not historico.disponible:
        raise HTTPException(503, historico.mensaje)
    return historico


def _municipio(codigo):
    codigo = str(codigo).strip()
    if codigo not in MUNICIPIOS:
        raise HTTPException(404, f'Municipio {codigo} no incluido. Válidos: {", ".join(MUNICIPIOS)}.')
    return codigo


def _numero(valor, decimales=2):
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    return round(float(valor), decimales)


def _semana(fecha_inicio):
    fecha_inicio = pd.Timestamp(fecha_inicio).date()
    ano, semana = semana_epidemiologica(fecha_inicio)
    return {'ano': ano, 'semana': semana, 'fecha_inicio': fecha_inicio.isoformat(),
            'fecha_fin': (fecha_inicio + timedelta(days=6)).isoformat()}


def _filas_semanas(tabla):
    return [{
        **_semana(f['FECHA_INICIO']), 'casos': int(f['CASOS']),
        **{c.lower(): _numero(f[c]) for c in CLIMA},
    } for _, f in tabla.iterrows()]


def _resultado(codigo, valor, fecha_ultima, real=None):
    pron = _pron()
    umbrales = pron.umbrales(codigo, config.percentil_moderado, config.percentil_alto)
    objetivo = pd.Timestamp(fecha_ultima) + timedelta(weeks=pron.horizonte)
    real = None if real is None or (isinstance(real, float) and math.isnan(real)) else int(real)
    return {
        'municipio': {'codigo': codigo, 'nombre': MUNICIPIOS[codigo]},
        'emision': _semana(fecha_ultima),
        'objetivo': _semana(objetivo),
        'horizonte_semanas': pron.horizonte,
        'casos_predichos': _numero(valor),
        'casos_predichos_redondeados': max(0, int(round(valor))),
        'nivel_alerta': pron.nivel_alerta(valor, umbrales),
        'umbrales_alerta': {k: _numero(v, 1) for k, v in umbrales.items()},
        'casos_reales': real,
        'error_absoluto': None if real is None else _numero(abs(valor - real)),
    }


def _metricas(tabla):
    error = tabla['CASOS_PREDICHOS'] - tabla['CASOS_REALES']
    return {'n': int(len(tabla)), 'mae': _numero(error.abs().mean()),
            'rmse': _numero(np.sqrt((error ** 2).mean())), 'sesgo_medio': _numero(error.mean())}


# ---- Interfaz web ---------------------------------------------------------
app.mount('/static', StaticFiles(directory=ESTATICOS), name='static')


@app.get('/', include_in_schema=False)
def inicio():
    return FileResponse(ESTATICOS / 'index.html')


# ---- API --------------------------------------------------------------------
@app.get('/api/salud', tags=['Servicio'], summary='Estado del servicio')
def salud():
    pron, historico = _pron(), estado['historico']
    return {
        'estado': 'ok', 'version': VERSION, 'modelo': pron.config['nombre'],
        'segundos_carga_modelo': _numero(pron.segundos_carga),
        'historico_disponible': historico.disponible,
        'pronosticos_precalculados': int(len(historico.predicciones)) if historico.disponible else 0,
        'segundos_activo': int(time.time() - estado['inicio']),
    }


@app.get('/api/configuracion', tags=['Servicio'], summary='Parámetros activos de la aplicación')
def configuracion():
    return {'titulo': config.titulo, 'percentil_alerta_moderada': config.percentil_moderado,
            'percentil_alerta_alta': config.percentil_alto, 'max_semanas_entrada': config.max_semanas_entrada,
            'verificar_sha256': config.verificar_integridad, 'puerto_mlflow': config.puerto_mlflow or None,
            'historico_disponible': estado['historico'].disponible, 'aviso_historico': estado['historico'].mensaje}


@app.get('/api/modelo', tags=['Modelo'], summary='Descripción y métricas del modelo seleccionado')
def modelo():
    pron = _pron()
    c = pron.config
    validacion = c['seleccion']['MODELO_ELEGIDO']
    return {
        'id': c['ID'], 'nombre': c['nombre'], 'arquitectura': c['arquitectura'], 'parametros': pron.parametros,
        'historia_semanas': pron.historia, 'horizonte_semanas': pron.horizonte, 'objetivo': c['objetivo'],
        'variables': c['columnas_temporales'], 'municipios': MUNICIPIOS,
        'criterio_seleccion': c['seleccion']['criterio_principal'],
        'metricas': {
            'validacion_2022_2023': {'n': validacion['N'], 'mae': _numero(validacion['MAE']), 'rmse': _numero(validacion['RMSE'])},
            'prueba_2024_2025': {'n': c['metricas_test']['N'], 'mae': _numero(c['metricas_test']['MAE']),
                                 'rmse': _numero(c['metricas_test']['RMSE'])},
        },
        'versiones': c['versiones'],
    }


@app.get('/api/municipios', tags=['Pronóstico histórico'], summary='Municipios, umbrales de alerta y semanas disponibles')
def municipios():
    pron, historico = _pron(), estado['historico']
    salida = []
    for codigo, nombre in MUNICIPIOS.items():
        item = {'codigo': codigo, 'nombre': nombre,
                'umbrales_alerta': {k: _numero(v, 1) for k, v in pron.umbrales(codigo, config.percentil_moderado, config.percentil_alto).items()}}
        if historico.disponible:
            primera, ultima = historico.rango(codigo)
            item['primera_emision'] = {'ano': int(primera['ANO']), 'semana': int(primera['SEMANA_EPI'])}
            item['ultima_emision'] = {'ano': int(ultima['ANO']), 'semana': int(ultima['SEMANA_EPI'])}
        salida.append(item)
    return {'historico_disponible': historico.disponible, 'aviso': historico.mensaje, 'municipios': salida}


@app.get('/api/pronostico', tags=['Pronóstico histórico'],
         summary='Pronóstico t+4 de un municipio emitido en una semana del histórico')
def pronostico(codigo: Annotated[str, Query(description='Código DANE, p. ej. 76001 (Cali)')],
               ano: Annotated[int | None, Query(description='Año epidemiológico de la última semana observada')] = None,
               semana: Annotated[int | None, Query(ge=1, le=53, description='Semana epidemiológica de la última semana observada')] = None):
    historico, pron = _historico(), _pron()
    codigo = _municipio(codigo)
    try:
        fila = historico.emision(codigo, ano, semana)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    previa, siguiente = historico.ventana(fila, pron.historia, pron.horizonte)
    return {
        **_resultado(codigo, fila['CASOS_PREDICHOS'], fila['FECHA_INICIO'], fila['CASOS_REALES']),
        'conjunto': fila['CONJUNTO'],
        'historia': _filas_semanas(previa),
        'semanas_posteriores': _filas_semanas(siguiente),
    }


@app.get('/api/resumen', tags=['Pronóstico histórico'], summary='Pronóstico t+4 de los cinco municipios para una semana de emisión')
def resumen(ano: int | None = None, semana: Annotated[int | None, Query(ge=1, le=53)] = None):
    historico = _historico()
    salida = []
    for codigo in MUNICIPIOS:
        try:
            fila = historico.emision(codigo, ano, semana)
        except LookupError as error:
            raise HTTPException(404, str(error)) from error
        salida.append({**_resultado(codigo, fila['CASOS_PREDICHOS'], fila['FECHA_INICIO'], fila['CASOS_REALES']),
                       'conjunto': fila['CONJUNTO']})
    return {'resultados': salida}


@app.get('/api/desempeno', tags=['Pronóstico histórico'], summary='Pronósticos frente a casos reales en un rango de años')
def desempeno(codigo: str, desde: Annotated[int, Query(ge=2007)] = 2022, hasta: Annotated[int, Query(le=2100)] = 2025):
    historico = _historico()
    codigo = _municipio(codigo)
    t = historico.predicciones
    t = t.loc[t['COD_DANE'].eq(codigo) & t['ANO_OBJETIVO'].between(desde, hasta) & t['CASOS_REALES'].notna()]
    if t.empty:
        raise HTTPException(404, f'No hay semanas con casos reales entre {desde} y {hasta}.')
    return {
        'municipio': {'codigo': codigo, 'nombre': MUNICIPIOS[codigo]},
        'metricas': {'Total': _metricas(t), **{c: _metricas(t.loc[t['CONJUNTO'].eq(c)]) for c in ORDEN_CONJUNTOS
                                               if t['CONJUNTO'].eq(c).any()}},
        'serie': [{'fecha_objetivo': f['FECHA_OBJETIVO'].date().isoformat(), 'ano': int(f['ANO_OBJETIVO']),
                   'semana': int(f['SEMANA_OBJETIVO']), 'predicho': _numero(f['CASOS_PREDICHOS']),
                   'real': int(f['CASOS_REALES']), 'conjunto': f['CONJUNTO']} for _, f in t.iterrows()],
    }


class Semana(BaseModel):
    cod_dane: str = Field(examples=['76001'])
    fecha_inicio: date = Field(description='Domingo que inicia la semana epidemiológica', examples=['2021-12-26'])
    casos: int = Field(ge=0)
    temp_media_c: float
    temp_max_c: float
    temp_min_c: float
    precipitacion_mm: float = Field(ge=0)


class SolicitudPronostico(BaseModel):
    semanas: list[Semana] = Field(description='Al menos 20 semanas consecutivas por municipio')


def _pronosticar_tabla(tabla):
    if len(tabla) > config.max_semanas_entrada:
        raise ErrorEntrada(f'Se admiten como máximo {config.max_semanas_entrada} filas (MAX_SEMANAS_ENTRADA).')
    resultados = []
    for r in _pron().pronosticar_entrada(tabla):
        resultados.append({**_resultado(r['codigo'], r['valor'], r['ultima']['FECHA_INICIO']),
                           'historia': _filas_semanas(r['historia'])})
    return {'resultados': resultados}


@app.post('/api/pronostico/semanas', tags=['Pronóstico con datos propios'], summary='Pronosticar a partir de semanas enviadas en JSON')
def pronostico_json(solicitud: SolicitudPronostico):
    tabla = pd.DataFrame([s.model_dump() for s in solicitud.semanas])
    if not tabla.empty:
        tabla['fecha_inicio'] = tabla['fecha_inicio'].astype(str)
    return _pronosticar_tabla(tabla)


@app.post('/api/pronostico/archivo', tags=['Pronóstico con datos propios'], summary='Pronosticar a partir de un CSV (ver /api/plantilla.csv)')
async def pronostico_archivo(archivo: UploadFile = File(description='CSV con las columnas de la plantilla')):
    contenido = await archivo.read(5_000_001)
    if len(contenido) > 5_000_000:
        raise ErrorEntrada('El archivo supera 5 MB.')
    try:
        texto = contenido.decode('utf-8-sig')
    except UnicodeDecodeError:
        texto = contenido.decode('latin-1')
    encabezado = texto.splitlines()[0] if texto.strip() else ''
    # Excel en español guarda con punto y coma y coma decimal.
    separador, decimal = (';', ',') if encabezado.count(';') > encabezado.count(',') else (',', '.')
    try:
        tabla = pd.read_csv(io.StringIO(texto), sep=separador, decimal=decimal, dtype={'COD_DANE': str, 'cod_dane': str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ErrorEntrada('No se pudo leer el CSV. Usa la plantilla como guía.') from error
    return _pronosticar_tabla(tabla)


@app.get('/api/plantilla.csv', tags=['Pronóstico con datos propios'], summary='Plantilla de entrada con 20 semanas de ejemplo')
def plantilla():
    historico, pron = estado['historico'], _pron()
    if historico.disponible:
        d = historico.semanas.groupby('COD_DANE', sort=True).tail(pron.historia).copy()
        d['FECHA_INICIO'] = d['FECHA_INICIO'].dt.strftime('%Y-%m-%d')
        d.insert(1, 'MUNICIPIO', d['COD_DANE'].map(MUNICIPIOS))
        d = d[['COD_DANE', 'MUNICIPIO', 'FECHA_INICIO', 'CASOS', *CLIMA]]
    else:
        d, _ = pron.ejemplo_entrada()
    csv = d.round(2).to_csv(index=False)
    return Response(csv.encode('utf-8-sig'), media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="plantilla_pronostico_dengue.csv"'})
