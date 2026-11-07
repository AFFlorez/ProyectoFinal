"""Parámetros de la aplicación. Se leen de variables de entorno (ver .env.example)."""
import os
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def _texto(nombre, defecto):
    valor = os.environ.get(nombre, '').strip()
    return valor or defecto


def _numero(nombre, defecto, tipo, minimo, maximo):
    crudo = _texto(nombre, str(defecto))
    try:
        valor = tipo(crudo)
    except ValueError as error:
        raise ValueError(f'{nombre}={crudo!r} no es un número válido.') from error
    if not minimo <= valor <= maximo:
        raise ValueError(f'{nombre} debe estar entre {minimo} y {maximo}; se recibió {valor}.')
    return valor


def _booleano(nombre, defecto):
    return _texto(nombre, 'true' if defecto else 'false').lower() in {'1', 'true', 'si', 'sí', 'yes', 'on'}


def _ruta(nombre, defecto):
    ruta = Path(_texto(nombre, str(defecto)))
    return ruta if ruta.is_absolute() else RAIZ / ruta


@dataclass(frozen=True)
class Configuracion:
    titulo: str
    carpeta_modelo: Path
    carpeta_datos: Path
    url_modelo: str
    verificar_integridad: bool
    percentil_moderado: float
    percentil_alto: float
    max_semanas_entrada: int
    nivel_log: str
    origenes_cors: tuple
    puerto_mlflow: str


def cargar_configuracion():
    config = Configuracion(
        titulo=_texto('APP_TITULO', 'Pronóstico semanal de dengue · Valle del Cauca'),
        carpeta_modelo=_ruta('MODELO_DIR', 'modelos'),
        carpeta_datos=_ruta('DATOS_DIR', 'data/DatosProcesados'),
        url_modelo=_texto('MODELO_URL', ''),
        verificar_integridad=_booleano('VERIFICAR_SHA256', True),
        percentil_moderado=_numero('ALERTA_PERCENTIL_MODERADO', 75, float, 1, 99),
        percentil_alto=_numero('ALERTA_PERCENTIL_ALTO', 90, float, 1, 99.9),
        max_semanas_entrada=_numero('MAX_SEMANAS_ENTRADA', 2000, int, 20, 100_000),
        nivel_log=_texto('LOG_LEVEL', 'INFO').upper(),
        origenes_cors=tuple(o.strip() for o in _texto('CORS_ORIGENES', '*').split(',') if o.strip()),
        puerto_mlflow=_texto('MLFLOW_PUERTO', ''),
    )
    if config.puerto_mlflow and not config.puerto_mlflow.isdigit():
        raise ValueError('MLFLOW_PUERTO debe ser un número de puerto o quedar vacío.')
    if config.percentil_moderado >= config.percentil_alto:
        raise ValueError('ALERTA_PERCENTIL_MODERADO debe ser menor que ALERTA_PERCENTIL_ALTO.')
    return config
