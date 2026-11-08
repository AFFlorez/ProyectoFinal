"""La serie semanal versionada debe reproducir las métricas oficiales del modelo.

Referencias: modelos/configuracion.json (validación y prueba) y
modelos/experimentos/metricas_tres_conjuntos.csv (entrenamiento).
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.config import RAIZ
from app.historico import Historico
from app.pronosticador import Pronosticador

DATOS = RAIZ / 'data' / 'DatosProcesados'
pytestmark = pytest.mark.skipif(not list(DATOS.glob('semanal_*.csv')), reason='Sin serie semanal en data/DatosProcesados')


@pytest.fixture(scope='module')
def historico():
    return Historico(DATOS, Pronosticador(RAIZ / 'modelos'))


@pytest.fixture(scope='module')
def referencias():
    config = json.loads((RAIZ / 'modelos' / 'configuracion.json').read_text(encoding='utf-8-sig'))
    tres = pd.read_csv(RAIZ / 'modelos' / 'experimentos' / 'metricas_tres_conjuntos.csv').set_index('MODELO').loc[config['nombre']]
    return {
        'Entrenamiento': {'N': 3780, 'MAE': tres['MAE_TRAINING'], 'RMSE': tres['RMSE_TRAINING']},
        'Validación': config['seleccion']['MAE'],
        'Prueba': config['metricas_test'],
    }


def test_serie_completa(historico):
    d = historico.semanas
    assert d['COD_DANE'].nunique() == 5
    assert d.groupby('COD_DANE').size().nunique() == 1
    assert not d[['CASOS', 'TEMP_MEDIA_C', 'TEMP_MAX_C', 'TEMP_MIN_C', 'PRECIPITACION_MM']].isna().any().any()


@pytest.mark.parametrize('conjunto', ['Entrenamiento', 'Validación', 'Prueba'])
def test_reproduce_metricas_oficiales(historico, referencias, conjunto):
    t = historico.predicciones.loc[historico.predicciones['CONJUNTO'].eq(conjunto)]
    error = t['CASOS_PREDICHOS'] - t['CASOS_REALES']
    referencia = referencias[conjunto]
    assert len(t) == referencia['N']
    assert error.abs().mean() == pytest.approx(referencia['MAE'], abs=1e-3)
    assert np.sqrt((error ** 2).mean()) == pytest.approx(referencia['RMSE'], abs=1e-3)
