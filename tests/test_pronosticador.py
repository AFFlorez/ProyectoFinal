"""Comprueba que la app reproduce las predicciones exportadas por el notebook (punto 20.4)."""
from datetime import date

import numpy as np
import pytest

from app.calendario import semana_epidemiologica, semanas_del_ano
from app.config import RAIZ
from app.pronosticador import ErrorEntrada, Pronosticador


@pytest.fixture(scope='module')
def pronosticador():
    return Pronosticador(RAIZ / 'modelos')


def test_calendario_epidemiologico():
    assert semana_epidemiologica(date(2022, 1, 2)) == (2022, 1)
    assert semana_epidemiologica(date(2021, 12, 26)) == (2021, 52)
    assert semana_epidemiologica(date(2020, 12, 27)) == (2020, 53)
    assert [semanas_del_ano(a) for a in (2014, 2015, 2020, 2021)] == [53, 52, 53, 52]


def test_reproduce_tensor_exportado(pronosticador):
    ejemplo, _ = pronosticador.ejemplo_entrada()
    preparado = pronosticador.preparar_entrada(ejemplo.drop(columns='MUNICIPIO'))
    X = pronosticador.transformar(preparado).reshape(5, pronosticador.historia, -1)
    with np.load(RAIZ / 'modelos' / 'entradas_ejemplo_validacion.npz') as referencia:
        np.testing.assert_allclose(X, referencia['historia'], atol=1e-4)


def test_reproduce_predicciones_exportadas(pronosticador):
    ejemplo, referencia = pronosticador.ejemplo_entrada()
    resultados = pronosticador.pronosticar_entrada(ejemplo)
    obtenido = {r['codigo']: r['valor'] for r in resultados}
    for _, fila in referencia.iterrows():
        assert obtenido[fila['COD_DANE']] == pytest.approx(fila['CASOS_PREDICHOS'], rel=1e-3, abs=1e-3)
    assert {r['objetivo'].strftime('%Y-%m-%d') for r in resultados} == set(referencia['FECHA_OBJETIVO'])


def test_rechaza_semanas_incompletas(pronosticador):
    ejemplo, _ = pronosticador.ejemplo_entrada()
    with pytest.raises(ErrorEntrada, match='semanas'):
        pronosticador.preparar_entrada(ejemplo.iloc[3:])
    with pytest.raises(ErrorEntrada, match='Faltan columnas'):
        pronosticador.preparar_entrada(ejemplo.drop(columns='CASOS'))


def test_umbrales_crecientes(pronosticador):
    for codigo in pronosticador.orden:
        u = pronosticador.umbrales(codigo, 75, 90)
        assert 0 <= u['moderado'] < u['alto']
