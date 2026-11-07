"""Pruebas de la API sin la serie histórica: el modo de datos propios debe funcionar siempre."""
import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def cliente(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv('DATOS_DIR', str(tmp_path_factory.mktemp('sin_datos')))
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


def test_salud_y_modelo(cliente):
    salud = cliente.get('/api/salud').json()
    assert salud['estado'] == 'ok' and salud['historico_disponible'] is False
    modelo = cliente.get('/api/modelo').json()
    assert modelo['historia_semanas'] == 20 and modelo['horizonte_semanas'] == 4
    assert cliente.get('/').status_code == 200


def test_historico_no_disponible_responde_503(cliente):
    respuesta = cliente.get('/api/pronostico', params={'codigo': '76001'})
    assert respuesta.status_code == 503


def test_plantilla_y_archivo(cliente):
    plantilla = cliente.get('/api/plantilla.csv')
    assert plantilla.status_code == 200
    respuesta = cliente.post('/api/pronostico/archivo', files={'archivo': ('p.csv', plantilla.content, 'text/csv')})
    assert respuesta.status_code == 200, respuesta.text
    resultados = respuesta.json()['resultados']
    assert {r['municipio']['codigo'] for r in resultados} == {'76001', '76111', '76520', '76834', '76892'}
    cali = next(r for r in resultados if r['municipio']['codigo'] == '76001')
    assert cali['objetivo']['fecha_inicio'] == '2022-01-23'
    assert cali['casos_predichos'] == pytest.approx(74.24, abs=0.5)
    assert cali['nivel_alerta'] in {'bajo', 'moderado', 'alto'}


def test_csv_excel_espanol(cliente):
    texto = cliente.get('/api/plantilla.csv').content.decode('utf-8-sig')
    excel = texto.replace(',', ';').replace('.', ',')
    respuesta = cliente.post('/api/pronostico/archivo', files={'archivo': ('p.csv', io.BytesIO(excel.encode()), 'text/csv')})
    assert respuesta.status_code == 200, respuesta.text


def test_json_con_error_claro(cliente):
    semanas = [{'cod_dane': '76001', 'fecha_inicio': '2021-12-26', 'casos': 3, 'temp_media_c': 24,
                'temp_max_c': 30, 'temp_min_c': 19, 'precipitacion_mm': 10}]
    respuesta = cliente.post('/api/pronostico/semanas', json={'semanas': semanas})
    assert respuesta.status_code == 422
    assert 'se requieren 20 semanas' in respuesta.json()['detail']
