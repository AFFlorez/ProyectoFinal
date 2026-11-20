"""Pruebas de la configuración: rutas de datos por defecto y compatibilidad con la carpeta anterior."""
from app.config import RAIZ, cargar_configuracion


def test_datos_por_defecto(monkeypatch):
    monkeypatch.delenv('DATOS_DIR', raising=False)
    assert cargar_configuracion().carpeta_datos == RAIZ / 'datos' / 'DatosProcesados'


def test_ruta_anterior_data_se_redirige(monkeypatch):
    # Instancias creadas antes del cambio de nombre tienen DATOS_DIR=data/DatosProcesados en app.env.
    monkeypatch.setenv('DATOS_DIR', 'data/DatosProcesados')
    assert cargar_configuracion().carpeta_datos == RAIZ / 'datos' / 'DatosProcesados'


def test_ruta_explicita_se_respeta(monkeypatch, tmp_path):
    monkeypatch.setenv('DATOS_DIR', str(tmp_path))
    assert cargar_configuracion().carpeta_datos == tmp_path
