"""Respaldar los Excel completos y reducir DataRaw al Valle del Cauca.

Ejecutar desde el proyecto con .venv/Scripts/python.exe preparar_datos_dengue.py.
No elimina filas por fecha faltante ni por municipio, ni cambia los conteos.
"""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import shutil
import sys
import pandas as pd
from datos_dengue import (
    COLUMNAS_CASOS, DEPARTAMENTO, codigo_departamento, leer_archivo_dengue,
)

RAIZ = Path(__file__).resolve().parent.parent / 'data'
DATOS = RAIZ / 'DataRaw'
ORIGINALES = RAIZ / 'DataRawOriginal'
ANOS = range(2007, 2026)


def sha256(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open('rb') as entrada:
        for bloque in iter(lambda: entrada.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def preparar():
    manifiesto = DATOS / 'manifiesto_reduccion.json'
    if manifiesto.is_file():
        previo = json.loads(manifiesto.read_text(encoding='utf-8'))
        for fila in previo['archivos']:
            assert sha256(ORIGINALES / fila['original']) == fila['sha256_original']
            assert sha256(DATOS / fila['reducido']) == fila['sha256_reducido']
            assert not (DATOS / fila['original']).exists()
        print('DataRaw ya está reducido y coincide con el respaldo verificado.', flush=True)
        return previo

    assert DATOS.resolve().parent == RAIZ and ORIGINALES.resolve().parent == RAIZ
    nombres = [f'Datos_{ano}_210.xlsx' for ano in ANOS]
    if not ORIGINALES.exists():
        for nombre in nombres:
            if not (DATOS / nombre).is_file():
                raise FileNotFoundError(DATOS / nombre)
        requerido = sum(p.stat().st_size for p in DATOS.rglob('*') if p.is_file())
        if shutil.disk_usage(RAIZ).free < requerido + 128 * 1024**2:
            raise OSError('No hay espacio suficiente para conservar la copia original.')
        print('Copiando DataRaw completo a DataRawOriginal...', flush=True)
        shutil.copytree(DATOS, ORIGINALES, copy_function=shutil.copy2)

    # Comprobar la copia completa antes de transformar o retirar ningún Excel.
    originales_hash = {}
    for nombre in nombres:
        original = ORIGINALES / nombre
        if not original.is_file():
            raise FileNotFoundError(f'El respaldo no está completo: {original}')
        originales_hash[nombre] = sha256(original)
        actual = DATOS / nombre
        if actual.exists() and sha256(actual) != originales_hash[nombre]:
            raise ValueError(f'El respaldo y DataRaw difieren: {nombre}. No se sustituye nada.')
    print('Respaldo verificado: los 19 Excel coinciden byte por byte.', flush=True)

    temporal = DATOS / '.reduccion_temporal'
    temporal.mkdir(exist_ok=True)
    registros = []
    for ano, nombre in zip(ANOS, nombres):
        origen = ORIGINALES / nombre
        print(f'[{ano}] Leyendo cinco columnas y seleccionando el departamento {DEPARTAMENTO}...', flush=True)
        datos = leer_archivo_dengue(origen)
        seleccion = codigo_departamento(datos['COD_DPTO_O']).eq(DEPARTAMENTO).fillna(False)
        valle = datos.loc[seleccion, COLUMNAS_CASOS].reset_index(drop=True)
        assert not valle.empty, f'{nombre}: no hay registros del departamento {DEPARTAMENTO}.'
        destino = temporal / f'Datos_{ano}_210.csv'
        valle.to_csv(destino, index=False, encoding='utf-8-sig', lineterminator='\n')
        recuperado = leer_archivo_dengue(destino)
        # Comparar todos los valores, incluidos códigos, fechas, nulos y duplicados.
        pd.testing.assert_frame_equal(recuperado, valle, check_exact=True)
        assert codigo_departamento(recuperado['COD_DPTO_O']).eq(DEPARTAMENTO).all()
        registros.append({
            'ano_archivo': ano, 'original': nombre, 'reducido': destino.name,
            'filas_originales': len(datos), 'filas_valle': len(valle),
            'columnas': COLUMNAS_CASOS,
            'bytes_original': origen.stat().st_size, 'bytes_reducido': destino.stat().st_size,
            'sha256_original': originales_hash[nombre], 'sha256_reducido': sha256(destino),
            'valores_releidos_identicos': True,
        })
        print(f'  {len(datos):,} registros → {len(valle):,} del Valle; CSV comprobado.', flush=True)
        del datos, valle, recuperado, seleccion
        gc.collect()

    # Solo sustituir la carpeta de trabajo cuando TODOS los CSV estén comprobados.
    for fila in registros:
        assert sha256(ORIGINALES / fila['original']) == fila['sha256_original']
        assert sha256(temporal / fila['reducido']) == fila['sha256_reducido']
    for fila in registros:
        os.replace(temporal / fila['reducido'], DATOS / fila['reducido'])
    for fila in registros:
        actual = DATOS / fila['original']
        if actual.exists():
            # Rutas fijas, dentro de DataRaw; la copia íntegra ya está verificada.
            assert actual.resolve().parent == DATOS.resolve()
            assert sha256(actual) == fila['sha256_original']
            assert sha256(DATOS / fila['reducido']) == fila['sha256_reducido']
            actual.unlink()
    temporal.rmdir()  # Solo la carpeta temporal vacía, sin eliminación recursiva.
    resultado = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'departamento': DEPARTAMENTO, 'columnas': COLUMNAS_CASOS,
        'formato': 'CSV UTF-8 con BOM',
        'limpieza_fechas_o_municipios_aplicada': False,
        'se_conservan_duplicados_y_orden': True,
        'bytes_originales': sum(f['bytes_original'] for f in registros),
        'bytes_reducidos': sum(f['bytes_reducido'] for f in registros),
        'archivos': registros,
    }
    manifiesto.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(registros).drop(columns=['columnas']).to_csv(
        DATOS / 'resumen_reduccion.csv', index=False, encoding='utf-8-sig',
    )
    print(f"Conversión terminada: {resultado['bytes_originales']/1024**2:.1f} → {resultado['bytes_reducidos']/1024**2:.1f} MB.", flush=True)
    return resultado


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preparar()
