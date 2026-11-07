"""Auditoría reproducible de los CSV de dengue del proyecto.

Este programa SOLO LEE los CSV. No limpia, elimina ni modifica registros.
Para ejecutarlo desde la carpeta ProyectoFinal:
    python validar_datos_dengue.py

Lee data/DataRaw (o DataRaw en la raíz, en la estructura anterior).
Los reportes se guardan en Auditoria_Datos/ (fuera de los datos).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

COLUMNAS = ['COD_DPTO_O', 'COD_MUN_O', 'SEMANA', 'ANO', 'INI_SIN']
ANOS = range(2007, 2026)
RAIZ = Path(__file__).resolve().parent.parent
# Tras la reorganización del repositorio los CSV están en data/DataRaw;
# se admite también la ubicación anterior (DataRaw en la raíz).
DATA_DIR_POR_DEFECTO = next(
    (ruta for ruta in (RAIZ / 'data' / 'DataRaw', RAIZ / 'DataRaw') if ruta.is_dir()),
    RAIZ / 'data' / 'DataRaw',
)


def huella_sha256(ruta: Path) -> str:
    """Calcula una 'huella digital' para saber si un archivo cambió."""
    calculo = hashlib.sha256()
    with ruta.open('rb') as archivo:
        for parte in iter(lambda: archivo.read(1024 * 1024), b''):
            calculo.update(parte)
    return calculo.hexdigest()


def normalizar_codigo(serie: pd.Series, ancho: int) -> pd.Series:
    """Conserva códigos como texto para no perder ceros a la izquierda."""
    return (serie.astype('string').str.strip()
            .str.replace(r'^="([^"]*)"$', r'\1', regex=True)
            .str.replace(r'\.0+$', '', regex=True)
            .str.zfill(ancho))


def auditar(data_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifiesto = data_dir / 'manifiesto_reduccion.json'
    huellas_esperadas = {}
    conteos_esperados = {}
    if manifiesto.exists():
        previo = json.loads(manifiesto.read_text(encoding='utf-8'))
        for fila in previo.get('archivos', []):
            huellas_esperadas[fila['reducido']] = fila.get('sha256_reducido')
            conteos_esperados[fila['reducido']] = fila.get('filas_valle')

    resumen = []
    incidencias = []
    for ano in ANOS:
        ruta = data_dir / f'Datos_{ano}_210.csv'
        if not ruta.exists():
            incidencias.append({'archivo': ruta.name, 'nivel': 'ERROR', 'detalle': 'Falta el archivo anual'})
            resumen.append({'ano': ano, 'archivo': ruta.name, 'existe': False})
            continue

        # dtype='string' evita que 001 pase a 1; mantener NA permite encontrar fechas vacías.
        try:
            datos = pd.read_csv(ruta, encoding='utf-8-sig', dtype='string', keep_default_na=False, na_values=[''])
        except Exception as error:
            incidencias.append({'archivo': ruta.name, 'nivel': 'ERROR', 'detalle': f'No se pudo leer: {error}'})
            resumen.append({'ano': ano, 'archivo': ruta.name, 'existe': True, 'legible': False})
            continue

        faltan = [col for col in COLUMNAS if col not in datos.columns]
        if faltan:
            incidencias.append({'archivo': ruta.name, 'nivel': 'ERROR', 'detalle': 'Columnas ausentes: ' + ', '.join(faltan)})
            resumen.append({'ano': ano, 'archivo': ruta.name, 'existe': True, 'legible': True, 'columnas_completas': False})
            continue

        # Las verificaciones siguientes son controles de calidad, NO limpieza.
        departamento = normalizar_codigo(datos['COD_DPTO_O'], 2)
        codigo_municipio = normalizar_codigo(datos['COD_MUN_O'], 3)
        semanas = pd.to_numeric(datos['SEMANA'], errors='coerce')
        anos_columna = pd.to_numeric(datos['ANO'], errors='coerce')
        sin_fecha = int(datos['INI_SIN'].fillna('').str.strip().eq('').sum())
        sin_municipio = int(datos['COD_MUN_O'].fillna('').str.strip().eq('').sum())
        fuera_departamento = int((~departamento.eq('76')).fillna(True).sum())
        semanas_invalidas = int((~semanas.between(1, 53)).fillna(True).sum())
        ano_incorrecto = int((~anos_columna.eq(ano)).fillna(True).sum())
        duplicados_exactos = int(datos.duplicated().sum())
        huella = huella_sha256(ruta)
        # Algunos ZIP cambian LF por CRLF. Comparar además la huella del mismo
        # archivo normalizado a LF, sin modificar el archivo almacenado.
        contenido = ruta.read_bytes()
        huella_lf = hashlib.sha256(contenido.replace(b'\r\n', b'\n')).hexdigest()
        esperado = huellas_esperadas.get(ruta.name)
        coincide_huella = (esperado == huella) if esperado is not None else None
        coincide_tras_normalizar = (esperado == huella_lf) if esperado is not None else None
        coincide_conteo = (conteos_esperados.get(ruta.name) == len(datos)) if ruta.name in conteos_esperados else None
        fila = {
            'ano': ano, 'archivo': ruta.name, 'existe': True, 'legible': True,
            'columnas_completas': True, 'filas': len(datos),
            'departamentos_fuera_del_76': fuera_departamento,
            'codigos_municipio_vacios': sin_municipio,
            'semanas_fuera_de_1_a_53': semanas_invalidas,
            'anos_diferentes_al_archivo': ano_incorrecto,
            'fechas_INI_SIN_vacias': sin_fecha,
            'filas_exactamente_duplicadas': duplicados_exactos,
            'sha256_archivo_actual': huella,
            'coincide_sha256_manifiesto_byte_a_byte': coincide_huella,
            'coincide_sha256_al_normalizar_saltos_de_linea': coincide_tras_normalizar,
            'coincide_filas_manifiesto': coincide_conteo,
        }
        resumen.append(fila)
        for prueba, cantidad in [
            ('Departamento distinto de 76', fuera_departamento),
            ('Municipio sin código', sin_municipio),
            ('Semana fuera de rango', semanas_invalidas),
            ('Año distinto al nombre del archivo', ano_incorrecto),
        ]:
            if cantidad:
                incidencias.append({'archivo': ruta.name, 'nivel': 'ERROR', 'detalle': f'{prueba}: {cantidad}'})
        if (coincide_huella is False and coincide_tras_normalizar is False) or coincide_conteo is False:
            incidencias.append({'archivo': ruta.name, 'nivel': 'ERROR', 'detalle': 'Diferencia de contenido o conteo frente al manifiesto de reducción'})
        elif coincide_huella is False and coincide_tras_normalizar is True:
            incidencias.append({'archivo': ruta.name, 'nivel': 'INFO', 'detalle': 'Mismo contenido tras normalizar CRLF a LF; los bytes originales difieren'})
        if sin_fecha:
            incidencias.append({'archivo': ruta.name, 'nivel': 'AVISO', 'detalle': f'{sin_fecha} fechas INI_SIN vacías; NO se eliminan aquí'})
        if duplicados_exactos:
            incidencias.append({'archivo': ruta.name, 'nivel': 'AVISO', 'detalle': f'{duplicados_exactos} filas idénticas; pueden ser casos distintos, NO se eliminan'})

    tabla = pd.DataFrame(resumen)
    tabla.to_csv(output_dir / 'validacion_por_ano.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(incidencias, columns=['archivo', 'nivel', 'detalle']).to_csv(
        output_dir / 'incidencias.csv', index=False, encoding='utf-8-sig')
    total = int(tabla['filas'].fillna(0).sum()) if 'filas' in tabla else 0
    fechas = int(tabla['fechas_INI_SIN_vacias'].fillna(0).sum()) if 'fechas_INI_SIN_vacias' in tabla else 0
    errores = sum(i['nivel'] == 'ERROR' for i in incidencias)
    resultado = {
        'fecha_ejecucion_utc': datetime.now(timezone.utc).isoformat(),
        'directorio_datos': str(data_dir.resolve()),
        'archivos_esperados': len(ANOS),
        'archivos_legibles': int(tabla['legible'].fillna(False).sum()) if 'legible' in tabla else 0,
        'filas_totales': total,
        'fechas_vacias': fechas,
        'errores': errores,
        'avisos': sum(i['nivel'] == 'AVISO' for i in incidencias),
        'informativos': sum(i['nivel'] == 'INFO' for i in incidencias),
        'observaciones': [
            'Validación estructural: no demuestra exactitud epidemiológica ni correspondencia entre semana y fecha.',
            'Semanas de 1 a 53: solo rango; no se verificó el calendario epidemiológico de cada año.',
            'Los duplicados exactos se reportan pero no se eliminan: podrían representar casos distintos.',
            'Los archivos originales Excel no están en el ZIP: solo se verifica la huella del CSV contra el manifiesto.',
            'Una huella idéntica después de normalizar CRLF a LF prueba equivalencia de bytes normalizados, no identidad byte a byte del archivo ZIP.',
            'No se emplean datos de test para selección de modelos; este script no entrena ni selecciona modelos.',
        ],
    }
    (output_dir / 'resumen_auditoria.json').write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding='utf-8')
    lineas = [
        '# Bitácora — Auditoría inicial de los datos de dengue', '',
        f"Fecha (UTC): {resultado['fecha_ejecucion_utc']}", '',
        '## Objetivo',
        'Comprobar que los CSV anuales se pueden utilizar antes de preparar las series del modelo.', '',
        '## Procedimiento',
        'Se abrió cada CSV entre 2007 y 2025 conservando códigos como texto. Se comprobaron columnas, departamento, municipio, año, rango semanal, fechas vacías, duplicados y huellas SHA-256 del manifiesto.',
        'No se modificó ni eliminó ningún registro.', '',
        '## Resultados',
        f"- Archivos legibles: {resultado['archivos_legibles']} de {resultado['archivos_esperados']}.",
        f'- Registros: {total:,}.',
        f'- Fechas INI_SIN vacías: {fechas}.',
        f'- Errores estructurales o de integridad: {errores}.',
        f"- Avisos para revisión: {resultado['avisos']}.",
        f"- Notas informativas: {resultado['informativos']}.", '',
        '## Limitaciones',
        *[f'- {nota}' for nota in resultado['observaciones']], '',
        '## Archivos de evidencia',
        '- `validacion_por_ano.csv`: una fila por año con cada comprobación.',
        '- `incidencias.csv`: avisos y problemas detectados.',
        '- `resumen_auditoria.json`: resumen de la ejecución.', '',
        '## Decisión',
        ('No se detectaron errores estructurales en los CSV. Se puede continuar a la siguiente validación.' if errores == 0 else 'Se detectaron errores. Revisar incidencias.csv antes de seguir.'),
    ]
    (output_dir / 'BITACORA_VALIDACION.md').write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    return resultado



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Validar los archivos CSV de dengue'
    )

    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DATA_DIR_POR_DEFECTO
    )

    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent
    )

    
    args = parser.parse_args()
    resultado = auditar(args.data_dir, args.output_dir)

    
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
    raise SystemExit(1 if resultado['errores'] else 0)
