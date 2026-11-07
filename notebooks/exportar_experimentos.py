"""Exportar a modelos/experimentos/ las tablas y figuras de la comparación de modelos.

Lee las salidas ya guardadas en ProyectoFinal.ipynb (no vuelve a entrenar) y deja
archivos pequeños y versionables que infra/mlflow/registrar_experimentos.py carga
en MLflow. Ejecutar desde la raíz después de correr el notebook completo:

    python notebooks/exportar_experimentos.py
"""
import base64
import io
import json
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOK = RAIZ / 'notebooks' / 'ProyectoFinal.ipynb'
DESTINO = RAIZ / 'modelos' / 'experimentos'

# Celdas del notebook: el código que produce cada salida se identifica por su texto,
# así un cambio en la numeración de las celdas no altera la exportación.
TABLAS = {
    'ranking_validacion.csv': ('predicciones_desarrollo = pd.concat(', 0),
    'arquitecturas.csv': ("estructuras_mostradas = [{'MODELO'", 0),
    'metricas_baselines.csv': ('metricas_baselines, metricas_municipales_baselines, ranking_baselines', 0),
    'resumen_epocas.csv': ('resumen_epocas = registro_entrenamiento_redes.copy()', 0),
    'evaluacion_prueba.csv': ('EJECUTAR_EVALUACION_FINAL = True', 0),
    'metricas_tres_conjuntos.csv': ('metricas_tres_conjuntos_cierre, figuras_finales = graficar_al_cierre(', 0),
}
FIGURAS = {
    'baselines_validacion': ('metricas_baselines, metricas_municipales_baselines, ranking_baselines', [0]),
    'curva_aprendizaje_modelo_elegido': ('# La regresión lineal y Random Forest no tienen una curva por época.', [0]),
    'elegido_vs_xgboost_por_municipio': ('mostrar_tabla_validacion(comparacion_baseline_municipal[', [0]),
    'elegido_vs_xgboost_series': ('mostrar_tabla_validacion(comparacion_baseline_municipal[', [1]),
    'locales_comparacion': ("colores_locales = {'LSTM32 + Dense16'", [0]),
    'lstm64_comparacion': ('# Comparar seis estructuras de 64 unidades', [0]),
    'lstm64_curvas': ('# Comparar seis estructuras de 64 unidades', [1]),
    'evaluacion_tres_conjuntos': ('metricas_tres_conjuntos_cierre, figuras_finales = graficar_al_cierre(', [0, 1, 2, 3, 4]),
    'modelo_elegido_validacion_prueba': ('# Reutilizar exclusivamente las predicciones ya calculadas del modelo', [0]),
}


def celda(celdas, inicio):
    encontradas = [c for c in celdas if c['cell_type'] == 'code' and ''.join(c['source']).lstrip().startswith(inicio)]
    if len(encontradas) != 1:
        raise ValueError(f'Se esperaba una celda que empiece con {inicio!r}; hay {len(encontradas)}.')
    if not encontradas[0].get('outputs'):
        raise ValueError(f'La celda {inicio!r} no tiene salidas: ejecutar el notebook completo antes de exportar.')
    return encontradas[0]


def aplanar(tabla):
    """Encabezados de varios niveles (tablas pivote) a una sola fila: MAE_VALIDATION, ..."""
    if not isinstance(tabla.columns, pd.MultiIndex):
        return tabla
    nombres = []
    for niveles in tabla.columns:
        utiles = [str(n) for n in niveles if not str(n).startswith('Unnamed')]
        indice = str(niveles[0]).startswith('Unnamed')  # Columna del índice: su nombre está en el último nivel.
        nombres.append(utiles[-1] if indice or len(utiles) == 1 else '_'.join(utiles[:2]).upper())
    tabla.columns = nombres
    return tabla


def tablas_html(c):
    return [aplanar(pd.read_html(io.StringIO(''.join(o['data']['text/html'])))[0])
            for o in c['outputs'] if 'text/html' in o.get('data', {})]


def figuras_png(c):
    return [base64.b64decode(o['data']['image/png']) for o in c['outputs'] if 'image/png' in o.get('data', {})]


def nombre_base(modelo):
    return re.sub(r' · \d+s$', '', modelo)


def main():
    celdas = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    (DESTINO / 'figuras').mkdir(parents=True, exist_ok=True)

    tablas = {}
    for archivo, (inicio, posicion) in TABLAS.items():
        tabla = tablas_html(celda(celdas, inicio))[posicion]
        tablas[archivo] = tabla.drop(columns=[c for c in tabla.columns if str(c).startswith('Unnamed')])

    ranking = tablas['ranking_validacion.csv']
    if len(ranking) != 100 or ranking['MODELO'].duplicated().any():
        raise ValueError('El ranking debe tener los 100 enfoques sin repetir.')
    arquitecturas = tablas['arquitecturas.csv'].assign(BASE=lambda t: t['MODELO'].map(nombre_base))
    ranking.insert(0, 'POSICION', range(1, len(ranking) + 1))
    ranking['BASE'] = ranking['MODELO'].map(nombre_base)
    ranking = ranking.merge(arquitecturas.drop(columns='MODELO'), on='BASE', how='left', validate='many_to_one')
    ranking['FAMILIA'] = ranking['BASE'].map(
        lambda b: 'LSTM local' if 'local' in b else 'LSTM global' if b.startswith('LSTM') else b)
    tablas['ranking_validacion.csv'] = ranking.drop(columns='BASE')

    for archivo, tabla in tablas.items():
        tabla.to_csv(DESTINO / archivo, index=False, encoding='utf-8')
        print(f'{archivo}: {len(tabla)} filas')

    for nombre, (inicio, posiciones) in FIGURAS.items():
        imagenes = figuras_png(celda(celdas, inicio))
        for k, posicion in enumerate(posiciones, start=1):
            sufijo = f'_{k}' if len(posiciones) > 1 else ''
            (DESTINO / 'figuras' / f'{nombre}{sufijo}.png').write_bytes(imagenes[posicion])
    print(f'Figuras: {len(list((DESTINO / "figuras").glob("*.png")))} en {DESTINO.relative_to(RAIZ)}/figuras')


if __name__ == '__main__':
    main()
