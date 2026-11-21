"""Experimento Poisson
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelado_notebook import construir_secuencias

HISTORIAS = (8, 12, 16, 20, 24)
ALPHAS = (0.01, 0.1)
HORIZONTE = 4
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent


ID_LSTM = 'COD_DANE'
OBJETIVO_LSTM = 'CASOS'
TEMPERATURAS_LSTM = ['TEMP_MEDIA_C', 'TEMP_MAX_C', 'TEMP_MIN_C']
PRECIPITACION_LSTM = 'PRECIPITACION_MM'
CLIMA_LSTM = [*TEMPERATURAS_LSTM, PRECIPITACION_LSTM]
CLAVES_LSTM = [ID_LSTM, 'ANO', 'SEMANA_EPI']
PERIODOS_LSTM = {'training': (2007, 2021), 'validation': (2022, 2023), 'test': (2024, 2025)}
columnas_requeridas_lstm = [
    'COD_DPTO_O', 'COD_MUN_O', ID_LSTM, 'MUNICIPIO', 'ANO', 'SEMANA_EPI',
    'FECHA_INICIO', 'FECHA_FIN', OBJETIVO_LSTM, *CLIMA_LSTM,
]



def ajustar_escaladores_lstm(entrenamiento):
    """Ajustar únicamente con la partición de entrenamiento ya validada."""
    inicio, fin = PERIODOS_LSTM['training']
    if entrenamiento.empty or not entrenamiento['ANO'].between(inicio, fin).all():
        raise ValueError('El ajuste requiere exclusivamente entrenamiento de 2007–2021.')
    escaladores_casos = {}
    auditoria = []
    for municipio, grupo in entrenamiento.groupby(ID_LSTM, sort=True):
        escalador = StandardScaler()
        escalador.fit(np.log1p(grupo[[OBJETIVO_LSTM]].to_numpy(dtype=float)))
        escaladores_casos[municipio] = escalador
        auditoria.append({
            'VARIABLE': OBJETIVO_LSTM, 'AMBITO': municipio, 'CONJUNTO_AJUSTE': 'training',
            'FILAS_AJUSTE': len(grupo), 'ANO_MIN': int(grupo['ANO'].min()), 'ANO_MAX': int(grupo['ANO'].max()),
        })
    escaladores_clima = {
        'temperaturas': StandardScaler().fit(entrenamiento[TEMPERATURAS_LSTM].to_numpy(dtype=float)),
        'precipitacion': StandardScaler().fit(np.log1p(entrenamiento[[PRECIPITACION_LSTM]].to_numpy(dtype=float))),
    }
    for variable in ['Temperaturas (tres columnas)', PRECIPITACION_LSTM]:
        auditoria.append({
            'VARIABLE': variable, 'AMBITO': 'Global: cinco municipios', 'CONJUNTO_AJUSTE': 'training',
            'FILAS_AJUSTE': len(entrenamiento), 'ANO_MIN': int(entrenamiento['ANO'].min()), 'ANO_MAX': int(entrenamiento['ANO'].max()),
        })
    return escaladores_casos, escaladores_clima, pd.DataFrame(auditoria)


def transformar_conjunto_lstm(datos, escaladores_casos, escaladores_clima, semanas_por_ano):
    """Conservar filas, índice y originales; añadir columnas sin reajustar escaladores."""
    salida = datos.copy(deep=True)
    salida['CASOS_log1p'] = np.log1p(salida[OBJETIVO_LSTM].to_numpy(dtype=float))
    salida['CASOS_std'] = np.nan
    for municipio in salida[ID_LSTM].unique():
        if municipio not in escaladores_casos:
            raise ValueError(f'Municipio sin escalador de entrenamiento: {municipio}')
        mascara = salida[ID_LSTM].eq(municipio)
        salida.loc[mascara, 'CASOS_std'] = escaladores_casos[municipio].transform(
            salida.loc[mascara, ['CASOS_log1p']].to_numpy(dtype=float)
        ).ravel()
    columnas_temperatura_std = [f'{columna}_std' for columna in TEMPERATURAS_LSTM]
    salida[columnas_temperatura_std] = escaladores_clima['temperaturas'].transform(
        salida[TEMPERATURAS_LSTM].to_numpy(dtype=float)
    )
    salida['PRECIPITACION_MM_log1p'] = np.log1p(salida[PRECIPITACION_LSTM].to_numpy(dtype=float))
    salida['PRECIPITACION_MM_std'] = escaladores_clima['precipitacion'].transform(
        salida[['PRECIPITACION_MM_log1p']].to_numpy(dtype=float)
    ).ravel()
    # Semana 1 tiene fase cero. El denominador procede del calendario completo
    # de ese año, no del máximo de semana observado en una ventana futura.
    numero_semanas = salida['ANO'].map(semanas_por_ano)
    if numero_semanas.isna().any() or not salida['SEMANA_EPI'].between(1, numero_semanas).all():
        raise ValueError('Año o semana sin correspondencia con el calendario epidemiológico.')
    fase = 2 * np.pi * (salida['SEMANA_EPI'].to_numpy(dtype=float) - 1) / numero_semanas.to_numpy(dtype=float)
    salida['semana_sin'] = np.sin(fase)
    salida['semana_cos'] = np.cos(fase)
    return salida



def preparar_secuencias(raiz: Path = RAIZ_PROYECTO):
    """Lee CSV semanales, verifica integridad y reproduce secciones 12 y 13."""
    raiz = Path(raiz).resolve()
    conjuntos_base_lstm = {}
    for nombre in PERIODOS_LSTM:
        origen = raiz / 'datos' / 'DatosProcesados' / f'semanal_{nombre}.csv'
        if not origen.is_file():
            raise FileNotFoundError(f'Falta el conjunto semanal: {origen}')
        datos = pd.read_csv(
            origen, encoding='utf-8-sig',
            dtype={'COD_DPTO_O': 'string', 'COD_MUN_O': 'string', ID_LSTM: 'string'},
        )
        ausentes = sorted(set(columnas_requeridas_lstm) - set(datos.columns))
        if ausentes:
            raise ValueError(f'{nombre}: faltan columnas requeridas: {ausentes}')
        for columna in ['FECHA_INICIO', 'FECHA_FIN']:
            datos[columna] = pd.to_datetime(datos[columna], errors='raise')
        conjuntos_base_lstm[nombre] = datos

    faltantes_lstm = pd.DataFrame({
        nombre: datos[columnas_requeridas_lstm].replace(r'^\s*$', pd.NA, regex=True).isna().sum()
        for nombre, datos in conjuntos_base_lstm.items()
    })
    if faltantes_lstm.to_numpy().any():
        print(faltantes_lstm.loc[faltantes_lstm.any(axis=1)].to_string())
        raise ValueError('Hay campos requeridos faltantes. Definir una estrategia explícita antes de estandarizar; no se imputaron ni eliminaron filas.')

    orden_municipios_lstm = sorted(conjuntos_base_lstm['training'][ID_LSTM].unique())
    if len(orden_municipios_lstm) != 5:
        raise ValueError('Se esperan los cinco municipios seleccionados en el punto 10.')

    for nombre, datos in conjuntos_base_lstm.items():
        inicio, fin = PERIODOS_LSTM[nombre]
        if set(datos[ID_LSTM]) != set(orden_municipios_lstm):
            raise ValueError(f'{nombre}: municipios diferentes de los de entrenamiento.')
        if not datos[ID_LSTM].astype('string').str.fullmatch(r'76\d{3}').all():
            raise ValueError(f'{nombre}: COD_DANE debe conservarse como código de cinco caracteres del Valle.')
        if not datos['COD_DPTO_O'].astype('string').eq('76').all():
            raise ValueError(f'{nombre}: contiene datos de otro departamento.')
        if not datos['ANO'].between(inicio, fin).all() or set(datos['ANO']) != set(range(inicio, fin + 1)):
            raise ValueError(f'{nombre}: periodo incompatible con la separación temporal existente.')
        if datos.duplicated(CLAVES_LSTM).any():
            raise ValueError(f'{nombre}: municipio/año/semana duplicados.')
        valores = datos[[OBJETIVO_LSTM, *CLIMA_LSTM, 'ANO', 'SEMANA_EPI']].to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            raise ValueError(f'{nombre}: se encontraron valores no numéricos o infinitos.')
        if (datos[[OBJETIVO_LSTM, PRECIPITACION_LSTM]] < 0).any().any():
            raise ValueError(f'{nombre}: casos o precipitación negativos; log1p no se aplicará.')
        if not np.equal(datos[OBJETIVO_LSTM], np.floor(datos[OBJETIVO_LSTM])).all():
            raise ValueError(f'{nombre}: CASOS debe contener conteos enteros.')
        if not (datos['FECHA_FIN'] - datos['FECHA_INICIO']).dt.days.eq(6).all():
            raise ValueError(f'{nombre}: hay semanas que no abarcan siete días.')
        for municipio, grupo in datos.groupby(ID_LSTM, sort=False):
            if not grupo['FECHA_INICIO'].is_monotonic_increasing:
                raise ValueError(f'{nombre}/{municipio}: orden temporal incorrecto.')
            if not grupo['FECHA_INICIO'].diff().dropna().dt.days.eq(7).all():
                raise ValueError(f'{nombre}/{municipio}: semanas faltantes o fechas repetidas.')

    
    calendario_lstm = pd.concat([
        datos[['ANO', 'SEMANA_EPI', 'FECHA_INICIO', 'FECHA_FIN']]
        for datos in conjuntos_base_lstm.values()
    ], ignore_index=True).drop_duplicates().sort_values(['ANO', 'SEMANA_EPI'])
    if calendario_lstm.duplicated(['ANO', 'SEMANA_EPI']).any():
        raise ValueError('El calendario de una semana no coincide entre municipios.')
    semanas_por_ano_lstm = {}
    for ano, grupo in calendario_lstm.groupby('ANO'):
        semanas = grupo['SEMANA_EPI'].tolist()
        numero = len(semanas)
        if numero not in (52, 53) or semanas != list(range(1, numero + 1)):
            raise ValueError(f'{ano}: calendario incompleto o incompatible con 52/53 semanas.')
        semanas_por_ano_lstm[int(ano)] = numero
    for nombre, datos in conjuntos_base_lstm.items():
        esperado = calendario_lstm.loc[calendario_lstm['ANO'].between(*PERIODOS_LSTM[nombre]), ['ANO', 'SEMANA_EPI']]
        for municipio, grupo in datos.groupby(ID_LSTM):
            if not grupo[['ANO', 'SEMANA_EPI']].reset_index(drop=True).equals(esperado.reset_index(drop=True).astype(grupo[['ANO', 'SEMANA_EPI']].dtypes)):
                raise ValueError(f'{nombre}/{municipio}: no comparte el calendario completo.')
    for anterior, siguiente in [('training', 'validation'), ('validation', 'test')]:
        ultima = conjuntos_base_lstm[anterior]['FECHA_INICIO'].max()
        primera = conjuntos_base_lstm[siguiente]['FECHA_INICIO'].min()
        if primera - ultima != pd.Timedelta(days=7):
            raise ValueError('Los conjuntos temporales se solapan o no son consecutivos.')

    # 
    escaladores_casos_lstm, escaladores_clima_lstm, auditoria_ajuste_lstm = ajustar_escaladores_lstm(
        conjuntos_base_lstm['training']
    )
    conjuntos_transformados_lstm = {
        nombre: transformar_conjunto_lstm(
            datos, escaladores_casos_lstm, escaladores_clima_lstm, semanas_por_ano_lstm
        )
        for nombre, datos in conjuntos_base_lstm.items()
    }
    df_lstm_entrenamiento = conjuntos_transformados_lstm['training']
    df_lstm_validacion = conjuntos_transformados_lstm['validation']
    df_lstm_test = conjuntos_transformados_lstm['test']

    columnas_entrada_lstm = [
        'CASOS_std', *[f'{columna}_std' for columna in TEMPERATURAS_LSTM],
        'PRECIPITACION_MM_std', 'semana_sin', 'semana_cos',
    ]
    columna_objetivo_lstm = 'CASOS_std'
    codificador_municipios_actual = OneHotEncoder(
        sparse_output=False, handle_unknown='error', dtype=np.float32
    )
    codificador_municipios_actual.fit(df_lstm_entrenamiento[['COD_DANE']])
    secuencias_desarrollo = construir_secuencias(
        df_lstm_entrenamiento, df_lstm_validacion, columnas_entrada_lstm,
        codificador_municipios_actual, historias=HISTORIAS, horizonte=HORIZONTE,
    )
    secuencias_prueba = construir_secuencias(
        df_lstm_validacion, df_lstm_test, columnas_entrada_lstm,
        codificador_municipios_actual, historias=HISTORIAS, horizonte=HORIZONTE,
    )
    return secuencias_desarrollo, secuencias_prueba


def _extraer(parte, n_esperado: int, nombre: str):
    if len(parte['matriz']) != n_esperado or len(parte['meta']) != n_esperado:
        raise ValueError(f'{nombre}: se esperaban {n_esperado} filas')
    columnas = {'CASOS_ORIGINALES', 'COD_DANE', 'FECHA_OBJETIVO'}
    if not columnas.issubset(parte['meta'].columns):
        raise ValueError(f'{nombre}: metadatos incompletos')
    x = np.asarray(parte['matriz'], dtype=float)
    y = parte['meta']['CASOS_ORIGINALES'].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f'{nombre}: existen valores no finitos')
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ValueError(f'{nombre}: se requieren conteos enteros no negativos')
    return x, y


def evaluar_poisson(secuencias_desarrollo, salida=RAIZ_PROYECTO / 'ResultadosModelos' / 'poisson',
                    historias=HISTORIAS, alphas=ALPHAS, max_iter=6000, tol=1e-6):
    """Ajusta en train, selecciona solo por MAE validación y exporta resultados."""
    resultados, predicciones = [], []
    for historia in historias:
        tr = secuencias_desarrollo[historia]['training']
        va = secuencias_desarrollo[historia]['validation']
        xtr, ytr = _extraer(tr, 3780, f'entrenamiento {historia}')
        xva, yva = _extraer(va, 505, f'validación {historia}')
        for alpha in alphas:
            modelo = PoissonRegressor(alpha=alpha, fit_intercept=True,
                                     solver='lbfgs', max_iter=max_iter, tol=tol)
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter('always', ConvergenceWarning)
                modelo.fit(xtr, ytr)
            converge = not any(issubclass(a.category, ConvergenceWarning) for a in avisos)
            pred = modelo.predict(xva)
            if not np.isfinite(pred).all():
                raise ValueError(f'Predicciones no finitas: {historia}, alpha={alpha}')
            codigo = f'poisson_v{historia}_alpha{alpha}'
            resultados.append({
                'ID': codigo, 'HISTORIA': historia, 'ALPHA': alpha,
                'CONVERGIO': converge, 'ITERACIONES': int(modelo.n_iter_),
                'N': len(yva), 'MAE': mean_absolute_error(yva, pred),
                'RMSE': np.sqrt(mean_squared_error(yva, pred)),
                'SESGO_MEDIO': float(np.mean(pred - yva)),
            })
            tabla = va['meta'][['COD_DANE', 'FECHA_OBJETIVO', 'CASOS_ORIGINALES']].copy().reset_index(drop=True)
            tabla['ID'] = codigo
            tabla['CASOS_PREDICHOS'] = pred
            predicciones.append(tabla)
    tabla_resultados = pd.DataFrame(resultados)
    ranking = (tabla_resultados.loc[tabla_resultados['CONVERGIO']]
               .sort_values(['MAE', 'RMSE']).reset_index(drop=True))
    if ranking.empty:
        raise RuntimeError('Ninguna configuración convergió')
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    tabla_resultados.to_csv(salida / 'metricas_validacion.csv', index=False, encoding='utf-8-sig')
    pd.concat(predicciones, ignore_index=True).to_csv(
        salida / 'predicciones_validacion.csv', index=False, encoding='utf-8-sig'
    )
    ranking.to_csv(salida / 'ranking_validacion.csv', index=False, encoding='utf-8-sig')
    config = {
        'seleccion': ranking.iloc[0]['ID'],
        'criterio': 'menor MAE validacion solo entre convergidos',
        'horizonte': 't+4', 'historias': list(historias), 'alphas': list(alphas),
        'max_iter': max_iter, 'tol': tol, 'uso_test': 'no se utilizo para seleccion',
    }
    (salida / 'configuracion.json').write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return {'metricas': tabla_resultados, 'ranking': ranking,
            'mejor': ranking.iloc[0].to_dict(), 'ruta': salida}


def evaluar_poisson_prueba(secuencias_desarrollo, secuencias_prueba,
                            salida=RAIZ_PROYECTO / 'ResultadosModelos' / 'poisson'):
    """Reajusta solo el ganador en train y evalúa una vez en test."""
    carpeta = Path(salida)
    configuracion = json.loads((carpeta / 'configuracion.json').read_text(encoding='utf-8'))
    seleccion = configuracion['seleccion']
    coincidencia = re.fullmatch(r'poisson_v(\d+)_alpha(.+)', seleccion)
    if coincidencia is None:
        raise ValueError('Identificador de Poisson no reconocido')
    historia, alpha = int(coincidencia.group(1)), float(coincidencia.group(2))
    entrenamiento = secuencias_desarrollo[historia]['training']
    prueba = secuencias_prueba[historia]['validation']  # Segundo argumento de construir_secuencias.
    xtr, ytr = _extraer(entrenamiento, 3780, 'entrenamiento final')
    xte, yte = _extraer(prueba, 510, 'prueba final')
    modelo = PoissonRegressor(
        alpha=alpha, fit_intercept=True, solver='lbfgs',
        max_iter=int(configuracion['max_iter']), tol=float(configuracion['tol']),
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        modelo.fit(xtr, ytr)
    pred = modelo.predict(xte)
    if not np.isfinite(pred).all():
        raise ValueError('Predicciones no finitas en prueba')
    tabla = pd.read_csv(carpeta / 'metricas_validacion.csv')
    candidatas = tabla.loc[tabla['ID'] == seleccion]
    if len(candidatas) != 1:
        raise ValueError('Selección no encontrada de forma única en métricas de validación')
    fila = candidatas.iloc[0]
    resultado = pd.DataFrame([{
        'MODELO': seleccion, 'HORIZONTE': configuracion['horizonte'],
        'N_TEST': len(yte),
        'MAE_VALIDATION': float(fila['MAE']),
        'RMSE_VALIDATION': float(fila['RMSE']),
        'MAE_EVALUATION': mean_absolute_error(yte, pred),
        'RMSE_EVALUATION': np.sqrt(mean_squared_error(yte, pred)),
    }])
    tabla_pred = prueba['meta'][['COD_DANE', 'FECHA_OBJETIVO', 'CASOS_ORIGINALES']].copy().reset_index(drop=True)
    tabla_pred['ID'] = seleccion
    tabla_pred['CASOS_PREDICHOS'] = pred
    resultado.to_csv(carpeta / 'metricas_finales.csv', index=False, encoding='utf-8-sig')
    tabla_pred.to_csv(carpeta / 'predicciones_prueba.csv', index=False, encoding='utf-8-sig')
    return resultado


def main():
    desarrollo, prueba = preparar_secuencias()
    resultado_validacion = evaluar_poisson(desarrollo)
    resultado_prueba = evaluar_poisson_prueba(desarrollo, prueba)
    print('Mejor Poisson por validación:')
    print(pd.DataFrame([resultado_validacion['mejor']]).to_string(index=False))
    print('Evaluación final (sin selección mediante prueba):')
    print(resultado_prueba.to_string(index=False))
    print('Resultados guardados en:', resultado_validacion['ruta'])


if __name__ == '__main__':
    main()
