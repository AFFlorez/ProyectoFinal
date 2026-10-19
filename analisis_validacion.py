"""Diagnóstico de validación a partir de las predicciones de la sesión."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _metricas(tabla, columnas):
    filas = []
    for claves, g in tabla.groupby(columnas, observed=True, sort=True):
        claves = claves if isinstance(claves, tuple) else (claves,)
        error = g['CASOS_PREDICHOS'].to_numpy(float) - g['CASOS_REALES'].to_numpy(float)
        filas.append({**dict(zip(columnas, claves)), 'N': len(g),
            'MAE': np.abs(error).mean(), 'RMSE': np.sqrt(np.mean(error**2)),
            'SESGO_MEDIO': error.mean(), 'CASOS_MEDIOS': g['CASOS_ORIGINALES'].mean()})
    return pd.DataFrame(filas)


def _diferencia_relativa(candidato, referencia):
    if referencia == 0:
        return 'igual' if candidato == 0 else 'mayor'
    diferencia = 100 * (candidato - referencia) / referencia
    if abs(diferencia) < 1e-10:
        return 'igual'
    return f"un {abs(diferencia):.2f}% {'menor' if diferencia < 0 else 'mayor'}"


def preparar_analisis_validacion(predicciones, metricas, municipales, ranking, seleccion):
    """No ajusta modelos ni lee archivos; usa las variables calculadas antes de 18.2."""
    assert set(predicciones['CONJUNTO']) == {'training', 'validation'}
    assert predicciones['HORIZONTE'].eq(4).all()
    assert not predicciones.duplicated(['ID', 'CONJUNTO', 'COD_DANE', 'FECHA_OBJETIVO']).any()
    assert ranking['CONJUNTO'].eq('validation').all() and ranking['ID'].is_unique
    assert seleccion['ID_ELEGIDO'] in set(ranking['ID'])
    candidato = ranking.loc[ranking['ID'].eq(seleccion['ID_ELEGIDO'])].iloc[0]
    minimo_mae = ranking.loc[ranking['MAE'].idxmin()]
    motivo = seleccion.get('motivo_eleccion', 'Menor MAE general de validación.')
    if candidato['ID'] != minimo_mae['ID']:
        assert seleccion.get('seleccion_manual') and seleccion.get('motivo_eleccion')
    familias_baseline = ['Regresión lineal', 'Random Forest', 'XGBoost']
    baseline = ranking.loc[ranking['FAMILIA'].isin(familias_baseline)].sort_values(['MAE', 'RMSE', 'ID']).iloc[0]
    referencias = ranking.sort_values(['MAE', 'RMSE', 'ID']).groupby('FAMILIA', sort=False).head(1)
    ids_referencias = set(referencias['ID'])
    if candidato['ID'] not in ids_referencias:
        referencias = pd.concat([referencias, ranking.loc[ranking['ID'].eq(candidato['ID'])]], ignore_index=True)
    referencias = referencias.copy()
    referencias['ROL'] = referencias['ID'].map(lambda identificador:
        'Elegido; menor MAE de su familia' if identificador == candidato['ID'] and identificador in ids_referencias else
        'Modelo elegido para el análisis' if identificador == candidato['ID'] else 'Menor MAE de su familia')
    elegidos = [candidato['ID'], baseline['ID']]
    val = predicciones.loc[predicciones['CONJUNTO'].eq('validation')].copy()
    val['FECHA_OBJETIVO'] = pd.to_datetime(val['FECHA_OBJETIVO'])
    val['FECHA_EMISION'] = pd.to_datetime(val['FECHA_EMISION'])
    val['ERROR'] = val['CASOS_PREDICHOS'] - val['CASOS_REALES']
    assert set(val['ID']) == set(ranking['ID'])
    for identificador in referencias['ID']:
        g = val.loc[val['ID'].eq(identificador)]
        base = val.loc[val['ID'].eq(candidato['ID'])]
        claves = ['COD_DANE', 'FECHA_OBJETIVO']
        assert set(map(tuple, g[claves].to_numpy())) == set(map(tuple, base[claves].to_numpy()))

    familias = referencias[['ID', 'ROL', 'FAMILIA', 'MODELO', 'HISTORIA', 'N', 'MAE', 'RMSE']].copy()
    train = metricas.loc[metricas['CONJUNTO'].eq('training'), ['ID', 'MAE', 'RMSE']]
    familias = familias.merge(train.rename(columns={'MAE': 'MAE_TRAIN', 'RMSE': 'RMSE_TRAIN'}),
        on='ID', validate='one_to_one').rename(columns={'MAE': 'MAE_VAL', 'RMSE': 'RMSE_VAL'})
    mejor_rmse = ranking.loc[ranking['RMSE'].idxmin()]
    roles_eleccion = [(candidato['ID'], 'Modelo elegido'), (minimo_mae['ID'], 'Menor MAE general')]
    if candidato['FAMILIA'] == 'LSTM' and candidato['RECETA'] == 'drop20':
        roles_eleccion.append((f"lstm_referencia_v{int(candidato['HISTORIA'])}", 'Sin Dropout, misma historia'))
    filas_eleccion = {}
    for identificador, rol in roles_eleccion:
        fila = ranking.loc[ranking['ID'].eq(identificador)].iloc[0].to_dict()
        if identificador in filas_eleccion:
            filas_eleccion[identificador]['ROL'] += '; ' + rol
        else:
            filas_eleccion[identificador] = {**fila, 'ROL': rol}
    eleccion = pd.DataFrame(filas_eleccion.values())
    texto_familias = (
        f"Desde este punto analizamos **{candidato['MODELO']}**, con **MAE {candidato['MAE']:.3f}** "
        f"y **RMSE {candidato['RMSE']:.3f} casos** en {int(candidato['N'])} pronósticos de t+4. "
        f"Se eligió por el menor MAE de validación entre {len(ranking)} enfoques.\n\n"
        f"El mínimo de MAE sigue siendo **{minimo_mae['MODELO']}**, con MAE {minimo_mae['MAE']:.3f} "
        f"y RMSE {minimo_mae['RMSE']:.3f}. Frente a esa referencia, el MAE del modelo elegido es "
        f"{_diferencia_relativa(candidato['MAE'], minimo_mae['MAE'])} y su RMSE es "
        f"{_diferencia_relativa(candidato['RMSE'], minimo_mae['RMSE'])}. "
        "La primera tabla distingue la elección del mínimo de MAE y permite contrastar, cuando corresponde, "
        "la red sin Dropout con la misma cantidad de historia.\n\n"
        f"El baseline con menor MAE general es **{baseline['MODELO']}**: MAE {baseline['MAE']:.3f} "
        f"y RMSE {baseline['RMSE']:.3f}. Frente a él, el MAE del candidato es "
        f"{_diferencia_relativa(candidato['MAE'], baseline['MAE'])} y su RMSE es "
        f"{_diferencia_relativa(candidato['RMSE'], baseline['RMSE'])}.\n\n"
        f"El menor RMSE general corresponde a **{mejor_rmse['MODELO']}** "
        f"({mejor_rmse['RMSE']:.3f}). La tabla por familias conserva sus referencias de menor MAE "
        "y añade el modelo elegido, identificado en la columna ROL. Las columnas de entrenamiento describen el ajuste sobre datos ya vistos; "
        "su diferencia con validación también puede reflejar cambios entre periodos y no demuestra por sí sola sobreajuste."
        "\n\nNo se mantuvo una preferencia previa por una arquitectura o por usar Dropout. "
        "Estos resultados no prueban estabilidad fuera de este periodo: para estudiarla harían falta "
        "otros cortes temporales o semillas, sin utilizar evaluación para ajustar la decisión."
    )

    vc = val.loc[val['ID'].eq(candidato['ID'])].copy()
    mc = municipales.loc[municipales['CONJUNTO'].eq('validation') & municipales['ID'].eq(candidato['ID'])].copy()
    volumen = vc.groupby(['COD_DANE', 'MUNICIPIO'], observed=True).agg(
        CASOS_MEDIOS=('CASOS_ORIGINALES', 'mean'), ERROR_ABSOLUTO_TOTAL=('ERROR', lambda s: s.abs().sum())).reset_index()
    mc = mc.merge(volumen, on=['COD_DANE', 'MUNICIPIO'], validate='one_to_one')
    total_error = mc['ERROR_ABSOLUTO_TOTAL'].sum()
    mc['APORTE_ERROR_PCT'] = 100 * mc['ERROR_ABSOLUTO_TOTAL'] / total_error if total_error > 0 else 0.
    mc['SUBESTIMACION_PCT'] = 100 * mc['FRACCION_SUBESTIMACION']
    mc = mc.sort_values('COD_DANE').reset_index(drop=True)
    mayor = mc.loc[mc['MAE'].idxmax()]
    menor = mc.loc[mc['MAE'].idxmin()]
    principales = mc.nlargest(min(2, len(mc)), 'APORTE_ERROR_PCT')
    sesgo = float(vc['ERROR'].mean())
    direccion = 'subestima' if sesgo < 0 else 'sobreestima' if sesgo > 0 else 'no tiene sesgo medio'
    sesgo_texto = f"{direccion} en {abs(sesgo):.3f} casos por pronóstico, en promedio" if sesgo != 0 else direccion
    negativos = int(vc['CASOS_PREDICHOS'].lt(0).sum())
    texto_negativos = ('Se conservó **1 pronóstico negativo**, sin recortarlo.' if negativos == 1 else
        f'Se conservaron **{negativos} pronósticos negativos**, sin recortarlos.' if negativos else
        'El candidato no produjo pronósticos negativos en validación.')
    texto_municipios = (
        f"El MAE va de **{menor['MAE']:.3f} en {menor['MUNICIPIO']}** a "
        f"**{mayor['MAE']:.3f} en {mayor['MUNICIPIO']}**. Los dos mayores aportes al error absoluto "
        f"son {' y '.join(principales['MUNICIPIO'])}: juntos representan "
        f"**{principales['APORTE_ERROR_PCT'].sum():.2f}%** de ese error.\n\n"
        "Cada fila conserva el número de pronósticos y el promedio de casos observados. "
        "Un MAE pequeño en un municipio con pocos casos no demuestra una mejor precisión proporcional. "
        "El aporte al error indica qué municipios pesan más en la suma de errores absolutos; "
        "no es una medida de importancia sanitaria ni una tasa ajustada por población.\n\n"
        f"En el conjunto completo, el candidato **{sesgo_texto}**. "
        f"El {100 * vc['ERROR'].lt(-1e-3).mean():.2f}% de sus predicciones queda por debajo del valor real "
        "con una diferencia mayor que 0,001 casos. Un sesgo negativo indica subestimación y uno positivo, "
        "sobreestimación; no implica que todos los errores tengan el mismo signo. "
        + texto_negativos
    )

    mb = municipales.loc[municipales['CONJUNTO'].eq('validation') & municipales['ID'].eq(baseline['ID']),
        ['COD_DANE', 'MAE', 'RMSE']].rename(columns={'MAE': 'MAE_BASELINE', 'RMSE': 'RMSE_BASELINE'})
    comparacion = mc[['COD_DANE', 'MUNICIPIO', 'MAE', 'RMSE']].rename(
        columns={'MAE': 'MAE_CANDIDATO', 'RMSE': 'RMSE_CANDIDATO'}).merge(mb, on='COD_DANE', validate='one_to_one')
    comparacion['DIFERENCIA_MAE'] = comparacion['MAE_BASELINE'] - comparacion['MAE_CANDIDATO']
    comparacion['DIFERENCIA_RMSE'] = comparacion['RMSE_BASELINE'] - comparacion['RMSE_CANDIDATO']
    comparacion['REDUCCION_MAE_PCT'] = 100 * comparacion['DIFERENCIA_MAE'] / comparacion['MAE_BASELINE'].replace(0, np.nan)
    ciudades_baseline = comparacion.loc[comparacion['DIFERENCIA_MAE'].lt(0), 'MUNICIPIO'].tolist()
    detalle_baseline = (f"El baseline obtiene menor MAE en **{' y '.join(ciudades_baseline)}**. "
        if ciudades_baseline else '')
    texto_baseline = (
        f"Comparamos **{candidato['MODELO']}** con **{baseline['MODELO']}** en todas las ciudades. "
        "El candidato mantiene la elección registrada arriba y el baseline se fija por su menor MAE general. "
        f"El candidato tiene menor MAE en **{int(comparacion['DIFERENCIA_MAE'].gt(0).sum())} de {len(comparacion)} municipios** "
        f"y menor RMSE en **{int(comparacion['DIFERENCIA_RMSE'].gt(0).sum())} de {len(comparacion)}**. "
        + detalle_baseline + '\n\n'
        "Las diferencias se calculan como baseline menos candidato: un valor positivo favorece al candidato "
        "y uno negativo, al baseline. La reducción porcentual de MAE usa el error del baseline como referencia; "
        "no es un porcentaje del número de casos. Mantenemos los mismos dos modelos para todas las ciudades."
    )

    anual = val.loc[val['ID'].isin(elegidos)].copy()
    anual['ANO_OBJETIVO'] = anual['FECHA_OBJETIVO'].dt.year
    anual = _metricas(anual, ['ID', 'MODELO', 'ANO_OBJETIVO'])
    ac = anual.loc[anual['ID'].eq(candidato['ID'])].sort_values('ANO_OBJETIVO')
    peor_ano = ac.loc[ac['MAE'].idxmax()]
    lecturas = [f"**{int(r.ANO_OBJETIVO)}**: {int(r.N)} pronósticos, MAE {r.MAE:.3f}, RMSE {r.RMSE:.3f} "
        f"y promedio observado de {r.CASOS_MEDIOS:.2f} casos" for r in ac.itertuples()]
    texto_anual = (
        "Al separar el candidato por año de la semana objetivo encontramos: " + '; '.join(lecturas) + '.\n\n'
        f"El mayor MAE anual del candidato corresponde a **{int(peor_ano['ANO_OBJETIVO'])}**. "
        "El promedio de los dos años puede ocultar periodos con errores y volúmenes de casos distintos. "
        "La tabla incluye también el mismo baseline y muestra cuántos pronósticos entran en cada año. "
        "Este desglose conserva las ventanas y los modelos originales; no implica un ajuste anual nuevo."
    )

    mayores = vc.assign(ERROR_ABSOLUTO=vc['ERROR'].abs()).nlargest(min(10, len(vc)), 'ERROR_ABSOLUTO')
    suma_cuadrados = float((vc['ERROR']**2).sum())
    aporte = 100 * float((mayores['ERROR']**2).sum()) / suma_cuadrados if suma_cuadrados > 0 else 0.
    texto_mayores = (
        f"Los **{len(mayores)} mayores errores absolutos** representan **{aporte:.2f}%** de la suma de errores "
        f"cuadráticos del candidato. En **{int(mayores['ERROR'].lt(-1e-3).sum())} de esos {len(mayores)} "
        "pronósticos** se subestimaron los casos. Esto permite ver cuánto influyen unas pocas semanas en RMSE. "
        "La tabla muestra la fecha de emisión, la semana que se pronosticó, los casos observados y el error con signo.\n\n"
        "Estas semanas se identifican después de conocer los valores reales de validación: son un diagnóstico "
        "de errores, no una regla para anticipar brotes. El horizonte sigue siendo t+4 y no se incorporan esas "
        "observaciones futuras a las entradas del modelo."
    )
    detalle_mejores = municipales.loc[municipales['CONJUNTO'].eq('validation') &
        municipales['ID'].isin(ranking.head(6)['ID'])].copy()
    detalle_mejores['ORDEN_GENERAL'] = detalle_mejores['ID'].map(
        {identificador: i for i, identificador in enumerate(ranking.head(6)['ID'], start=1)})
    detalle_mejores = detalle_mejores.sort_values(['ORDEN_GENERAL', 'COD_DANE'])
    tablas = dict(eleccion=eleccion, familias=familias, municipios=mc, frente_baseline=comparacion, por_ano=anual,
        mayores_errores=mayores, mejores_configuraciones=detalle_mejores)
    textos = dict(eleccion='La elección registrada se contrasta con el mínimo de MAE y con la referencia sin Dropout de la misma historia.',
        familias=texto_familias, municipios=texto_municipios, frente_baseline=texto_baseline,
        por_ano=texto_anual, mayores_errores=texto_mayores,
        mejores_configuraciones='Se conserva el detalle municipal de las seis configuraciones con menor MAE general. '
        'Estas filas permiten contrastar alternativas; el modelo elegido para el análisis se mantiene fijo.')
    return dict(tablas=tablas, textos=textos, candidato=candidato['MODELO'], baseline=baseline['MODELO'],
        id_candidato=candidato['ID'], id_baseline=baseline['ID'], seleccion=dict(seleccion))


def graficar_comparacion_municipal(analisis, salida):
    tabla = analisis['tablas']['frente_baseline']
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    posiciones = np.arange(len(tabla))
    for eje, metrica in zip(ejes, ['MAE', 'RMSE']):
        eje.barh(posiciones - .18, tabla[metrica + '_CANDIDATO'], height=.34, color='#8D4CA6', label=analisis['candidato'])
        eje.barh(posiciones + .18, tabla[metrica + '_BASELINE'], height=.34, color='#238A76', label=analisis['baseline'])
        eje.set(yticks=posiciones, yticklabels=tabla['MUNICIPIO'], xlabel=f'{metrica} (casos)', title=metrica)
        eje.invert_yaxis()
        eje.grid(axis='x', alpha=.2)
    handles, labels = ejes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle('Validación · Candidato y mejor baseline por MAE general · Mismos modelos en los cinco municipios · t+4')
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(salida / f'comparacion_municipal.{extension}', dpi=220, bbox_inches='tight')
    return fig


def graficar_series_candidato(predicciones, analisis, ranking, salida):
    """Predicciones ya calculadas en validación, con un candidato fijo para las cinco ciudades."""
    minimo = ranking.loc[ranking['MAE'].idxmin()]
    ids = list(dict.fromkeys([analisis['id_candidato'], analisis['id_baseline'], minimo['ID']]))
    val = predicciones.loc[predicciones['CONJUNTO'].eq('validation') & predicciones['ID'].isin(ids)].copy()
    val['FECHA_OBJETIVO'] = pd.to_datetime(val['FECHA_OBJETIVO'])
    colores = dict(zip(ids, ['#8D4CA6', '#238A76', '#718196']))
    etiquetas = {analisis['id_candidato']: 'Elegido: ' + analisis['candidato'],
        analisis['id_baseline']: 'Baseline: ' + analisis['baseline']}
    if minimo['ID'] not in etiquetas:
        etiquetas[minimo['ID']] = 'Menor MAE: ' + minimo['MODELO']
    municipios = list(val.groupby('COD_DANE', observed=True, sort=True))
    fig, ejes = plt.subplots(len(municipios), 1, figsize=(14, 2.6 * len(municipios)), constrained_layout=True)
    for eje, (_, g) in zip(np.atleast_1d(ejes), municipios):
        real = g.loc[g['ID'].eq(analisis['id_candidato'])].sort_values('FECHA_OBJETIVO')
        eje.plot(real['FECHA_OBJETIVO'], real['CASOS_ORIGINALES'], color='#252525', label='Observado', linewidth=1.5)
        for identificador in ids:
            p = g.loc[g['ID'].eq(identificador)].sort_values('FECHA_OBJETIVO')
            eje.plot(p['FECHA_OBJETIVO'], p['CASOS_PREDICHOS'], color=colores[identificador],
                label=etiquetas[identificador], linewidth=1.05)
        localizador = mdates.AutoDateLocator(minticks=5, maxticks=8)
        eje.xaxis.set_major_locator(localizador)
        eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(localizador))
        eje.set(title=real['MUNICIPIO'].iloc[0], ylabel='Casos', xlabel='Inicio de la semana objetivo t+4')
        eje.grid(alpha=.2)
    handles, labels = np.atleast_1d(ejes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=8)
    fig.suptitle('Validación · Modelo elegido y referencias fijas · Mismos modelos en los cinco municipios · t+4')
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(salida / f'series_candidato_validacion.{extension}', dpi=220, bbox_inches='tight')
    return fig


def guardar_analisis_validacion(analisis, salida):
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    partes = ['# Análisis de validación del punto 18.2']
    for clave, tabla in analisis['tablas'].items():
        tabla.to_csv(salida / f'{clave}.csv', index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
        partes.append(analisis['textos'][clave])
    partes.append('Estas lecturas describen una semilla y el bloque de validación. La selección continúa siendo provisional; '
        'no se ha comprobado superioridad estadística ni generalización a otros periodos. El conjunto de evaluación permanece reservado.')
    (salida / 'ANALISIS_VALIDACION.md').write_text('\n\n'.join(partes), encoding='utf-8')
    (salida / 'seleccion_analisis.json').write_text(json.dumps(analisis['seleccion'], ensure_ascii=False, indent=2,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')
