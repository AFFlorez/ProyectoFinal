"""Evaluación final reproducible del modelo seleccionado por validación.

No entrena, selecciona o reajusta usando el conjunto de evaluación.
"""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import importlib.metadata
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from inferencia_dengue import PronosticadorDengue

REFERENCIA_HISTORIAS = Path('ResultadosModelos/historia_12_16_20_t4/ejecucion_ba352a18')
REFERENCIA_ARQUITECTURAS = Path('ResultadosModelos/arquitecturas_lstm_global_t4/ejecucion_1db6e884')
REFERENCIA_PREPROCESAMIENTO = Path('ResultadosModelos/lstm_global/ejecucion_7f082bbd/preprocesamiento_existente.joblib')
MODELOS = ['Regresión lineal', 'Random Forest', 'LSTM global']
CONJUNTOS = ['training', 'validation', 'evaluation']
ETIQUETAS = {'training':'Entrenamiento · ajuste sobre datos usados para entrenar',
             'validation':'Validación · usada para seleccionar modelo y checkpoint',
             'evaluation':'Evaluación final · test 2024–2025'}
COLORES = dict(zip(MODELOS, ['#176B93', '#238A76', '#8D4CA6']))


def huella(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def congelar_y_exportar():
    """Registrar la elección y copiar modelos antes de evaluar los casos del test."""
    historia = REFERENCIA_HISTORIAS / 'ventana_16'
    metricas = pd.read_csv(REFERENCIA_HISTORIAS/'resumen_tres_ventanas.csv', encoding='utf-8-sig')
    arquitecturas = pd.read_csv(REFERENCIA_ARQUITECTURAS/'comparacion_arquitecturas.csv', encoding='utf-8-sig')
    for metrica in ['MAE', 'RMSE']:
        ganador = metricas.loc[metricas[metrica].idxmin()]
        assert ganador['MODELO']=='LSTM global' and ganador['LOOKBACK']==16
        assert arquitecturas.loc[arquitecturas[metrica].idxmin(),'MODELO']=='LSTM 32 (referencia)'
    prep = joblib.load(REFERENCIA_PREPROCESAMIENTO)
    cfg = json.loads((historia/'configuracion.json').read_text(encoding='utf-8'))
    assert cfg['lookback']==16 and cfg['horizonte_pronostico']==4 and cfg['n_salidas']==1
    ruta = Path('ResultadosModelos/evaluacion_final_t4') / f'ejecucion_{uuid4().hex[:8]}'
    paquete = ruta / 'modelo_exportado'
    for p in [paquete, ruta/'tablas', ruta/'figuras']:
        p.mkdir(parents=True, exist_ok=False)
    origenes = {
        'modelo_lstm_global_t4.keras':historia/'modelos/lstm_global_t4_v16/mejor_modelo.keras',
        'regresion_lineal_t4.joblib':historia/'modelos/regresion_lineal.joblib',
        'random_forest_t4.joblib':historia/'modelos/random_forest.joblib',
    }
    decision = {
        'fecha_utc':datetime.now(timezone.utc).isoformat(),
        'modelo_seleccionado':'LSTM global de 32 unidades; Dense 16 ReLU; salida lineal t+4',
        'lookback':16, 'horizonte':4, 'seleccion':'Menor MAE y RMSE de validación entre las configuraciones comparadas',
        'MAE_validacion':float(ganador['MAE']), 'RMSE_validacion':float(ganador['RMSE']),
        'referencia_historia':str(historia), 'referencia_arquitecturas':str(REFERENCIA_ARQUITECTURAS),
        'baselines':'Regresión lineal y Random Forest de 16 semanas, fijados antes del test; no se buscan sus mejores ventanas en el test.',
        'reentrenamiento':False, 'ajuste_con_validacion':False, 'test_usado_para_seleccionar':False,
        'politica_salida':'Continua, sin redondear ni recortar; se informa la cantidad de valores negativos.',
        'disponibilidad_seleccion':'Evaluar solo emisiones desde el fin de la última semana de validación: el checkpoint ya debe poder haberse elegido.',
        'archivos_elegidos':{nombre:{'origen':str(p),'sha256':huella(p)} for nombre,p in origenes.items()},
        'preprocesamiento_sha256':huella(REFERENCIA_PREPROCESAMIENTO),
    }
    # Este archivo se escribe antes de abrir los casos de evaluación.
    (ruta/'seleccion_antes_test.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    for nombre, p in origenes.items():
        shutil.copy2(p, paquete/nombre)
        assert huella(p)==huella(paquete/nombre)
    joblib.dump(prep, paquete/'preprocesamiento.joblib', compress=3)
    shutil.copy2(Path(__file__).with_name('inferencia_dengue.py'), paquete/'inferencia_dengue.py')
    calendario = pd.concat([pd.read_csv(f'DatosProcesados/semanal_{c}.csv',usecols=['ANO','SEMANA_EPI','FECHA_INICIO','FECHA_FIN'],encoding='utf-8-sig') for c in ['training','validation','test']],ignore_index=True).drop_duplicates().sort_values(['ANO','SEMANA_EPI'])
    assert not calendario.duplicated(['ANO','SEMANA_EPI']).any()
    for _,g in calendario.groupby('ANO'):
        assert len(g) in [52,53] and g['SEMANA_EPI'].tolist()==list(range(1,len(g)+1))
    calendario.to_csv(paquete/'calendario_semanal.csv',index=False,encoding='utf-8-sig')
    cfg_exportado = {**decision, 'columnas_temporales':prep['columnas_temporales'],
        'municipios':{'76001':'CALI','76111':'GUADALAJARA DE BUGA','76520':'PALMIRA','76834':'TULUA','76892':'YUMBO'},
        'categorias_onehot':prep['codificador_municipios'].categories_[0].tolist(),
        'semanas_por_ano':{str(int(a)):int(len(g)) for a,g in calendario.groupby('ANO')},
        'entradas_modelo':{'historia':[None,16,7], 'municipio':[None,5]},
        'salida_modelo':[None,1], 'transformacion_casos':'log1p y StandardScaler municipal; inversión inverse_transform y expm1',
        'entrenamiento':'2007–2021; checkpoint elegido con validación 2022–2023',
        'actualizacion':'Historia observada hasta t; pesos fijos; no se emplea clima futuro.',
    }
    (paquete/'configuracion.json').write_text(json.dumps(cfg_exportado,ensure_ascii=False,indent=2),encoding='utf-8')
    versiones = {p:importlib.metadata.version(p) for p in ['tensorflow','numpy','pandas','scikit-learn','joblib']}
    (paquete/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versiones.items())+'\n',encoding='utf-8')
    return ruta, paquete, decision


def preparar_conjuntos(paquete):
    """Conservar train/validation y construir todos los objetivos del test con contexto pasado."""
    predictor = PronosticadorDengue(paquete)
    fuentes = {}
    for split,archivo,inicio,fin in [('training','training',2007,2021),('validation','validation',2022,2023),('evaluation','test',2024,2025)]:
        d = pd.read_csv(f'DatosProcesados/semanal_{archivo}.csv',dtype={'COD_DANE':'string'},parse_dates=['FECHA_INICIO','FECHA_FIN'],encoding='utf-8-sig')
        assert d['ANO'].between(inicio,fin).all() and set(d['COD_DANE'])==set(predictor.config['municipios'])
        fuentes[split] = d
    semanas = pd.concat(fuentes.values(),ignore_index=True)
    transformadas = predictor.transformar_semanas(semanas)
    for _,g in transformadas.groupby('COD_DANE',observed=True):
        assert g.sort_values('FECHA_INICIO')['FECHA_INICIO'].diff().dropna().dt.days.eq(7).all()
    tabla = transformadas.set_index(['COD_DANE','FECHA_INICIO']).sort_index()
    congelados = joblib.load(REFERENCIA_HISTORIAS/'ventana_16/insumos_control.joblib')
    datos = {}
    for split, sufijo in [('training','train'),('validation','val')]:
        datos[split] = {'X':congelados[f'X_{sufijo}'].copy(),'y':congelados[f'y_{sufijo}'].copy(),
                        'metadata':congelados[f'metadata_{sufijo}'].copy(deep=True)}
    filas=[]
    for r in fuentes['evaluation'].sort_values(['COD_DANE','FECHA_INICIO']).itertuples():
        ultima = r.FECHA_INICIO-pd.Timedelta(weeks=4)
        filas.append({'FILA_TENSOR':len(filas),'COD_DANE':str(r.COD_DANE),'MUNICIPIO':r.MUNICIPIO,
            'FECHA_INICIO_HISTORICO':ultima-pd.Timedelta(weeks=15),'FECHA_ULTIMA_ENTRADA':ultima,
            'FECHA_EMISION':ultima+pd.Timedelta(days=6),'FECHA_OBJETIVO':r.FECHA_INICIO,
            'HORIZONTE':4,'CONJUNTO':'evaluation'})
    meta_test = pd.DataFrame(filas)
    # El checkpoint se eligió con toda la validación. Sus últimas observaciones
    # no estaban disponibles para emisiones anteriores al cierre de ese bloque.
    corte_seleccion = fuentes['validation']['FECHA_FIN'].max()
    disponibles = meta_test['FECHA_EMISION'].ge(corte_seleccion)
    excluidas = meta_test.loc[~disponibles].copy()
    excluidas['MOTIVO'] = 'Emisión anterior al cierre del bloque usado para elegir el checkpoint'
    excluidas['CORTE_SELECCION'] = corte_seleccion
    meta_test = meta_test.loc[disponibles].reset_index(drop=True)
    meta_test['FILA_TENSOR'] = np.arange(len(meta_test))
    assert meta_test['FECHA_EMISION'].ge(corte_seleccion).all()
    datos['evaluation']={'metadata':meta_test,'objetivos_excluidos':excluidas,'corte_seleccion':corte_seleccion}
    for split,d in datos.items():
        meta = d['metadata']
        meta['COD_DANE'] = meta['COD_DANE'].astype('string')
        X = np.empty((len(meta),16,7),np.float32)
        y = np.empty((len(meta),1),np.float32)
        raw,anios,semanas_objetivo=[],[],[]
        for i,r in enumerate(meta.itertuples()):
            fechas = pd.date_range(end=r.FECHA_ULTIMA_ENTRADA,periods=16,freq='7D')
            indice = pd.MultiIndex.from_product([[r.COD_DANE],fechas])
            X[i] = tabla.reindex(indice)[predictor.columnas].to_numpy(np.float32)
            objetivo = tabla.loc[(r.COD_DANE,r.FECHA_OBJETIVO)]
            y[i,0] = np.float32(objetivo['CASOS_std'])
            raw.append(int(objetivo['CASOS'])); anios.append(int(objetivo['ANO'])); semanas_objetivo.append(int(objetivo['SEMANA_EPI']))
            assert fechas[-1]+pd.Timedelta(weeks=4)==r.FECHA_OBJETIVO
            assert fechas[-1]+pd.Timedelta(days=6)==r.FECHA_EMISION < r.FECHA_OBJETIVO
        assert np.isfinite(X).all() and np.isfinite(y).all()
        if split!='evaluation':
            np.testing.assert_array_equal(X,d['X'])
            np.testing.assert_array_equal(y,d['y'])
        d['X'],d['y']=X,y
        meta['ANO_OBJETIVO']=anios
        meta['SEMANA_EPI_OBJETIVO']=semanas_objetivo
        meta['CASOS_ORIGINALES']=raw
        np.testing.assert_allclose(predictor.invertir(y,meta['COD_DANE']),raw,rtol=1e-5,atol=1e-3)
        assert not meta.duplicated(['COD_DANE','FECHA_OBJETIVO']).any()
    assert len(datos['training']['X'])==3800 and len(datos['validation']['X'])==505 and len(datos['evaluation']['X'])==510
    assert len(excluidas)==15
    for a,b in [('training','validation'),('validation','evaluation')]:
        assert datos[a]['metadata']['FECHA_OBJETIVO'].max()<datos[b]['metadata']['FECHA_OBJETIVO'].min()
    ejemplo = fuentes['training'].sort_values(['COD_DANE','FECHA_INICIO']).groupby('COD_DANE',observed=True).tail(16)
    ejemplo.to_csv(Path(paquete)/'ejemplo_historia_16_semanas.csv',index=False,encoding='utf-8-sig')
    return predictor, datos, fuentes


def calcular_metricas(predicciones, grupos):
    filas=[]
    for claves,g in predicciones.groupby(grupos,observed=True,sort=False):
        claves = claves if isinstance(claves,tuple) else (claves,)
        error = g['CASOS_PREDICHOS'].to_numpy()-g['CASOS_REALES'].to_numpy()
        filas.append({**dict(zip(grupos,claves)), 'N_PRONOSTICOS':len(g),
            'MAE':float(np.abs(error).mean()),'RMSE':float(np.sqrt(np.mean(error**2))),
            'SESGO_MEDIO':float(error.mean()), 'FRACCION_SUBESTIMACION':float(np.mean(error < -1e-3)),
            'PRONOSTICOS_NEGATIVOS':int(g['CASOS_PREDICHOS'].lt(0).sum())})
    return pd.DataFrame(filas)


def evaluar_modelos(predictor, datos, ruta):
    tablas=[]
    pred_std={}
    for split in CONJUNTOS:
        d=datos[split]; meta=d['metadata']
        pred_std[split]=predictor.predecir_tensores(d['X'],meta['COD_DANE'])
        reales = predictor.invertir(d['y'],meta['COD_DANE'])
        for modelo in MODELOS:
            tabla=meta[['FILA_TENSOR','COD_DANE','MUNICIPIO','FECHA_INICIO_HISTORICO','FECHA_ULTIMA_ENTRADA','FECHA_EMISION','FECHA_OBJETIVO','ANO_OBJETIVO','SEMANA_EPI_OBJETIVO','HORIZONTE','CONJUNTO','CASOS_ORIGINALES']].copy()
            tabla.insert(0,'MODELO',modelo)
            tabla['CASOS_REALES']=reales
            tabla['CASOS_PREDICHOS']=predictor.invertir(pred_std[split][modelo],meta['COD_DANE'])
            tabla['ERROR']=tabla['CASOS_PREDICHOS']-tabla['CASOS_REALES']
            tabla['PRONOSTICO_NEGATIVO']=tabla['CASOS_PREDICHOS'].lt(0)
            tablas.append(tabla)
    predicciones = pd.concat(tablas,ignore_index=True)
    # La validación reproduce el experimento que sustentó la selección.
    previa = pd.read_csv(REFERENCIA_HISTORIAS/'ventana_16/predicciones_modelos.csv',dtype={'COD_DANE':'string'},encoding='utf-8-sig')
    for modelo in MODELOS:
        a = predicciones.loc[predicciones['CONJUNTO'].eq('validation')&predicciones['MODELO'].eq(modelo)].sort_values('FILA_TENSOR')
        b = previa.loc[previa['MODELO'].eq(modelo)].sort_values('FILA_TENSOR')
        np.testing.assert_array_equal(a['COD_DANE'],b['COD_DANE'])
        np.testing.assert_allclose(a[['CASOS_REALES','CASOS_PREDICHOS']],b[['CASOS_REALES','CASOS_PREDICHOS']],rtol=2e-5,atol=1e-4)
    general=calcular_metricas(predicciones,['CONJUNTO','MODELO'])
    municipal=calcular_metricas(predicciones,['CONJUNTO','MODELO','COD_DANE','MUNICIPIO'])
    macro=municipal.groupby(['CONJUNTO','MODELO'],observed=True).agg(MAE_PROMEDIO_MUNICIPAL=('MAE','mean'),RMSE_PROMEDIO_MUNICIPAL=('RMSE','mean')).reset_index()
    general=general.merge(macro,on=['CONJUNTO','MODELO'],validate='one_to_one')
    cobertura = pd.concat([d['metadata'].assign(CONJUNTO=c) for c,d in datos.items()],ignore_index=True).groupby(['CONJUNTO','COD_DANE','MUNICIPIO'],observed=True).agg(
        VENTANAS=('FILA_TENSOR','size'),PRIMER_OBJETIVO=('FECHA_OBJETIVO','min'),ULTIMO_OBJETIVO=('FECHA_OBJETIVO','max'),ANO_EPI_MIN=('ANO_OBJETIVO','min'),ANO_EPI_MAX=('ANO_OBJETIVO','max')).reset_index()
    filas=[]
    for split,g in general.groupby('CONJUNTO',sort=False):
        t=g.set_index('MODELO')
        for base in MODELOS[:2]:
            for metrica in ['MAE','RMSE']:
                error_base=t.loc[base,metrica]; error_lstm=t.loc['LSTM global',metrica]
                filas.append({'CONJUNTO':split,'BASELINE':base,'METRICA':metrica,'ERROR_BASELINE':error_base,'ERROR_LSTM':error_lstm,
                    'MEJORA_LSTM_PCT':100*(error_base-error_lstm)/error_base if error_base else np.nan})
    mejoras=pd.DataFrame(filas)
    tablas_resultados={'predicciones_todos_conjuntos':predicciones,'metricas_generales':general,
        'metricas_municipales':municipal,'cobertura':cobertura,'mejoras_frente_baselines':mejoras}
    for nombre,tabla in tablas_resultados.items():
        tabla.to_csv(Path(ruta)/'tablas'/f'{nombre}.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    for split in CONJUNTOS:
        predicciones.loc[predicciones['CONJUNTO'].eq(split)].to_csv(Path(ruta)/'tablas'/f'predicciones_{split}.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    datos['evaluation']['objetivos_excluidos'].to_csv(Path(ruta)/'tablas/objetivos_excluidos_por_disponibilidad.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    joblib.dump(datos,Path(ruta)/'insumos_evaluacion.joblib',compress=3)
    return predicciones, general, municipal, cobertura, mejoras


def graficar_resultados(predicciones, metricas, municipales, ruta):
    ruta=Path(ruta)
    figuras=[]
    def guardar(fig,nombre):
        for ext in ['png','svg']:
            fig.savefig(ruta/'figuras'/f'{nombre}.{ext}',dpi=300,bbox_inches='tight')
        figuras.append(nombre)
        plt.show()
    with plt.rc_context({'font.size':10,'svg.fonttype':'none'}):
        fig,ejes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        for fila,split in enumerate(CONJUNTOS):
            g=metricas.loc[metricas['CONJUNTO'].eq(split)].set_index('MODELO').reindex(MODELOS)
            for eje,metrica in zip(ejes[fila],['MAE','RMSE']):
                b=eje.barh(MODELOS,g[metrica],color=[COLORES[m] for m in MODELOS])
                eje.bar_label(b,fmt='%.2f',padding=4)
                eje.set(title=f'{split} · {metrica}',xlabel=f'{metrica} (casos)',xlim=(0,max(g[metrica].max()*1.22,1)))
                eje.grid(axis='x',alpha=.2)
        fig.suptitle('Modelos congelados de 16 semanas · Objetivo t+4\nTrain: ajuste · Validation: selección · Evaluation: test final')
        guardar(fig,'01_metricas_train_validation_evaluation')
        for i,(codigo,g) in enumerate(predicciones.groupby('COD_DANE',observed=True,sort=True),2):
            fig,ejes=plt.subplots(3,1,figsize=(13,9),constrained_layout=True)
            for eje,split in zip(ejes,CONJUNTOS):
                periodo=g.loc[g['CONJUNTO'].eq(split)]
                real=periodo.loc[periodo['MODELO'].eq('LSTM global')].sort_values('FECHA_OBJETIVO')
                eje.plot(real['FECHA_OBJETIVO'],real['CASOS_ORIGINALES'],label='Observado',color='#252525',linewidth=1.6)
                for modelo in MODELOS:
                    p=periodo.loc[periodo['MODELO'].eq(modelo)].sort_values('FECHA_OBJETIVO')
                    eje.plot(p['FECHA_OBJETIVO'],p['CASOS_PREDICHOS'],label=modelo,color=COLORES[modelo],linewidth=1.05,alpha=.9)
                loc=mdates.AutoDateLocator(minticks=5,maxticks=9)
                eje.xaxis.set_major_locator(loc); eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                eje.set(title=ETIQUETAS[split],ylabel='Casos semanales',xlabel='Inicio de la semana objetivo t+4')
                eje.grid(alpha=.2)
            handles,labels=ejes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside lower center',ncol=4)
            fig.suptitle(f'{g["MUNICIPIO"].iloc[0]} · Reales y pronosticados · 16 semanas de historia\nCada panel tiene su propia escala temporal y vertical; una salida t+4')
            guardar(fig,f'{i:02d}_series_{codigo}_t4')
        fig,ejes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
        test=municipales.loc[municipales['CONJUNTO'].eq('evaluation')]
        for eje,metrica in zip(ejes,['MAE','RMSE']):
            tabla=test.pivot(index='MUNICIPIO',columns='MODELO',values=metrica).reindex(columns=MODELOS)
            tabla.plot.barh(ax=eje,color=[COLORES[m] for m in MODELOS])
            eje.set(title=metrica,xlabel=f'{metrica} (casos)',ylabel='');eje.grid(axis='x',alpha=.2)
        handles,labels=ejes[0].get_legend_handles_labels()
        for eje in ejes:eje.get_legend().remove()
        fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle('Evaluación final 2024–2025 · Errores municipales · Objetivo t+4')
        guardar(fig,'07_metricas_municipales_evaluation')
        fig,ejes=plt.subplots(1,3,figsize=(14,5),constrained_layout=True)
        test=predicciones.loc[predicciones['CONJUNTO'].eq('evaluation')]
        nombres=sorted(test['MUNICIPIO'].unique())
        for eje,modelo in zip(ejes,MODELOS):
            g=test.loc[test['MODELO'].eq(modelo)]
            eje.boxplot([g.loc[g['MUNICIPIO'].eq(m),'ERROR'] for m in nombres],tick_labels=nombres,orientation='horizontal',showfliers=True,showmeans=True)
            eje.axvline(0,color='#666666',linestyle='--',linewidth=1)
            eje.set(title=modelo,xlabel='Pronóstico − observado (casos)');eje.grid(axis='x',alpha=.2)
        fig.suptitle('Evaluación final · Residuos por municipio · t+4\nSe conservan valores extremos; escalas independientes por modelo')
        guardar(fig,'08_residuos_evaluation')
    return figuras


def generar_conclusiones(metricas, municipales, mejoras):
    test=metricas.loc[metricas['CONJUNTO'].eq('evaluation')].set_index('MODELO')
    val=metricas.loc[metricas['CONJUNTO'].eq('validation')].set_index('MODELO')
    train=metricas.loc[metricas['CONJUNTO'].eq('training')].set_index('MODELO')
    mejor_mae,mejor_rmse=test['MAE'].idxmin(),test['RMSE'].idxmin()
    lstm=test.loc['LSTM global']
    lecturas=[]
    for base in MODELOS[:2]:
        cambios=[]
        for metrica in ['MAE','RMSE']:
            x=float(mejoras.loc[mejoras['CONJUNTO'].eq('evaluation')&mejoras['BASELINE'].eq(base)&mejoras['METRICA'].eq(metrica),'MEJORA_LSTM_PCT'].iloc[0])
            cambios.append(f"{metrica}: {abs(x):.2f}% de {'reducción' if x>=0 else 'aumento'}")
        lecturas.append(f"frente a {base} ({'; '.join(cambios)})")
    municipales_test=municipales.loc[municipales['CONJUNTO'].eq('evaluation')]
    detalle=[]
    for nombre,g in municipales_test.groupby('MUNICIPIO',sort=True):
        mae=g.loc[g['MAE'].idxmin()];rmse=g.loc[g['RMSE'].idxmin()]
        detalle.append(f"{nombre}: menor MAE con {mae['MODELO']} ({mae['MAE']:.3f}) y menor RMSE con {rmse['MODELO']} ({rmse['RMSE']:.3f})")
    return f"""**Modelo evaluado.** Se congeló la LSTM global de 32 unidades, Dense de 16, identidad municipal one-hot e historia de 16 semanas por su resultado de validación. Se exportó el checkpoint existente y los escaladores de entrenamiento. No hubo reentrenamiento, recalibración ni selección con el test.

**Resultado fuera de muestra.** Sobre las 510 semanas-municipio elegibles del test, la LSTM obtiene MAE **{lstm['MAE']:.3f} casos** y RMSE **{lstm['RMSE']:.3f} casos**. Los menores errores entre estos tres modelos congelados corresponden a **{mejor_mae}** en MAE y **{mejor_rmse}** en RMSE. La comparación de la LSTM es: {'; '.join(lecturas)}. Son diferencias descriptivas; no se realizaron pruebas de significación ni intervalos que consideren la dependencia temporal. Este resultado no se usa para cambiar retrospectivamente el modelo elegido.

**Train, validación y evaluación.** Los MAE de la LSTM son {train.loc['LSTM global','MAE']:.3f}, {val.loc['LSTM global','MAE']:.3f} y {lstm['MAE']:.3f}; sus RMSE son {train.loc['LSTM global','RMSE']:.3f}, {val.loc['LSTM global','RMSE']:.3f} y {lstm['RMSE']:.3f}, respectivamente. Entrenamiento muestra ajuste dentro de muestra y validación participó en la selección. El test aporta la evaluación fuera de muestra después de cerrar esa selección. Las diferencias entre periodos también pueden reflejar cambios en el volumen o distribución de casos: por sí solas no prueban una causa ni sobreajuste.

**Resultados municipales.** {'; '.join(detalle)}. Un mínimo agrupado no garantiza el mínimo en cada municipio. Las gráficas conservan observados, tres pronósticos y los valores extremos. La evaluación se refiere a estos cinco municipios y este bloque temporal; no demuestra el mismo desempeño en otras zonas o años.

**Cómo se emitieron los pronósticos.** La historia se actualiza cada semana con observaciones hasta t, manteniendo pesos fijos, y se pronostica la semana t+4. Durante el test se pueden usar semanas de test anteriores a la emisión, como ocurriría al actualizar el histórico, nunca valores posteriores a t. No es un pronóstico de dos años desde una fecha única ni la suma del próximo mes. Se supone disponibilidad oportuna de los casos y el clima; los retrasos reales de publicación no están modelados.

**Cobertura y límites.** Se conservaron las 3.800 ventanas de entrenamiento y 505 de validación originales. En evaluación se incluyen 102 de las 105 semanas epidemiológicas de 2024–2025 por municipio, del 21 de enero de 2024 al 28 de diciembre de 2025. Se excluyen los objetivos del 31 de diciembre de 2023, 7 y 14 de enero de 2024 porque sus emisiones serían anteriores al 30 de diciembre de 2023, cierre de la validación usada para elegir el checkpoint. Los tres modelos usan las mismas fechas elegibles. El calendario sigue siendo el del insumo: la semana 53 de 2025 termina el 3 de enero de 2026. La salida principal no se redondea ni recorta: la LSTM produce {int(lstm['PRONOSTICOS_NEGATIVOS'])} valores negativos en el test, que se señalan si aparecen. Sigue pendiente contrastar persistencia o una referencia estacional, y estudiar estabilidad con más cortes o semillas. La evaluación no demuestra efectos causales del clima, detección formal de brotes ni beneficio sanitario de una intervención. Cualquier nueva selección que use este test lo convertiría en otro conjunto de desarrollo y necesitaría un periodo independiente posterior.

**Corrección de cobertura durante la verificación.** El primer cálculo incluyó las 525 semanas-municipio preparadas. Al comprobar cuándo habría estado disponible la selección del checkpoint se detectó que 15 emisiones eran anteriores al cierre de validación. Se corrigió únicamente esa cobertura, por fechas y antes de cerrar la entrega: se conservaron todos los pesos, escaladores y configuraciones. No se eligieron exclusiones según los errores del test. El cálculo inicial queda identificado como preliminar y no se utiliza como resultado final.
"""


def cerrar_exportacion(ruta, paquete, decision, predictor, datos, fuentes, predicciones, metricas, municipales, mejoras):
    ruta,paquete=Path(ruta),Path(paquete)
    conclusiones=generar_conclusiones(metricas,municipales,mejoras)
    (ruta/'RESULTADOS_Y_CONCLUSIONES.md').write_text('# Evaluación final del pronóstico t+4\n\n'+conclusiones,encoding='utf-8')
    instrucciones='''# Modelo exportado: dengue, historia de 16 semanas, objetivo t+4

Este paquete contiene la LSTM seleccionada por validación, los dos baselines de 16 semanas, los escaladores y el codificador municipal ya ajustados. No necesita las carpetas originales de resultados ni el notebook para pronosticar. La salida principal es continua y no se recorta; PRONOSTICO_NEGATIVO permite identificar valores que requieren tratamiento antes de uso operativo.

## Uso

Instala requirements.txt en un entorno compatible. Desde esta carpeta:

```python
import pandas as pd
from inferencia_dengue import PronosticadorDengue

predictor = PronosticadorDengue('.')
historia = pd.read_csv('ejemplo_historia_16_semanas.csv', dtype={'COD_DANE': 'string'})
predicciones = predictor.pronosticar(historia)
print(predicciones)
```

El ejemplo contiene las últimas 16 semanas de entrenamiento de cada municipio y pronostica la primera semana objetivo de la validación utilizada. Para otros pronósticos sustituye el ejemplo por las observaciones disponibles hasta la semana de emisión. Se requieren COD_DANE, ANO, SEMANA_EPI, FECHA_INICIO, FECHA_FIN, CASOS, TEMP_MEDIA_C, TEMP_MAX_C, TEMP_MIN_C y PRECIPITACION_MM. Los códigos deben ser strings; no se imputan faltantes. Se utiliza la última ventana de 16 semanas consecutivas por municipio.

El one-hot municipal se construye con el codificador exportado, conservando el orden de sus categorías. El modelo devuelve casos de la cuarta semana futura, no un acumulado mensual. No proporciona intervalos de incertidumbre. Los datos históricos deben haber estado disponibles al emitir; no se modelan retrasos de notificación.

El calendario exportado cubre 2007–2025. Para entradas de años posteriores, pasa semanas_por_ano={2026: numero_verificado} a pronosticar; debe ser 52 o 53 según el calendario epidemiológico que corresponda. No se infiere ese número desde una ventana parcial ni se utiliza el calendario ISO como sustituto. Verifica también las fechas de esas nuevas semanas.

Los tres modelos conservan su entrenamiento 2007–2021; la LSTM usa el checkpoint elegido con validación 2022–2023. No se ajustaron con el test. Consulta configuracion.json para configuración y procedencia. La evaluación final se conserva en la carpeta superior, sin usarla para reajustar el paquete.
'''
    (paquete/'README.md').write_text(instrucciones,encoding='utf-8')
    pred_ejemplo=predictor.pronosticar(pd.read_csv(paquete/'ejemplo_historia_16_semanas.csv',dtype={'COD_DANE':'string'},encoding='utf-8-sig'))
    for modelo in MODELOS:
        actual=pred_ejemplo.loc[pred_ejemplo['MODELO'].eq(modelo)].sort_values('COD_DANE')
        ref=predicciones.loc[predicciones['CONJUNTO'].eq('validation')&predicciones['MODELO'].eq(modelo)].sort_values('FECHA_OBJETIVO').groupby('COD_DANE',observed=True).head(1).sort_values('COD_DANE')
        np.testing.assert_array_equal(actual['FECHA_OBJETIVO'],ref['FECHA_OBJETIVO'])
        np.testing.assert_allclose(actual['CASOS_PREDICHOS'],ref['CASOS_PREDICHOS'],rtol=2e-5,atol=1e-4)
    pred_ejemplo.to_csv(paquete/'ejemplo_predicciones.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    entradas={f'DatosProcesados/semanal_{c}.csv':huella(f'DatosProcesados/semanal_{c}.csv') for c in ['training','validation','test']}
    manifiesto={
        'decision_congelada':decision,'test_evaluado':True,'reentrenamiento':False,
        'n_ventanas':{c:len(d['X']) for c,d in datos.items()},
        'corte_seleccion':str(datos['evaluation']['corte_seleccion']),
        'objetivos_excluidos_por_disponibilidad':len(datos['evaluation']['objetivos_excluidos']),
        'periodos':{'training':'2007–2021','validation':'2022–2023','evaluation':'2024–2025 (años epidemiológicos)'},
        'insumos_sha256':entradas,'salidas':{str(p.relative_to(ruta)):huella(p) for p in ruta.rglob('*') if p.is_file()},
        'politica_test':'Pesos y configuración congelados antes de evaluar. No seleccionar ni ajustar con estos resultados.'}
    (ruta/'manifiesto_evaluacion.json').write_text(json.dumps(manifiesto,ensure_ascii=False,indent=2),encoding='utf-8')
    zip_path=Path(shutil.make_archive(str(ruta/'modelo_exportado_t4_v16'),'zip',root_dir=paquete))
    return conclusiones, zip_path
