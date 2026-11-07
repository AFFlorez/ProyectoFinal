"""Utilidades del notebook: entradas actuales, entrenamientos nuevos y validación.

Este módulo no busca ni recupera experimentos anteriores al importarse.
Las llamadas a fit de los modelos predictivos están en ProyectoFinal.ipynb.
"""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import gc
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HISTORIAS = (8, 12, 16, 20, 24)
HORIZONTE = 4
EPOCAS_MAX = 200
PACIENCIA_ES = 20
RECETAS_64 = {
    'B64': {'unidades':[64], 'dense':16, 'dropout':0.},
    '64_dense16_drop20': {'unidades':[64], 'dense':16, 'dropout':.2},
    '64_dense8': {'unidades':[64], 'dense':8, 'dropout':0.},
    '64_dense8_drop20': {'unidades':[64], 'dense':8, 'dropout':.2},
    '64_sin_dense': {'unidades':[64], 'dense':0, 'dropout':0.},
    '64_sin_dense_drop20': {'unidades':[64], 'dense':0, 'dropout':.2},
}
RECETAS = {
    'referencia': {'unidades':[32], 'dense':16, 'dropout':0., 'descripcion':'32 · Dense16 · D0'},
    'sin_dense': {'unidades':[32], 'dense':0, 'dropout':0., 'descripcion':'32 · sin Dense intermedia'},
    'drop20': {'unidades':[32], 'dense':16, 'dropout':.2, 'descripcion':'32 · Dense16 · D0,20'},
    'drop50': {'unidades':[32], 'dense':16, 'dropout':.5, 'descripcion':'32 · Dense16 · D0,50'},
    'dos_capas': {'unidades':[32,16], 'dense':16, 'dropout':.2, 'descripcion':'32→16 · Dense16 · D0,20'},
    'tres_capas': {'unidades':[32,16,8], 'dense':16, 'dropout':.2, 'descripcion':'32→16→8 · Dense16 · D0,20'},
}
CONFIG_XGB = dict(n_estimators=800, max_depth=3, learning_rate=.03, min_child_weight=5,
    subsample=.8, colsample_bytree=.8, reg_alpha=0., reg_lambda=1.,
    objective='reg:squarederror', eval_metric='rmse', early_stopping_rounds=50,
    random_state=42, n_jobs=4, tree_method='hist', device='cpu')


def construir_secuencias(entrenamiento, validacion, columnas, codificador,
                         historias=HISTORIAS, horizonte=HORIZONTE):
    """Ventanas nuevas, con soporte común y sin usar evaluación ni artefactos."""
    historias = tuple(sorted(historias))
    maximo = max(historias)
    panel = pd.concat([entrenamiento.assign(CONJUNTO='training'),
                       validacion.assign(CONJUNTO='validation')], ignore_index=True)
    panel = panel.sort_values(['COD_DANE','FECHA_INICIO']).reset_index(drop=True)
    assert not panel.duplicated(['COD_DANE','FECHA_INICIO']).any()
    assert set(panel['COD_DANE']) == set(entrenamiento['COD_DANE'])
    corte = entrenamiento['FECHA_FIN'].max()
    datos = {v:{} for v in historias}
    buffers = {(v,c): {'X':[], 'y':[], 'meta':[]} for v in historias for c in ['training','validation']}
    for codigo, g in panel.groupby('COD_DANE', sort=True, observed=True):
        g = g.reset_index(drop=True)
        assert g['FECHA_INICIO'].diff().dropna().dt.days.eq(7).all()
        valores = g[columnas].to_numpy(np.float32)
        assert np.isfinite(valores).all()
        for objetivo in range(maximo-1+horizonte, len(g)):
            r = g.iloc[objetivo]
            fin = objetivo-horizonte
            ultima = g.iloc[fin]
            emision = ultima['FECHA_FIN']
            conjunto = r['CONJUNTO']
            if conjunto=='validation' and emision < corte:
                continue
            assert ultima['FECHA_INICIO']+pd.Timedelta(weeks=horizonte)==r['FECHA_INICIO']
            assert emision < r['FECHA_INICIO']
            for v in historias:
                b = buffers[v,conjunto]
                b['X'].append(valores[fin-v+1:fin+1].copy())
                b['y'].append([np.float32(r['CASOS_std'])])
                b['meta'].append(dict(FILA_TENSOR=len(b['meta']), COD_DANE=str(codigo),
                    MUNICIPIO=r['MUNICIPIO'], CONJUNTO=conjunto, HORIZONTE=horizonte,
                    FECHA_PRIMERA_ENTRADA=g.iloc[fin-v+1]['FECHA_INICIO'],
                    FECHA_ULTIMA_ENTRADA=ultima['FECHA_INICIO'], FECHA_EMISION=emision,
                    FECHA_OBJETIVO=r['FECHA_INICIO'], CASOS_ORIGINALES=int(r['CASOS'])))
    for v in historias:
        for c in ['training','validation']:
            b = buffers[v,c]
            X = np.asarray(b['X'], dtype=np.float32)
            y = np.asarray(b['y'], dtype=np.float32)
            meta = pd.DataFrame(b['meta'])
            meta['COD_DANE'] = meta['COD_DANE'].astype('string')
            assert X.ndim==3 and X.shape[1:]==(v,len(columnas)) and y.shape==(len(X),1)
            aux = codificador.transform(meta[['COD_DANE']]).astype(np.float32)
            matriz = np.concatenate([X.reshape(len(X),-1).astype(float),aux],axis=1)
            datos[v][c] = dict(X=X, y=y, meta=meta, aux=aux, matriz=matriz)
            base = datos[historias[0]][c]
            np.testing.assert_array_equal(y,base['y'])
            for campo in ['COD_DANE','FECHA_OBJETIVO','FECHA_EMISION']:
                np.testing.assert_array_equal(meta[campo],base['meta'][campo])
            np.testing.assert_array_equal(X[:,-historias[0]:],base['X'])
            assert meta.groupby('COD_DANE',observed=True)['FECHA_OBJETIVO'].apply(
                lambda s:s.diff().dropna().dt.days.eq(7).all()).all()
    return datos


def catalogo_redes():
    """85 enfoques LSTM: 70 globales y 15 conjuntos de cinco redes locales."""
    filas=[]
    for familia in ['LSTM']:
        for v in HISTORIAS:
            for receta,cfg in RECETAS.items():
                filas.append(dict(ID=f'{familia.lower()}_{receta}_v{v}',FAMILIA=familia,
                    RECETA=receta,HISTORIA=v,NOMBRE=f'{familia} {cfg["descripcion"]} · {v}s',
                    TIPO='red',CONFIG=cfg.copy(),MATRIZ=True))
    for receta,cfg in RECETAS_64.items():
        for v in HISTORIAS:
            prefijo='lstm_adicional_B64' if receta=='B64' else f'lstm_{receta}'
            intermedia=f'Dense{cfg["dense"]}' if cfg['dense'] else 'sin Dense intermedia'
            dropout=str(cfg['dropout']).replace('.',',')
            filas.append(dict(ID=f'{prefijo}_v{v}',FAMILIA='LSTM',RECETA=receta,
                HISTORIA=v,NOMBRE=f'LSTM 64 · {intermedia} · D{dropout} · {v}s',
                TIPO='red',CONFIG=cfg.copy(),MATRIZ=False))
    for codigo,unidades in [('D32x2',[32,32])]:
        for v in HISTORIAS:
            cfg=dict(unidades=unidades,dense=16,dropout=0.)
            nombre='→'.join(map(str,unidades))
            filas.append(dict(ID=f'lstm_adicional_{codigo}_v{v}',FAMILIA='LSTM',RECETA=codigo,
                HISTORIA=v,NOMBRE=f'LSTM {nombre} · Dense16 · D0 · {v}s',TIPO='red',CONFIG=cfg,MATRIZ=False))
    for v in HISTORIAS:
        filas.append(dict(ID=f'lstm_compacta16_8_v{v}',FAMILIA='LSTM',RECETA='compacta16_8',
            HISTORIA=v,NOMBRE=f'LSTM 16 · Dense8 · D0 · {v}s',TIPO='red',
            CONFIG=dict(unidades=[16],dense=8,dropout=0.),MATRIZ=False))
    for v in HISTORIAS:
        filas.append(dict(ID=f'local_v{v}',FAMILIA='LSTM local',RECETA='referencia local',
            HISTORIA=v,NOMBRE=f'LSTM local (5 redes) · {v}s',TIPO='local',
            CONFIG=dict(unidades=[32],dense=16,dropout=0.),MATRIZ=False))
    for receta,unidades,dense in [('compacta16_8',16,8),('compacta8_4',8,4)]:
        for v in HISTORIAS:
            filas.append(dict(ID=f'local_{receta}_v{v}',FAMILIA='LSTM local',RECETA=receta,
                HISTORIA=v,NOMBRE=f'LSTM local {unidades} · Dense{dense} (5 redes) · {v}s',TIPO='local',
                CONFIG=dict(unidades=[unidades],dense=dense,dropout=0.),MATRIZ=False))
    return filas


def nueva_ejecucion(preprocesamiento, columnas, catalogo):
    ruta=Path('ResultadosModelos/entrenamiento_en_notebook')/f'ejecucion_{uuid4().hex[:8]}'
    for sub in ['modelos','tablas','figuras']:
        (ruta/sub).mkdir(parents=True,exist_ok=False)
    config=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),historias=HISTORIAS,horizonte=HORIZONTE,
        semilla=42,criterio_seleccion='MAE de validación en casos originales',epochs_max=EPOCAS_MAX,
        batch_size=32,learning_rate=.001,early_stopping=PACIENCIA_ES,reduce_lr=5,
        recetas=RECETAS,recetas_64=RECETAS_64,xgboost=CONFIG_XGB,redes=catalogo,columnas=columnas,
        entrenamientos_anteriores_reutilizados=False,evaluacion_activada=False)
    (ruta/'configuracion.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    joblib.dump(preprocesamiento,ruta/'preprocesamiento_actual.joblib',compress=3)
    shutil.copy2(Path(__file__),ruta/'modelado_notebook.py')
    return ruta


def datos_municipales(datos,codigo):
    pos=np.flatnonzero(datos['meta']['COD_DANE'].eq(codigo))
    return {k:(valor.iloc[pos].reset_index(drop=True) if k=='meta' else valor[pos])
        for k,valor in datos.items()}


def crear_red(spec,n_variables,n_municipios):
    cfg=spec['CONFIG'];local=spec['TIPO']=='local'
    historia=tf.keras.Input(shape=(spec['HISTORIA'],n_variables),name='historia')
    z=historia
    for i,unidades in enumerate(cfg['unidades']):
        z=tf.keras.layers.LSTM(unidades,return_sequences=i<len(cfg['unidades'])-1,
            dropout=0.,recurrent_dropout=0.,name=f'recurrente_{i+1}')(z)
        if cfg['dropout']:
            z=tf.keras.layers.Dropout(cfg['dropout'],seed=42+i,name=f'dropout_{i+1}')(z)
    entradas={'historia':historia}
    if not local:
        municipio=tf.keras.Input(shape=(n_municipios,),name='municipio')
        entradas['municipio']=municipio
        z=tf.keras.layers.Concatenate(name='historia_y_municipio')([z,municipio])
    if cfg['dense']:
        z=tf.keras.layers.Dense(cfg['dense'],activation='relu',name='dense_intermedia')(z)
    salida=tf.keras.layers.Dense(1,activation='linear',name='casos_std_t4')(z)
    return tf.keras.Model(entradas,salida,name=spec['ID'])


def dataset_red(datos,local=False,solo_entradas=False,batch_size=32):
    entradas={'historia':datos['X']}
    if not local: entradas['municipio']=datos['aux']
    contenido=entradas if solo_entradas else (entradas,datos['y'])
    opciones=tf.data.Options();opciones.experimental_deterministic=True
    opciones.threading.private_threadpool_size=1;opciones.threading.max_intra_op_parallelism=1
    return tf.data.Dataset.from_tensor_slices(contenido).batch(batch_size).with_options(opciones)


def predecir_red_actual(modelo,datos,local=False):
    return modelo.predict(dataset_red(datos,local=local,solo_entradas=True,batch_size=256),verbose=0).astype(float)


def tabla_predicciones(spec,datos,predicciones,prep,conjunto):
    meta=datos['meta']
    t=meta.copy()
    t['ID']=spec['ID'];t['MODELO']=spec['NOMBRE'];t['FAMILIA']=spec['FAMILIA']
    t['RECETA']=spec['RECETA'];t['HISTORIA']=spec['HISTORIA'];t['CONJUNTO']=conjunto
    t['CASOS_REALES']=invertir(datos['y'],meta['COD_DANE'],prep)
    t['CASOS_PREDICHOS']=invertir(predicciones,meta['COD_DANE'],prep)
    t['ERROR']=t['CASOS_PREDICHOS']-t['CASOS_REALES']
    np.testing.assert_allclose(t['CASOS_REALES'],t['CASOS_ORIGINALES'],rtol=1e-5,atol=1e-3)
    assert np.isfinite(t[['CASOS_REALES','CASOS_PREDICHOS']].to_numpy()).all()
    return t


def resumir_predicciones(predicciones,catalogo):
    general=calcular_metricas(predicciones,['ID','MODELO','FAMILIA','RECETA','HISTORIA','CONJUNTO'])
    municipal=calcular_metricas(predicciones,['ID','MODELO','FAMILIA','HISTORIA','CONJUNTO','COD_DANE','MUNICIPIO'])
    macro=municipal.groupby(['ID','CONJUNTO'],observed=True).agg(
        MAE_PROMEDIO_MUNICIPAL=('MAE','mean'),RMSE_PROMEDIO_MUNICIPAL=('RMSE','mean')).reset_index()
    general=general.merge(macro,on=['ID','CONJUNTO'],validate='one_to_one')
    general['PARAMETROS_RED']=general['ID'].map({s['ID']:s.get('PARAMETROS_RED') for s in catalogo})
    ranking=general.loc[general['CONJUNTO'].eq('validation')].sort_values(['MAE','RMSE','ID']).reset_index(drop=True)
    return general,municipal,ranking


def graficar_baselines(predicciones,ranking,ruta):
    fig,ejes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    colores=[plt.get_cmap('viridis')(i) for i in np.linspace(.15,.85,len(HISTORIAS))]
    for eje,metrica in zip(ejes,['MAE','RMSE']):
        ranking.pivot(index='FAMILIA',columns='HISTORIA',values=metrica).plot.barh(ax=eje,color=colores)
        eje.set(title=metrica,xlabel=f'{metrica} (casos)',ylabel='');eje.grid(axis='x',alpha=.2)
    handles,labels=ejes[0].get_legend_handles_labels()
    for eje in ejes:eje.get_legend().remove()
    fig.legend(handles,labels,title='Historia (semanas)',loc='outside left center')
    fig.suptitle('Baselines entrenados en esta ejecución · Solo validación · t+4')
    for ext in ['png','svg']: fig.savefig(Path(ruta)/f'figuras/baselines_validacion.{ext}',dpi=200,bbox_inches='tight')
    plt.show()
    elegidos=ranking.groupby('FAMILIA',sort=False).head(1)['ID'].tolist()
    val=predicciones.loc[predicciones['CONJUNTO'].eq('validation')]
    fig,ejes=plt.subplots(5,1,figsize=(13,12),constrained_layout=True)
    for eje,(_,g) in zip(ejes,val.groupby('COD_DANE',sort=True,observed=True)):
        real=g.loc[g['ID'].eq(elegidos[0])].sort_values('FECHA_OBJETIVO')
        eje.plot(real['FECHA_OBJETIVO'],real['CASOS_ORIGINALES'],label='Observado',color='#252525')
        for identificador in elegidos:
            p=g.loc[g['ID'].eq(identificador)].sort_values('FECHA_OBJETIVO')
            eje.plot(p['FECHA_OBJETIVO'],p['CASOS_PREDICHOS'],label=p['MODELO'].iloc[0],linewidth=1)
        loc=mdates.AutoDateLocator(minticks=5,maxticks=8)
        eje.xaxis.set_major_locator(loc);eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        eje.set(title=g['MUNICIPIO'].iloc[0],ylabel='Casos',xlabel='Semana objetivo t+4');eje.grid(alpha=.2)
    h,l=ejes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('Baselines · Mejor MAE de validación de cada familia · Mismos modelos para los cinco municipios')
    for ext in ['png','svg']: fig.savefig(Path(ruta)/f'figuras/series_baselines_validacion.{ext}',dpi=200,bbox_inches='tight')
    plt.show()


def exportar_resultados_actuales(ruta,pred,general,municipal,ranking,catalogo,seleccion=None):
    ruta=Path(ruta)
    for conjunto in ['training','validation']:
        pred.loc[pred['CONJUNTO'].eq(conjunto)].to_csv(ruta/'tablas'/f'predicciones_{conjunto}.csv',
            index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    for nombre,t in [('metricas_desarrollo',general),('metricas_municipales',municipal),('ranking_validacion',ranking)]:
        t.to_csv(ruta/'tablas'/f'{nombre}.csv',index=False,encoding='utf-8-sig')
    (ruta/'catalogo_modelos.json').write_text(json.dumps(catalogo,ensure_ascii=False,indent=2),encoding='utf-8')
    if seleccion is not None:
        (ruta/'seleccion_validacion.json').write_text(json.dumps(seleccion,ensure_ascii=False,indent=2,
            default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    (ruta/'README.md').write_text(
        '# Entrenamiento realizado en ProyectoFinal.ipynb\n\n'
        f'Esta ejecución contiene {len(catalogo)} enfoques entrenados desde cero en el notebook. '
        'Las secuencias y los escaladores proceden de sus puntos 12 y 13; no se recuperaron resultados de otras corridas. '
        'Cada resultado se calcula después del ajuste: los baselines se entrenan en el punto 14 y las redes en el 17. '
        'La comparación de desarrollo utiliza solo entrenamiento y validación. El test se consulta en el bloque final del notebook.\n\n'
        'Los modelos están en modelos/, las métricas y predicciones en tablas/ y las gráficas en figuras/. '
        'Las rutas del catálogo corresponden únicamente a esta ejecución.\n',encoding='utf-8')


def invertir(z,codigos,prep):
    z=np.asarray(z,dtype=float).reshape(-1,1);codigos=pd.Series(np.asarray(codigos).astype(str))
    salida=np.empty(len(z))
    for m,pos in codigos.groupby(codigos).indices.items():
        salida[pos]=np.expm1(prep['escaladores_casos'][m].inverse_transform(z[pos])).ravel()
    assert np.isfinite(salida).all()
    return salida


def inferir_red(modelo,X,aux=None):
    partes=[]
    for i in range(0,len(X),32):
        entrada={'historia':X[i:i+32]}
        if aux is not None:entrada['municipio']=aux[i:i+32]
        partes.append(modelo(entrada,training=False).numpy())
    return np.concatenate(partes).astype(float)


def predecir_spec(spec,d):
    if spec['TIPO']=='red':
        m=tf.keras.models.load_model(spec['ARCHIVO'],compile=False)
        pred=inferir_red(m,d['X'],d['aux']);parametros=m.count_params()
        del m;tf.keras.backend.clear_session()
    elif spec['TIPO']=='local':
        pred=np.empty((len(d['X']),1));parametros=0
        for codigo,archivo in spec['ARCHIVO'].items():
            pos=np.flatnonzero(d['meta']['COD_DANE'].eq(codigo))
            m=tf.keras.models.load_model(archivo,compile=False)
            pred[pos]=inferir_red(m,d['X'][pos]);parametros+=m.count_params()
            del m;tf.keras.backend.clear_session()
    elif spec['TIPO']=='xgboost':
        m=XGBRegressor();m.load_model(spec['ARCHIVO'])
        pred=m.predict(d['matriz'],iteration_range=(0,m.best_iteration+1)).reshape(-1,1).astype(float);parametros=None
    else:
        m=joblib.load(spec['ARCHIVO']);pred=m.predict(d['matriz']).reshape(-1,1).astype(float);parametros=None
    assert pred.shape==(len(d['X']),1) and np.isfinite(pred).all()
    return pred,parametros


def calcular_metricas(tabla,grupos):
    filas=[]
    for claves,g in tabla.groupby(grupos,observed=True,sort=False):
        claves=claves if isinstance(claves,tuple) else (claves,)
        e=g['CASOS_PREDICHOS'].to_numpy()-g['CASOS_REALES'].to_numpy()
        filas.append({**dict(zip(grupos,claves)),'N':len(g),'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),
            'SESGO_MEDIO':float(e.mean()),'FRACCION_SUBESTIMACION':float(np.mean(e < -1e-3)),
            'PRONOSTICOS_NEGATIVOS':int(g['CASOS_PREDICHOS'].lt(0).sum())})
    return pd.DataFrame(filas)


def resumir_validacion(ranking):
    filas=[]
    for (familia,v),g in ranking.groupby(['FAMILIA','HISTORIA'],sort=False):
        fila={'FAMILIA':familia,'HISTORIA':v,'CONFIGURACIONES':len(g)}
        for metrica in ['MAE','RMSE']:
            r=g.loc[g[metrica].idxmin()]
            fila['MEJOR_'+metrica]=r['MODELO'];fila['MIN_'+metrica]=r[metrica]
        filas.append(fila)
    return pd.DataFrame(filas)


def graficar_desarrollo(pred,general,municipal,ranking,catalogo,ruta,mostrar=True):
    """Figuras de los entrenamientos y de validación, con historias variables."""
    rutas=[]
    def guardar(fig,nombre):
        for ext in ['png','svg']:
            fig.savefig(Path(ruta)/'figuras'/f'{nombre}.{ext}',dpi=220,bbox_inches='tight')
        rutas.append(nombre)
        if mostrar:plt.show()
        else:plt.close(fig)
    colores_historia={v:plt.get_cmap('viridis')(i) for v,i in zip(HISTORIAS,np.linspace(.15,.85,len(HISTORIAS)))}
    colores_familia={'LSTM':'#8D4CA6','LSTM local':'#238A76','Regresión lineal':'#176B93',
        'Random Forest':'#B45309','XGBoost':'#BE4256'}
    for v in HISTORIAS:
        fig,ejes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
        for eje,receta in zip(ejes.flat,RECETAS):
            spec=next(s for s in catalogo if s['ID']==f'lstm_{receta}_v{v}')
            h=pd.read_csv(spec['HISTORIAL'],encoding='utf-8-sig')
            eje.plot(h.EPOCA,h.loss,label='Entrenamiento',color='#176B93')
            eje.plot(h.EPOCA,h.val_loss,label='Validación',color='#B45309')
            eje.axvline(h.loc[h.val_loss.idxmin(),'EPOCA'],linestyle='--',color='#777777',label='Mejor época')
            eje.set(title=RECETAS[receta]['descripcion'],xlabel='Época',ylabel='MSE transformado')
            eje.grid(alpha=.2)
        ejes.flat[0].legend(fontsize=8)
        fig.suptitle(f'LSTM global · {v} semanas de historia · Entrenamiento y validación')
        guardar(fig,f'curvas_lstm_{v}')

    fig,ejes=plt.subplots(1,len(HISTORIAS),figsize=(20,4.5),constrained_layout=True)
    for eje,v in zip(np.atleast_1d(ejes),HISTORIAS):
        for receta,color,nombre in [('referencia','#8D4CA6','LSTM32 + Dense16'),('compacta16_8','#238A76','LSTM16 + Dense8')]:
            spec=next(s for s in catalogo if s['ID']==f'lstm_{receta}_v{v}')
            h=pd.read_csv(spec['HISTORIAL'],encoding='utf-8-sig')
            eje.plot(h.EPOCA,h.loss,color=color,label=f'{nombre} · train')
            eje.plot(h.EPOCA,h.val_loss,color=color,linestyle='--',label=f'{nombre} · validación')
        eje.set(title=f'{v} semanas',xlabel='Época',ylabel='MSE transformado')
        eje.grid(alpha=.2)
    h,l=np.atleast_1d(ejes)[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('LSTM16 + Dense8 frente a LSTM32 + Dense16 · Sin Dropout · t+4')
    guardar(fig,'11_lstm_16_dense8')

    fig,ejes=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
    g=ranking.loc[ranking.FAMILIA.eq('LSTM')&ranking.RECETA.isin(RECETAS)]
    for eje,metrica in zip(ejes,['MAE','RMSE']):
        tabla=g.pivot(index='RECETA',columns='HISTORIA',values=metrica).reindex(list(RECETAS))
        tabla.index=[RECETAS[r]['descripcion'] for r in tabla.index]
        tabla.plot.barh(ax=eje,color=list(colores_historia.values()))
        eje.set(title=metrica,xlabel=f'{metrica} (casos)',ylabel='')
        eje.grid(axis='x',alpha=.2)
        eje.get_legend().remove()
    h,l=ejes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=len(HISTORIAS),title='Historia (semanas)')
    fig.suptitle('LSTM globales · Seis estructuras y cinco historias · Solo validación')
    guardar(fig,'07_matriz_arquitecturas_historias')

    fig,ejes=plt.subplots(1,2,figsize=(16,9),constrained_layout=True)
    g=ranking.head(15).iloc[::-1]
    for eje,metrica in zip(ejes,['MAE','RMSE']):
        barras=eje.barh(g.MODELO,g[metrica],color=[colores_familia[f] for f in g.FAMILIA])
        eje.bar_label(barras,fmt='%.2f',padding=4)
        eje.set(title=metrica,xlabel=f'{metrica} (casos)',xlim=(0,g[metrica].max()*1.2))
        eje.grid(axis='x',alpha=.2)
    fig.suptitle('Quince menores MAE de validación · Cada fila conserva su RMSE')
    guardar(fig,'08_ranking_validacion_top15')

    mejores=ranking.groupby('FAMILIA',sort=False).head(1)
    ids=mejores.ID.tolist()
    val=pred.loc[pred.CONJUNTO.eq('validation')]
    fig,ejes=plt.subplots(5,1,figsize=(14,13),constrained_layout=True)
    for eje,(_,g) in zip(ejes,val.groupby('COD_DANE',observed=True,sort=True)):
        reales=g.loc[g.ID.eq(ids[0])].sort_values('FECHA_OBJETIVO')
        eje.plot(reales.FECHA_OBJETIVO,reales.CASOS_ORIGINALES,color='#252525',label='Observado',linewidth=1.5)
        for r in mejores.itertuples():
            p=g.loc[g.ID.eq(r.ID)].sort_values('FECHA_OBJETIVO')
            eje.plot(p.FECHA_OBJETIVO,p.CASOS_PREDICHOS,label=r.MODELO,color=colores_familia[r.FAMILIA],linewidth=1.1)
        loc=mdates.AutoDateLocator(minticks=4,maxticks=7)
        eje.xaxis.set_major_locator(loc)
        eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        eje.set(title=reales.MUNICIPIO.iloc[0],ylabel='Casos',xlabel='Semana objetivo t+4')
        eje.grid(alpha=.2)
    h,l=ejes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('Solo validación · Menor MAE general por familia · Mismos modelos para los cinco municipios')
    guardar(fig,'09_series_solo_validacion')

    fig,ejes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    g=ranking.loc[ranking.FAMILIA.isin(['Regresión lineal','Random Forest','XGBoost'])]
    for eje,metrica in zip(ejes,['MAE','RMSE']):
        g.pivot(index='FAMILIA',columns='HISTORIA',values=metrica).plot.barh(ax=eje,color=list(colores_historia.values()))
        eje.set(title=metrica,xlabel=f'{metrica} (casos)',ylabel='')
        eje.grid(axis='x',alpha=.2)
        eje.get_legend().remove()
    h,l=ejes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='outside lower center',ncol=len(HISTORIAS),title='Historia (semanas)')
    fig.suptitle('Baselines · Cinco historias · Solo validación · t+4')
    guardar(fig,'10_baselines_historias_validacion')
    return rutas


def evaluar_al_cierre(ruta,datos_desarrollo,prep,catalogo,ranking,seleccion):
    """Solo llamar tras activar explícitamente el bloque FINAL del notebook.

    El test ya es conocido históricamente. No es una nueva prueba ciega.
    """
    ruta=Path(ruta)
    ids=[seleccion['ID_ELEGIDO']]
    for familia in ['LSTM','LSTM local','Regresión lineal','Random Forest','XGBoost']:
        g=ranking.loc[ranking['FAMILIA'].eq(familia)]
        ids.append(g.loc[g['MAE'].idxmin(),'ID'])
    ids=list(dict.fromkeys(ids))
    salida=ruta/'evaluacion_al_cierre'/f'ejecucion_{uuid4().hex[:8]}'
    salida.mkdir(parents=True,exist_ok=False)
    specs=[next(r for r in catalogo if r['ID']==i) for i in ids]
    for spec in specs:
        archivos=spec['ARCHIVO'].values() if spec['TIPO']=='local' else [spec['ARCHIVO']]
        for archivo in archivos: Path(archivo).resolve().relative_to(ruta.resolve())
    # Registrar candidatos antes de abrir datos de evaluación.
    (salida/'seleccion_previa.json').write_text(json.dumps({'IDs':ids,'ID_ELEGIDO':seleccion['ID_ELEGIDO'],
        'criterio':seleccion.get('criterio_principal','MAE de validación'),'motivo_eleccion':seleccion.get('motivo_eleccion'),
        'referencias':'Menor MAE de validación por familia','test_historicamente_conocido':True},ensure_ascii=False,indent=2),encoding='utf-8')
    semanas=pd.concat([pd.read_csv(f'data/DatosProcesados/semanal_{c}.csv',dtype={'COD_DANE':'string'},parse_dates=['FECHA_INICIO','FECHA_FIN'],encoding='utf-8-sig').assign(CONJUNTO=c) for c in ['training','validation','test']],ignore_index=True)
    calendario=semanas[['ANO','SEMANA_EPI']].drop_duplicates().groupby('ANO').size().to_dict()
    for m,g in semanas.groupby('COD_DANE',observed=True):
        semanas.loc[g.index,'CASOS_std']=prep['escaladores_casos'][m].transform(np.log1p(g[['CASOS']].to_numpy(float))).ravel()
    temps=['TEMP_MEDIA_C','TEMP_MAX_C','TEMP_MIN_C']
    semanas[[c+'_std' for c in temps]]=prep['escaladores_clima']['temperaturas'].transform(semanas[temps].to_numpy(float))
    semanas['PRECIPITACION_MM_std']=prep['escaladores_clima']['precipitacion'].transform(np.log1p(semanas[['PRECIPITACION_MM']].to_numpy(float))).ravel()
    fase=2*np.pi*(semanas['SEMANA_EPI'].to_numpy(float)-1)/semanas['ANO'].map(calendario).to_numpy(float)
    semanas['semana_sin'],semanas['semana_cos']=np.sin(fase),np.cos(fase)
    corte=semanas.loc[semanas['CONJUNTO'].eq('validation'),'FECHA_FIN'].max()
    lookup=semanas.set_index(['COD_DANE','FECHA_INICIO'])
    test=semanas.loc[semanas['CONJUNTO'].eq('test')].sort_values(['COD_DANE','FECHA_INICIO'])
    filas=[]
    for r in test.itertuples():
        ultima=r.FECHA_INICIO-pd.Timedelta(weeks=4);emision=ultima+pd.Timedelta(days=6)
        if emision<corte:continue
        filas.append({'FILA_TENSOR':len(filas),'COD_DANE':r.COD_DANE,'MUNICIPIO':r.MUNICIPIO,'FECHA_ULTIMA_ENTRADA':ultima,
            'FECHA_EMISION':emision,'FECHA_OBJETIVO':r.FECHA_INICIO,'CASOS_ORIGINALES':int(r.CASOS)})
    meta=pd.DataFrame(filas);assert len(meta)==510
    tablas=[]
    for spec in specs:
        v=spec['HISTORIA'];X=np.empty((len(meta),v,7),np.float32);y=np.empty((len(meta),1),np.float32)
        for i,r in enumerate(meta.itertuples()):
            fechas=pd.date_range(end=r.FECHA_ULTIMA_ENTRADA,periods=v,freq='7D')
            X[i]=lookup.reindex(pd.MultiIndex.from_product([[r.COD_DANE],fechas]))[prep['columnas_temporales']].to_numpy(np.float32)
            y[i,0]=np.float32(lookup.loc[(r.COD_DANE,r.FECHA_OBJETIVO),'CASOS_std'])
        assert np.isfinite(X).all()
        aux=prep['codificador_municipios'].transform(meta[['COD_DANE']])
        if hasattr(aux,'toarray'):aux=aux.toarray()
        d={'X':X,'y':y,'meta':meta,'aux':aux,'matriz':np.concatenate([X.reshape(len(X),-1).astype(float),aux],axis=1)}
        z,_=predecir_spec(spec,d)
        t=meta.copy();t['ID']=spec['ID'];t['MODELO']=spec['NOMBRE'];t['CONJUNTO']='evaluation';t['HISTORIA']=v;t['HORIZONTE']=4
        t['CASOS_REALES']=invertir(y,meta['COD_DANE'],prep);t['CASOS_PREDICHOS']=invertir(z,meta['COD_DANE'],prep)
        t['ERROR']=t['CASOS_PREDICHOS']-t['CASOS_REALES'];tablas.append(t)
    pred=pd.concat(tablas,ignore_index=True)
    metricas=calcular_metricas(pred,['ID','MODELO','HISTORIA']);municipal=calcular_metricas(pred,['ID','MODELO','HISTORIA','COD_DANE','MUNICIPIO'])
    pred.to_csv(salida/'predicciones.csv',index=False,encoding='utf-8-sig');metricas.to_csv(salida/'metricas.csv',index=False,encoding='utf-8-sig');municipal.to_csv(salida/'metricas_municipales.csv',index=False,encoding='utf-8-sig')
    return pred,metricas,municipal,None,salida


def graficar_al_cierre(pred_desarrollo,pred_evaluacion,salida):
    """Figuras finales solo después de activar y realizar la evaluación."""
    ids=pred_evaluacion['ID'].unique()
    pred=pd.concat([pred_desarrollo.loc[pred_desarrollo['ID'].isin(ids)],pred_evaluacion],ignore_index=True)
    metricas=calcular_metricas(pred,['ID','MODELO','HISTORIA','CONJUNTO'])
    etiquetas={'training':'Entrenamiento · ajuste','validation':'Validación · selección','evaluation':'Evaluación al cierre · test históricamente conocido'}
    colores={identificador:plt.get_cmap('tab10')(i%10) for i,identificador in enumerate(ids)}
    archivos=[]
    for codigo,g in pred.groupby('COD_DANE',observed=True,sort=True):
        fig,ejes=plt.subplots(3,1,figsize=(14,10),constrained_layout=True)
        for eje,split in zip(ejes,['training','validation','evaluation']):
            periodo=g.loc[g['CONJUNTO'].eq(split)]
            real=periodo.loc[periodo['ID'].eq(ids[0])].sort_values('FECHA_OBJETIVO')
            eje.plot(real['FECHA_OBJETIVO'],real['CASOS_ORIGINALES'],label='Observado',color='#252525',linewidth=1.6)
            for ident in ids:
                p=periodo.loc[periodo['ID'].eq(ident)].sort_values('FECHA_OBJETIVO')
                eje.plot(p['FECHA_OBJETIVO'],p['CASOS_PREDICHOS'],label=p['MODELO'].iloc[0],color=colores[ident],linewidth=1)
            loc=mdates.AutoDateLocator(minticks=5,maxticks=8);eje.xaxis.set_major_locator(loc);eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            eje.set(title=etiquetas[split],ylabel='Casos',xlabel='Inicio de la semana objetivo t+4');eje.grid(alpha=.2)
        h,l=ejes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
        fig.suptitle(f'{g["MUNICIPIO"].iloc[0]} · Comparación al cierre · Escalas independientes por panel')
        for ext in ['png','svg']:fig.savefig(Path(salida)/f'series_{codigo}.{ext}',dpi=300,bbox_inches='tight')
        archivos.append(f'series_{codigo}');plt.show()
    metricas.to_csv(Path(salida)/'metricas_tres_conjuntos.csv',index=False,encoding='utf-8-sig')
    return metricas,archivos
