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

HISTORIAS = (12, 16, 20)
HORIZONTE = 4
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
    """44 enfoques recurrentes, entrenados mediante 56 redes individuales."""
    filas=[]
    for familia in ['LSTM','GRU']:
        for v in HISTORIAS:
            for receta,cfg in RECETAS.items():
                filas.append(dict(ID=f'{familia.lower()}_{receta}_v{v}',FAMILIA=familia,
                    RECETA=receta,HISTORIA=v,NOMBRE=f'{familia} {cfg["descripcion"]} · {v}s',
                    TIPO='red',CONFIG=cfg.copy(),MATRIZ=True))
    for codigo,unidades in [('B64',[64]),('D32x2',[32,32])]:
        cfg=dict(unidades=unidades,dense=16,dropout=0.)
        nombre='→'.join(map(str,unidades))
        filas.append(dict(ID=f'lstm_adicional_{codigo}_v16',FAMILIA='LSTM',RECETA=codigo,
            HISTORIA=16,NOMBRE=f'LSTM {nombre} · Dense16 · D0 · 16s',TIPO='red',CONFIG=cfg,MATRIZ=False))
    for v in HISTORIAS:
        filas.append(dict(ID=f'lstm_compacta16_8_v{v}',FAMILIA='LSTM',RECETA='compacta16_8',
            HISTORIA=v,NOMBRE=f'LSTM 16 · Dense8 · D0 · {v}s',TIPO='red',
            CONFIG=dict(unidades=[16],dense=8,dropout=0.),MATRIZ=False))
    for v in HISTORIAS:
        filas.append(dict(ID=f'local_v{v}',FAMILIA='LSTM local',RECETA='referencia local',
            HISTORIA=v,NOMBRE=f'LSTM local (5 redes) · {v}s',TIPO='local',
            CONFIG=dict(unidades=[32],dense=16,dropout=0.),MATRIZ=False))
    return filas


def nueva_ejecucion(preprocesamiento, columnas, catalogo):
    ruta=Path('ResultadosModelos/entrenamiento_en_notebook')/f'ejecucion_{uuid4().hex[:8]}'
    for sub in ['modelos','tablas','figuras']:
        (ruta/sub).mkdir(parents=True,exist_ok=False)
    config=dict(fecha_utc=datetime.now(timezone.utc).isoformat(),historias=HISTORIAS,horizonte=HORIZONTE,
        semilla=42,criterio_seleccion='MAE de validación en casos originales',epochs_max=100,
        batch_size=32,learning_rate=.001,early_stopping=10,reduce_lr=5,
        recetas=RECETAS,xgboost=CONFIG_XGB,redes=catalogo,columnas=columnas,
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
        Capa=tf.keras.layers.GRU if spec['FAMILIA']=='GRU' else tf.keras.layers.LSTM
        z=Capa(unidades,return_sequences=i<len(cfg['unidades'])-1,
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
    colores=['#8398A6','#176B93','#D97628']
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
        'La comparación utiliza solo entrenamiento y validación. La evaluación final permanece desactivada.\n\n'
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
    rutas=[]
    def guardar(fig,nombre):
        for ext in ['png','svg']:fig.savefig(Path(ruta)/'figuras'/f'{nombre}.{ext}',dpi=300,bbox_inches='tight')
        rutas.append(nombre)
        if mostrar:plt.show()
        else:plt.close(fig)
    colores_historia={12:'#8398A6',16:'#176B93',20:'#D97628'}
    color_familia={'LSTM':'#8D4CA6','GRU':'#D97628','Regresión lineal':'#176B93','Random Forest':'#238A76','XGBoost':'#BE4256','LSTM local':'#788D9D'}
    with plt.rc_context({'font.size':9,'svg.fonttype':'none'}):
        for v in HISTORIAS:
            for familia in ['LSTM','GRU']:
                fig,ejes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
                for eje,receta in zip(ejes.flat,RECETAS):
                    spec=next(r for r in catalogo if r['HISTORIA']==v and r['FAMILIA']==familia and r['RECETA']==receta)
                    h=pd.read_csv(spec['HISTORIAL'],encoding='utf-8-sig')
                    mejor=int(h.loc[h['val_loss'].idxmin(),'EPOCA'])
                    eje.plot(h['EPOCA'],h['loss'],label='Entrenamiento',color='#176B93')
                    eje.plot(h['EPOCA'],h['val_loss'],label='Validación',color='#D97628')
                    eje.axvline(mejor,linestyle='--',color='#777777',label='Mejor época')
                    eje.set(title=RECETAS[receta]['descripcion'],xlabel='Época',ylabel='MSE de CASOS_std');eje.grid(alpha=.2)
                ejes.flat[0].legend(fontsize=8)
                fig.suptitle(f'{familia} · Historia de {v} semanas · Seis estructuras · Entrenamiento y validación')
                guardar(fig,f'curvas_{familia.lower()}_{v}')
        fig,ejes=plt.subplots(1,len(HISTORIAS),figsize=(15,4.5),constrained_layout=True)
        for eje,v in zip(np.atleast_1d(ejes),HISTORIAS):
            for receta,color,nombre in [('referencia','#8D4CA6','LSTM32 + Dense16'),
                                        ('compacta16_8','#238A76','LSTM16 + Dense8')]:
                spec=next(r for r in catalogo if r['ID']==f'lstm_{receta}_v{v}')
                h=pd.read_csv(spec['HISTORIAL'],encoding='utf-8-sig')
                eje.plot(h['EPOCA'],h['loss'],color=color,label=f'{nombre} · train')
                eje.plot(h['EPOCA'],h['val_loss'],color=color,linestyle='--',label=f'{nombre} · validación')
            eje.set(title=f'Historia de {v} semanas',xlabel='Época',ylabel='MSE de CASOS_std')
            eje.grid(alpha=.2)
        handles,labels=np.atleast_1d(ejes)[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
        fig.suptitle('LSTM16 + Dense8 frente a LSTM32 + Dense16 · Sin Dropout · Objetivo t+4')
        guardar(fig,'11_lstm_16_dense8')
        fig,ejes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
        for fila,familia in enumerate(['LSTM','GRU']):
            g=ranking.loc[ranking['FAMILIA'].eq(familia)&ranking['RECETA'].isin(RECETAS)]
            for eje,metrica in zip(ejes[fila],['MAE','RMSE']):
                tabla=g.pivot(index='RECETA',columns='HISTORIA',values=metrica).reindex(list(RECETAS))
                tabla.index=[RECETAS[r]['descripcion'] for r in tabla.index]
                tabla.rename(columns={v:f'{v} semanas' for v in HISTORIAS}).plot.barh(ax=eje,color=list(colores_historia.values()))
                eje.set(title=f'{familia} · {metrica}',xlabel=f'{metrica} (casos)',ylabel='');eje.grid(axis='x',alpha=.2)
        handles,labels=ejes[0,0].get_legend_handles_labels()
        for eje in ejes.flat:eje.get_legend().remove()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,title='Historia')
        fig.suptitle('36 combinaciones recurrentes · Misma cobertura · Validación · Objetivo t+4')
        guardar(fig,'07_matriz_arquitecturas_historias')
        fig,ejes=plt.subplots(1,2,figsize=(16,9),constrained_layout=True)
        g=ranking.head(15).iloc[::-1]
        for eje,metrica in zip(ejes,['MAE','RMSE']):
            b=eje.barh(g['MODELO'],g[metrica],color=[color_familia[f] for f in g['FAMILIA']])
            eje.bar_label(b,fmt='%.2f',padding=4);eje.set(title=metrica,xlabel=f'{metrica} (casos)',xlim=(0,g[metrica].max()*1.2))
            eje.grid(axis='x',alpha=.2)
        fig.suptitle('Quince menores MAE de validación · Cada fila conserva su RMSE · Selección de desarrollo')
        guardar(fig,'08_ranking_validacion_top15')
        elegidos=[]
        for familia in ['LSTM','GRU','Regresión lineal','Random Forest','XGBoost']:
            g=ranking.loc[ranking['FAMILIA'].eq(familia)]
            elegidos.append(g.loc[g['MAE'].idxmin(),'ID'])
        fig,ejes=plt.subplots(5,1,figsize=(14,13),constrained_layout=True)
        val=pred.loc[pred['CONJUNTO'].eq('validation')]
        for eje,(codigo,g) in zip(ejes,val.groupby('COD_DANE',observed=True,sort=True)):
            real=g.loc[g['ID'].eq(elegidos[0])].sort_values('FECHA_OBJETIVO')
            eje.plot(real['FECHA_OBJETIVO'],real['CASOS_ORIGINALES'],color='#252525',label='Observado',linewidth=1.5)
            for identificador in elegidos:
                p=g.loc[g['ID'].eq(identificador)].sort_values('FECHA_OBJETIVO')
                eje.plot(p['FECHA_OBJETIVO'],p['CASOS_PREDICHOS'],label=p['MODELO'].iloc[0],color=color_familia[p['FAMILIA'].iloc[0]],linewidth=1.05)
            loc=mdates.AutoDateLocator(minticks=5,maxticks=8);eje.xaxis.set_major_locator(loc);eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            eje.set(title=g['MUNICIPIO'].iloc[0],ylabel='Casos',xlabel='Inicio de la semana objetivo t+4');eje.grid(alpha=.2)
        h,l=ejes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2,fontsize=8)
        fig.suptitle('Solo validación · Menor MAE general de cada familia\nMismos modelos para todos los municipios; selección dentro de desarrollo')
        guardar(fig,'09_series_solo_validacion')
        fig,ejes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        base=ranking.loc[ranking['FAMILIA'].isin(['Regresión lineal','Random Forest','XGBoost'])]
        for eje,metrica in zip(ejes,['MAE','RMSE']):
            base.pivot(index='FAMILIA',columns='HISTORIA',values=metrica).rename(columns={v:f'{v} semanas' for v in HISTORIAS}).plot.barh(ax=eje,color=list(colores_historia.values()))
            eje.set(title=metrica,xlabel=f'{metrica} (casos)',ylabel='');eje.grid(axis='x',alpha=.2)
        h,l=ejes[0].get_legend_handles_labels()
        for eje in ejes:eje.get_legend().remove()
        fig.legend(h,l,loc='outside lower center',ncol=3)
        fig.suptitle('Baselines · Historias de 12, 16 y 20 semanas · Solo validación · t+4')
        guardar(fig,'10_baselines_historias_validacion')
    return rutas


def conclusiones_desarrollo(ranking,municipal,catalogo):
    mae=ranking.loc[ranking['MAE'].idxmin()];rmse=ranking.loc[ranking['RMSE'].idxmin()]
    referencias=[]
    for f in ['LSTM','GRU']:
        g=ranking.loc[ranking['FAMILIA'].eq(f)];r=g.loc[g['MAE'].idxmin()]
        referencias.append(f"{f}: {r['MODELO']} (MAE {r['MAE']:.3f}, RMSE {r['RMSE']:.3f})")
    lecturas=[]
    for f in ['LSTM','GRU']:
        for v in HISTORIAS:
            base=ranking.loc[ranking['ID'].eq(f'{f.lower()}_referencia_v{v}')].iloc[0]
            variantes=['sin_dense','drop20','drop50','dos_capas','tres_capas']
            if f=='LSTM':variantes.append('compacta16_8')
            for receta in variantes:
                nuevo=ranking.loc[ranking['ID'].eq(f'{f.lower()}_{receta}_v{v}')].iloc[0]
                cambio=100*(base['MAE']-nuevo['MAE'])/base['MAE']
                lecturas.append({'FAMILIA':f,'HISTORIA':v,'RECETA':receta,'MAE_REFERENCIA':base['MAE'],'MAE_VARIANTE':nuevo['MAE'],
                    'MEJORA_MAE_PCT':cambio,'CAMBIO_RMSE':nuevo['RMSE']-base['RMSE']})
    cambios=pd.DataFrame(lecturas)
    drop=cambios.loc[cambios['RECETA'].isin(['drop20','drop50'])]
    profundas=cambios.loc[cambios['RECETA'].isin(['dos_capas','tres_capas'])]
    simples=cambios.loc[cambios['RECETA'].eq('sin_dense')]
    compactas=cambios.loc[cambios['RECETA'].eq('compacta16_8')]
    detalle_compactas='; '.join(
        f"{int(r.HISTORIA)} semanas: MAE {r.MAE_VARIANTE:.3f} frente a {r.MAE_REFERENCIA:.3f} de la referencia"
        for r in compactas.sort_values('HISTORIA').itertuples())
    mus=municipal.loc[municipal['CONJUNTO'].eq('validation')&municipal['ID'].eq(mae['ID'])]
    menor=mus.loc[mus['MAE'].idxmin()];mayor=mus.loc[mus['MAE'].idxmax()]
    texto=f"""**Con qué modelo seguiríamos por ahora.** El menor MAE de las {len(ranking)} configuraciones corresponde a **{mae['MODELO']}**: **{mae['MAE']:.3f} casos**, con RMSE de {mae['RMSE']:.3f}. Elegimos provisionalmente ese modelo porque fijamos MAE como criterio principal. Si miramos solo RMSE, el mínimo es de **{rmse['MODELO']}** ({rmse['RMSE']:.3f}). Ambos errores están expresados en casos de la semana t+4.

**LSTM, GRU y la cantidad de historia.** {'; '.join(referencias)}. Todas las historias tuvieron los mismos {int(catalogo[0]['N_ENTRENAMIENTO']):,} ejemplos de entrenamiento y {int(mae['N'])} objetivos de validación. Por tanto, la comparación cambia la información disponible dentro de cada ventana, sin añadir más fechas de entrenamiento a una opción.

**Qué pasó al reducir la red a LSTM16 y Dense8.** {detalle_compactas}. La variante reducida obtuvo un menor MAE en {int(compactas['MEJORA_MAE_PCT'].gt(0).sum())} de las {len(compactas)} historias. Ambas estructuras conservan el one-hot municipal, la salida de una neurona y Dropout en cero. Cambiamos a la vez las unidades LSTM y las neuronas de la Dense intermedia: esta comparación evalúa la estructura completa y no permite separar el efecto de cada reducción. Tener menos parámetros no demuestra por sí solo que haya menos sobreajuste.

**Qué aportaron Dropout y las capas.** Añadir únicamente Dropout redujo el MAE frente a la referencia en **{int(drop['MEJORA_MAE_PCT'].gt(0).sum())} de las {len(drop)} variantes**. Las estructuras de dos o tres capas con Dropout lo redujeron en **{int(profundas['MEJORA_MAE_PCT'].gt(0).sum())} de {len(profundas)}**, y quitar la Dense intermedia, en **{int(simples['MEJORA_MAE_PCT'].gt(0).sum())} de {len(simples)}**. Cada variante se contrasta con su propia familia e historia. Las redes profundas cambian a la vez capas y Dropout: estos resultados no aíslan el efecto de cada cambio. La tabla siguiente permite revisar dónde mejoró o empeoró cada estructura; mejora porcentual positiva significa menor MAE y cambio de RMSE negativo significa menor RMSE.

**Cómo cambia el error entre municipios.** En el candidato provisional, el MAE municipal va de {menor['MAE']:.3f} en {menor['MUNICIPIO']} a {mayor['MAE']:.3f} en {mayor['MUNICIPIO']}. Son errores en cantidades de casos y los municipios tienen volúmenes distintos; esa diferencia no demuestra por sí sola que el modelo funcione proporcionalmente mejor en uno. Se conserva un mismo modelo global, elegido por el MAE general, para los cinco municipios.

**Hasta dónde podemos concluir.** Estas diferencias describen una ejecución con una semilla y el mismo bloque de validación 2022–2023. Probar más configuraciones puede producir un mínimo optimista; además, las ventanas se solapan. Todavía hacen falta otros cortes cronológicos o semillas y referencias sencillas de persistencia o estacionales para revisar estabilidad. No se ha demostrado superioridad estadística, causalidad ni utilidad de una intervención sanitaria.

**El cierre queda pendiente.** Se mantuvieron los escaladores ajustados en entrenamiento y las predicciones continuas, incluidos los valores negativos que las tablas contabilizan. No se volvió a leer el test durante esta exploración. Como el bloque 2024–2025 ya se había observado antes, usarlo posteriormente tampoco constituirá una prueba ciega. La evaluación y su análisis final permanecen desactivados en el último punto del notebook mientras decidimos con qué configuración continuar.
"""
    return texto,cambios


def evaluar_al_cierre(ruta,datos_desarrollo,prep,catalogo,ranking,seleccion):
    """Solo llamar tras activar explícitamente el bloque FINAL del notebook.

    El test ya es conocido históricamente. No es una nueva prueba ciega.
    """
    ruta=Path(ruta)
    ids=[seleccion['ID_ELEGIDO']]
    for familia in ['LSTM','GRU','LSTM local','Regresión lineal','Random Forest','XGBoost']:
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
    (salida/'seleccion_previa.json').write_text(json.dumps({'IDs':ids,'criterio':'MAE de validación','test_historicamente_conocido':True},ensure_ascii=False,indent=2),encoding='utf-8')
    semanas=pd.concat([pd.read_csv(f'DatosProcesados/semanal_{c}.csv',dtype={'COD_DANE':'string'},parse_dates=['FECHA_INICIO','FECHA_FIN'],encoding='utf-8-sig').assign(CONJUNTO=c) for c in ['training','validation','test']],ignore_index=True)
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
    elegido=metricas.loc[metricas['ID'].eq(seleccion['ID_ELEGIDO'])].iloc[0]
    mejor=metricas.loc[metricas['MAE'].idxmin()]
    texto=f"El candidato elegido por validación, {elegido['MODELO']}, obtiene MAE {elegido['MAE']:.3f} y RMSE {elegido['RMSE']:.3f} en el bloque de evaluación. Entre los candidatos congelados, el menor MAE observado es de {mejor['MODELO']} ({mejor['MAE']:.3f}). Estos resultados describen un test que ya se había observado; no constituyen una nueva confirmación independiente ni justifican cambiar retrospectivamente la selección. No se reentrenó ni ajustó ningún modelo con este bloque."
    (salida/'ANALISIS_FINAL.md').write_text('# Análisis al cierre: evaluación históricamente conocida\n\n'+texto,encoding='utf-8')
    return pred,metricas,municipal,texto,salida


def graficar_al_cierre(pred_desarrollo,pred_evaluacion,salida):
    """Figuras finales solo después de activar y realizar la evaluación."""
    ids=pred_evaluacion['ID'].unique()
    pred=pd.concat([pred_desarrollo.loc[pred_desarrollo['ID'].isin(ids)],pred_evaluacion],ignore_index=True)
    metricas=calcular_metricas(pred,['ID','MODELO','HISTORIA','CONJUNTO'])
    etiquetas={'training':'Entrenamiento · ajuste','validation':'Validación · selección','evaluation':'Evaluación al cierre · test históricamente conocido'}
    colores=dict(zip(ids,['#8D4CA6','#D97628','#176B93','#238A76','#BE4256','#657B8C']))
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
