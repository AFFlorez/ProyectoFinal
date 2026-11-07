"""Exploración exclusivamente con train/validation; evaluación explícita al cierre."""
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import gc
import hashlib
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HISTORIAS = (12,16,20)
ORIGEN = Path('ResultadosModelos/historia_12_16_20_t4/ejecucion_ba352a18')
ORIGEN_ARQ = Path('ResultadosModelos/arquitecturas_lstm_global_t4/ejecucion_1db6e884')
ORIGEN_GRU = Path('ResultadosModelos/gru_xgboost_t4/ejecucion_04dafbf5')
PREP = Path('ResultadosModelos/lstm_global/ejecucion_7f082bbd/preprocesamiento_existente.joblib')
CONFIG_ENTRENAMIENTO = {'semilla':42,'batch_size':32,'epochs_max':100,'learning_rate':.001,
    'paciencia_early_stopping':10,'paciencia_reduce_lr':5,'factor_reduce_lr':.5,'learning_rate_min':1e-5}
CONFIG_XGB = {'n_estimators':800,'max_depth':3,'learning_rate':.03,'min_child_weight':5,
    'subsample':.8,'colsample_bytree':.8,'reg_alpha':0.0,'reg_lambda':1.0,
    'objective':'reg:squarederror','eval_metric':'rmse','early_stopping_rounds':50,
    'random_state':42,'n_jobs':4,'tree_method':'hist','device':'cpu'}
RECETAS = {
    'referencia':{'unidades':[32],'dense':16,'dropout':0.0,'descripcion':'32 · Dense16 · D0'},
    'sin_dense':{'unidades':[32],'dense':0,'dropout':0.0,'descripcion':'32 · sin Dense intermedia'},
    'drop20':{'unidades':[32],'dense':16,'dropout':.2,'descripcion':'32 · Dense16 · D0,20'},
    'drop50':{'unidades':[32],'dense':16,'dropout':.5,'descripcion':'32 · Dense16 · D0,50'},
    'dos_capas':{'unidades':[32,16],'dense':16,'dropout':.2,'descripcion':'32→16 · Dense16 · D0,20'},
    'tres_capas':{'unidades':[32,16,8],'dense':16,'dropout':.2,'descripcion':'32→16→8 · Dense16 · D0,20'},
}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cargar_desarrollo():
    """Solo entrenamiento y validación: ni test ni sus tensores o predicciones."""
    prep=joblib.load(PREP)
    semanas=pd.concat([pd.read_csv(f'data/DatosProcesados/semanal_{c}.csv',dtype={'COD_DANE':'string'},
        parse_dates=['FECHA_INICIO','FECHA_FIN'],encoding='utf-8-sig').assign(CONJUNTO=c)
        for c in ['training','validation']],ignore_index=True)
    assert semanas['ANO'].between(2007,2023).all()
    assert not semanas.duplicated(['COD_DANE','FECHA_INICIO']).any()
    calendario=semanas[['ANO','SEMANA_EPI']].drop_duplicates().groupby('ANO').size().to_dict()
    for m,g in semanas.groupby('COD_DANE',observed=True):
        assert g['FECHA_INICIO'].diff().dropna().dt.days.eq(7).all()
        semanas.loc[g.index,'CASOS_std']=prep['escaladores_casos'][m].transform(np.log1p(g[['CASOS']].to_numpy(float))).ravel()
    temperaturas=['TEMP_MEDIA_C','TEMP_MAX_C','TEMP_MIN_C']
    semanas[[c+'_std' for c in temperaturas]]=prep['escaladores_clima']['temperaturas'].transform(semanas[temperaturas].to_numpy(float))
    semanas['PRECIPITACION_MM_std']=prep['escaladores_clima']['precipitacion'].transform(np.log1p(semanas[['PRECIPITACION_MM']].to_numpy(float))).ravel()
    fase=2*np.pi*(semanas['SEMANA_EPI'].to_numpy(float)-1)/semanas['ANO'].map(calendario).to_numpy(float)
    semanas['semana_sin'],semanas['semana_cos']=np.sin(fase),np.cos(fase)
    lookup=semanas.set_index(['COD_DANE','FECHA_INICIO'])
    datos={}
    for v in HISTORIAS:
        archivo=ORIGEN/f'ventana_{v}'/('insumos_t4.joblib' if v==20 else 'insumos_control.joblib')
        originales=joblib.load(archivo)
        datos[v]={}
        for nombre,sufijo in [('training','train'),('validation','val')]:
            X=originales[f'X_{sufijo}'].copy();y=originales[f'y_{sufijo}'].copy()
            meta=originales[f'metadata_{sufijo}'].copy(deep=True)
            meta['COD_DANE']=meta['COD_DANE'].astype('string')
            assert X.shape==(3800 if nombre=='training' else 505,v,7) and y.shape==(len(X),1)
            assert meta['CONJUNTO'].eq(nombre).all() and meta['HORIZONTE'].eq(4).all()
            raw=[]
            for i,r in enumerate(meta.itertuples()):
                fechas=pd.date_range(end=r.FECHA_ULTIMA_ENTRADA,periods=v,freq='7D')
                indice=pd.MultiIndex.from_product([[r.COD_DANE],fechas])
                np.testing.assert_array_equal(X[i],lookup.reindex(indice)[prep['columnas_temporales']].to_numpy(np.float32))
                objetivo=lookup.loc[(r.COD_DANE,r.FECHA_OBJETIVO)]
                np.testing.assert_equal(y[i,0],np.float32(objetivo['CASOS_std']))
                assert objetivo['CONJUNTO']==nombre
                assert r.FECHA_OBJETIVO==r.FECHA_ULTIMA_ENTRADA+pd.Timedelta(weeks=4)
                assert r.FECHA_EMISION==r.FECHA_ULTIMA_ENTRADA+pd.Timedelta(days=6)<r.FECHA_OBJETIVO
                raw.append(int(objetivo['CASOS']))
            meta['CASOS_ORIGINALES']=raw
            aux=prep['codificador_municipios'].transform(meta[['COD_DANE']])
            if hasattr(aux,'toarray'):aux=aux.toarray()
            datos[v][nombre]={'X':X,'y':y,'meta':meta,'aux':aux,
                'matriz':np.concatenate([X.reshape(len(X),-1).astype(float),aux],axis=1)}
            np.testing.assert_array_equal(y,datos[12][nombre]['y'])
            for c in ['COD_DANE','FECHA_OBJETIVO','FECHA_EMISION']:
                np.testing.assert_array_equal(meta[c],datos[12][nombre]['meta'][c])
            np.testing.assert_array_equal(X[:,-12:],datos[12][nombre]['X'])
    return datos,prep


def catalogo_modelos():
    filas=[]
    for v in HISTORIAS:
        for familia in ['LSTM','GRU']:
            for receta,cfg in RECETAS.items():
                origen=None;historial=None
                if familia=='LSTM' and receta=='referencia':
                    p=ORIGEN/f'ventana_{v}/modelos/lstm_global_t4_v{v}'
                    origen=p/'mejor_modelo.keras';historial=p/'historial.csv'
                elif familia=='LSTM' and receta=='drop20' and v==16:
                    origen=ORIGEN_ARQ/'C32_dropout/mejor_modelo.keras';historial=ORIGEN_ARQ/'C32_dropout/historial.csv'
                elif familia=='GRU' and receta=='referencia' and v==16:
                    origen=ORIGEN_GRU/'modelos/gru_global_t4.keras';historial=ORIGEN_GRU/'tablas/historial_gru.csv'
                filas.append({'ID':f'{familia.lower()}_{receta}_v{v}','FAMILIA':familia,'RECETA':receta,'HISTORIA':v,
                    'NOMBRE':f'{familia} {cfg["descripcion"]} · {v}s','TIPO':'red','CONFIG':cfg,
                    'ARCHIVO':str(origen) if origen else None,'HISTORIAL':str(historial) if historial else None,
                    'REUTILIZADO':origen is not None,'MATRIZ':True})
        for familia,archivo in [('Regresión lineal','regresion_lineal'),('Random Forest','random_forest'),('XGBoost','xgboost')]:
            origen=ORIGEN/f'ventana_{v}/modelos/{archivo}.joblib' if familia!='XGBoost' else (ORIGEN_GRU/'modelos/xgboost_t4.ubj' if v==16 else None)
            filas.append({'ID':f'{archivo}_v{v}','FAMILIA':familia,'RECETA':'baseline','HISTORIA':v,
                'NOMBRE':f'{familia} · {v}s','TIPO':'xgboost' if familia=='XGBoost' else 'sklearn',
                'CONFIG':CONFIG_XGB if familia=='XGBoost' else {},'ARCHIVO':str(origen) if origen else None,
                'HISTORIAL':str(ORIGEN_GRU/'tablas/historial_xgboost.csv') if familia=='XGBoost' and v==16 else None,
                'REUTILIZADO':origen is not None,'MATRIZ':False})
        archivos={m.name.rsplit('_',1)[-1]:str(m/'mejor_modelo.keras') for m in (ORIGEN/f'ventana_{v}/modelos').glob('lstm_local*')}
        assert len(archivos)==5
        filas.append({'ID':f'local_v{v}','FAMILIA':'LSTM local','RECETA':'referencia local','HISTORIA':v,
            'NOMBRE':f'LSTM local (5 redes) · {v}s','TIPO':'local','CONFIG':{},'ARCHIVO':archivos,
            'HISTORIAL':None,'REUTILIZADO':True,'MATRIZ':False})
    for codigo,carpeta,nombre,unidades in [('B64','B64','LSTM 64 · Dense16 · D0 · 16s',[64]),('D32x2','D32x2','LSTM 32→32 · Dense16 · D0 · 16s',[32,32])]:
        filas.append({'ID':f'lstm_previa_{codigo}_v16','FAMILIA':'LSTM','RECETA':'previa '+codigo,'HISTORIA':16,
            'NOMBRE':nombre,'TIPO':'red','CONFIG':{'unidades':unidades,'dense':16,'dropout':0},
            'ARCHIVO':str(ORIGEN_ARQ/carpeta/'mejor_modelo.keras'),'HISTORIAL':str(ORIGEN_ARQ/carpeta/'historial.csv'),
            'REUTILIZADO':True,'MATRIZ':False})
    assert len(filas)==50 and sum(not r['REUTILIZADO'] for r in filas)==33
    return filas


def iniciar_experimento():
    ruta=Path('ResultadosModelos/exploracion_temporal_validacion')/f'ejecucion_{uuid4().hex[:8]}'
    (ruta/'modelos').mkdir(parents=True,exist_ok=False);(ruta/'tablas').mkdir();(ruta/'figuras').mkdir()
    catalogo=catalogo_modelos()
    config={'fecha_utc':datetime.now(timezone.utc).isoformat(),'historias':list(HISTORIAS),'horizonte':4,
        'redes':CONFIG_ENTRENAMIENTO,'xgboost':CONFIG_XGB,'recetas':RECETAS,'criterio_principal':'MAE agrupado de validación en casos',
        'criterio_secundario':'RMSE de validación; se informa por separado','semillas':[42],
        'matriz_recurrente':36,'ajustes_nuevos_redes':31,'ajustes_nuevos_xgboost':2,
        'uso_test':False,'test_ya_observado_historicamente':True,'evaluacion_final_activada':False,
        'insumos_sha256':{str(PREP):sha(PREP),**{f'data/DatosProcesados/semanal_{c}.csv':sha(f'data/DatosProcesados/semanal_{c}.csv') for c in ['training','validation']}}}
    (ruta/'configuracion_previa.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (ruta/'catalogo_inicial.json').write_text(json.dumps(catalogo,ensure_ascii=False,indent=2),encoding='utf-8')
    return ruta,catalogo


def dataset(d):
    op=tf.data.Options();op.experimental_deterministic=True
    op.threading.private_threadpool_size=1;op.threading.max_intra_op_parallelism=1
    return tf.data.Dataset.from_tensor_slices(({'historia':d['X'],'municipio':d['aux']},d['y'])).batch(32).with_options(op)


def entrenar_red(spec,datos,ruta):
    tf.keras.backend.clear_session();gc.collect()
    tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
    cfg=spec['CONFIG'];v=spec['HISTORIA']
    historia=tf.keras.Input(shape=(v,7),name='historia');municipio=tf.keras.Input(shape=(5,),name='municipio')
    z=historia
    for i,u in enumerate(cfg['unidades']):
        Capa=tf.keras.layers.LSTM if spec['FAMILIA']=='LSTM' else tf.keras.layers.GRU
        z=Capa(u,return_sequences=i<len(cfg['unidades'])-1,dropout=0.0,recurrent_dropout=0.0,name=f'recurrente_{i+1}')(z)
        if cfg['dropout']:
            z=tf.keras.layers.Dropout(cfg['dropout'],seed=42+i,name=f'dropout_{i+1}')(z)
    z=tf.keras.layers.Concatenate(name='historia_y_municipio')([z,municipio])
    if cfg['dense']:z=tf.keras.layers.Dense(cfg['dense'],activation='relu',name='dense_intermedia')(z)
    salida=tf.keras.layers.Dense(1,activation='linear',name='casos_std_t4')(z)
    modelo=tf.keras.Model({'historia':historia,'municipio':municipio},salida,name=spec['ID'])
    modelo.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse',metrics=['mae'])
    carpeta=Path(ruta)/'modelos'/spec['ID'];carpeta.mkdir(exist_ok=True)
    archivo=carpeta/'mejor_modelo.keras'
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min',patience=5,factor=.5,min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(str(archivo),monitor='val_loss',mode='min',save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(carpeta/'progreso.csv')),tf.keras.callbacks.TerminateOnNaN()]
    h=modelo.fit(dataset(datos[v]['training']),validation_data=dataset(datos[v]['validation']),
        epochs=100,callbacks=callbacks,shuffle=False,verbose=0)
    h=pd.DataFrame(h.history);h.insert(0,'EPOCA',np.arange(1,len(h)+1))
    assert np.isfinite(h[['loss','val_loss']].to_numpy()).all()
    h.to_csv(carpeta/'historial.csv',index=False,encoding='utf-8-sig')
    resultado={**spec,'ARCHIVO':str(archivo),'HISTORIAL':str(carpeta/'historial.csv'),
        'PARAMETROS_RED':modelo.count_params(),'MEJOR_PASO':int(h.loc[h['val_loss'].idxmin(),'EPOCA']),'PASOS_EJECUTADOS':len(h)}
    del modelo;tf.keras.backend.clear_session();gc.collect()
    return resultado


def entrenar_xgb(spec,datos,ruta):
    v=spec['HISTORIA'];t=datos[v]['training'];val=datos[v]['validation']
    modelo=XGBRegressor(**CONFIG_XGB)
    modelo.fit(t['matriz'],t['y'].ravel(),eval_set=[(t['matriz'],t['y'].ravel()),(val['matriz'],val['y'].ravel())],verbose=False)
    carpeta=Path(ruta)/'modelos'/spec['ID'];carpeta.mkdir(exist_ok=True)
    archivo=carpeta/'modelo.ubj';modelo.save_model(archivo)
    r=modelo.evals_result();h=pd.DataFrame({'RMSE_TRAIN_STD':r['validation_0']['rmse'],'RMSE_VAL_STD':r['validation_1']['rmse']})
    h.insert(0,'ITERACION',np.arange(1,len(h)+1));h.to_csv(carpeta/'historial.csv',index=False,encoding='utf-8-sig')
    return {**spec,'ARCHIVO':str(archivo),'HISTORIAL':str(carpeta/'historial.csv'),'PARAMETROS_RED':None,
        'MEJOR_PASO':int(modelo.best_iteration)+1,'PASOS_EJECUTADOS':len(h)}


def ejecutar_ajustes(datos,ruta,catalogo):
    registro=Path(ruta)/'catalogo_modelos.json'
    completados=json.loads(registro.read_text(encoding='utf-8')) if registro.exists() else []
    ids={r['ID'] for r in completados}
    for i,spec in enumerate(catalogo,1):
        if spec['ID'] in ids:continue
        if spec['REUTILIZADO']:r=spec.copy()
        else:
            print(f'[{i}/{len(catalogo)}] Entrenando {spec["NOMBRE"]}...',flush=True)
            r=entrenar_red(spec,datos,ruta) if spec['TIPO']=='red' else entrenar_xgb(spec,datos,ruta)
            print(f'  Mejor paso {r["MEJOR_PASO"]}; {r["PASOS_EJECUTADOS"]} ejecutados.',flush=True)
        completados.append(r)
        registro.write_text(json.dumps(completados,ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(completados)==50
    return completados


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


def comparar_desarrollo(datos,prep,catalogo,ruta):
    tablas=[];registro=[]
    for i,spec in enumerate(catalogo,1):
        print(f'[{i}/{len(catalogo)}] Midiendo train/validation: {spec["NOMBRE"]}',flush=True)
        v=spec['HISTORIA'];r=spec.copy()
        for conjunto in ['training','validation']:
            d=datos[v][conjunto];z,param=predecir_spec(spec,d)
            meta=d['meta'];t=meta[['FILA_TENSOR','COD_DANE','MUNICIPIO','FECHA_EMISION','FECHA_OBJETIVO','FECHA_ULTIMA_ENTRADA','CASOS_ORIGINALES']].copy()
            t['ID']=spec['ID'];t['MODELO']=spec['NOMBRE'];t['FAMILIA']=spec['FAMILIA'];t['RECETA']=spec['RECETA'];t['HISTORIA']=v
            t['CONJUNTO']=conjunto;t['HORIZONTE']=4
            t['CASOS_REALES']=invertir(d['y'],meta['COD_DANE'],prep)
            t['CASOS_PREDICHOS']=invertir(z,meta['COD_DANE'],prep)
            t['ERROR']=t['CASOS_PREDICHOS']-t['CASOS_REALES']
            np.testing.assert_allclose(t['CASOS_REALES'],t['CASOS_ORIGINALES'],rtol=1e-5,atol=1e-3)
            tablas.append(t)
            r['PARAMETROS_RED']=param
            if conjunto=='validation' and spec['TIPO']=='red':
                h=pd.read_csv(spec['HISTORIAL'],encoding='utf-8-sig')
                r['MEJOR_PASO']=int(h.loc[h['val_loss'].idxmin(),'EPOCA']);r['PASOS_EJECUTADOS']=len(h)
                np.testing.assert_allclose(np.mean((z-d['y'])**2),h['val_loss'].min(),rtol=2e-5,atol=2e-6)
        registro.append(r)
    pred=pd.concat(tablas,ignore_index=True)
    general=calcular_metricas(pred,['ID','MODELO','FAMILIA','RECETA','HISTORIA','CONJUNTO'])
    municipal=calcular_metricas(pred,['ID','MODELO','FAMILIA','HISTORIA','CONJUNTO','COD_DANE','MUNICIPIO'])
    macro=municipal.groupby(['ID','CONJUNTO'],observed=True).agg(MAE_PROMEDIO_MUNICIPAL=('MAE','mean'),RMSE_PROMEDIO_MUNICIPAL=('RMSE','mean')).reset_index()
    general=general.merge(macro,on=['ID','CONJUNTO'],validate='one_to_one')
    general['PARAMETROS_RED']=general['ID'].map({r['ID']:r.get('PARAMETROS_RED') for r in registro})
    val=general.loc[general['CONJUNTO'].eq('validation')].sort_values(['MAE','RMSE','ID']).reset_index(drop=True)
    seleccion={metrica:val.loc[val[metrica].idxmin()].to_dict() for metrica in ['MAE','RMSE']}
    seleccion.update({'criterio_principal':'MAE de validación agrupado, sin redondear','ID_ELEGIDO':val.iloc[0]['ID'],
        'candidato_provisional':True,'test_consultado_en_esta_exploracion':False,'test_historicamente_conocido':True,
        'evaluacion_activada':False,'configuraciones_comparadas':len(val)})
    for c in ['training','validation']:
        pred.loc[pred['CONJUNTO'].eq(c)].to_csv(Path(ruta)/'tablas'/f'predicciones_{c}.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    for nombre,t in [('metricas_desarrollo',general),('metricas_municipales',municipal),('ranking_validacion',val)]:
        t.to_csv(Path(ruta)/'tablas'/f'{nombre}.csv',index=False,encoding='utf-8-sig')
    (Path(ruta)/'seleccion_validacion.json').write_text(json.dumps(seleccion,ensure_ascii=False,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    (Path(ruta)/'catalogo_modelos.json').write_text(json.dumps(registro,ensure_ascii=False,indent=2),encoding='utf-8')
    joblib.dump({'datos':datos,'preprocesamiento':prep},Path(ruta)/'insumos_desarrollo.joblib',compress=3)
    return pred,general,municipal,val,seleccion


def recuperar_resultados(ruta):
    ruta=Path(ruta)
    c=json.loads((ruta/'configuracion_previa.json').read_text(encoding='utf-8'))
    for p,h in c['insumos_sha256'].items():assert sha(p)==h,'Cambió un insumo: '+p
    leer=lambda nombre:pd.read_csv(ruta/'tablas'/nombre,encoding='utf-8-sig')
    train=pd.read_csv(ruta/'tablas/predicciones_training.csv',dtype={'COD_DANE':'string'},parse_dates=['FECHA_EMISION','FECHA_OBJETIVO','FECHA_ULTIMA_ENTRADA'],encoding='utf-8-sig')
    val=pd.read_csv(ruta/'tablas/predicciones_validation.csv',dtype={'COD_DANE':'string'},parse_dates=['FECHA_EMISION','FECHA_OBJETIVO','FECHA_ULTIMA_ENTRADA'],encoding='utf-8-sig')
    return pd.concat([train,val],ignore_index=True),leer('metricas_desarrollo.csv'),leer('metricas_municipales.csv'),leer('ranking_validacion.csv'),json.loads((ruta/'seleccion_validacion.json').read_text(encoding='utf-8')),json.loads((ruta/'catalogo_modelos.json').read_text(encoding='utf-8'))


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
            for receta in ['sin_dense','drop20','drop50','dos_capas','tres_capas']:
                nuevo=ranking.loc[ranking['ID'].eq(f'{f.lower()}_{receta}_v{v}')].iloc[0]
                cambio=100*(base['MAE']-nuevo['MAE'])/base['MAE']
                lecturas.append({'FAMILIA':f,'HISTORIA':v,'RECETA':receta,'MAE_REFERENCIA':base['MAE'],'MAE_VARIANTE':nuevo['MAE'],
                    'MEJORA_MAE_PCT':cambio,'CAMBIO_RMSE':nuevo['RMSE']-base['RMSE']})
    cambios=pd.DataFrame(lecturas)
    drop=cambios.loc[cambios['RECETA'].isin(['drop20','drop50'])]
    profundas=cambios.loc[cambios['RECETA'].isin(['dos_capas','tres_capas'])]
    simples=cambios.loc[cambios['RECETA'].eq('sin_dense')]
    mus=municipal.loc[municipal['CONJUNTO'].eq('validation')&municipal['ID'].eq(mae['ID'])]
    menor=mus.loc[mus['MAE'].idxmin()];mayor=mus.loc[mus['MAE'].idxmax()]
    texto=f"""**Con qué modelo seguiríamos por ahora.** El menor MAE de las 50 configuraciones corresponde a **{mae['MODELO']}**: **{mae['MAE']:.3f} casos**, con RMSE de {mae['RMSE']:.3f}. Elegimos provisionalmente ese modelo porque fijamos MAE como criterio principal. Si miramos solo RMSE, el mínimo es de **{rmse['MODELO']}** ({rmse['RMSE']:.3f}). Ambos errores están expresados en casos de la semana t+4.

**LSTM, GRU y la cantidad de historia.** {'; '.join(referencias)}. Todas las historias tuvieron los mismos 3.800 ejemplos de entrenamiento y 505 objetivos de validación. Por tanto, la comparación cambia la información disponible dentro de cada ventana, sin añadir más fechas de entrenamiento a una opción.

**Qué aportaron Dropout y las capas.** Añadir únicamente Dropout redujo el MAE frente a la referencia en **{int(drop['MEJORA_MAE_PCT'].gt(0).sum())} de las {len(drop)} variantes**. Las estructuras de dos o tres capas con Dropout lo redujeron en **{int(profundas['MEJORA_MAE_PCT'].gt(0).sum())} de {len(profundas)}**, y quitar la Dense intermedia, en **{int(simples['MEJORA_MAE_PCT'].gt(0).sum())} de {len(simples)}**. Cada variante se contrasta con su propia familia e historia. Las redes profundas cambian a la vez capas y Dropout: estos resultados no aíslan el efecto de cada cambio. La tabla siguiente permite revisar dónde mejoró o empeoró cada estructura; mejora porcentual positiva significa menor MAE y cambio de RMSE negativo significa menor RMSE.

**Cómo cambia el error entre municipios.** En el candidato provisional, el MAE municipal va de {menor['MAE']:.3f} en {menor['MUNICIPIO']} a {mayor['MAE']:.3f} en {mayor['MUNICIPIO']}. Son errores en cantidades de casos y los municipios tienen volúmenes distintos; esa diferencia no demuestra por sí sola que el modelo funcione proporcionalmente mejor en uno. Se conserva un mismo modelo global, elegido por el MAE general, para los cinco municipios.

**Hasta dónde podemos concluir.** Estas diferencias describen una ejecución con una semilla y el mismo bloque de validación 2022–2023. Probar más configuraciones puede producir un mínimo optimista; además, las ventanas se solapan. Todavía hacen falta otros cortes cronológicos o semillas y referencias sencillas de persistencia o estacionales para revisar estabilidad. No se ha demostrado superioridad estadística, causalidad ni utilidad de una intervención sanitaria.

**El cierre queda pendiente.** Se mantuvieron los escaladores ajustados en entrenamiento y las predicciones continuas, incluidos los valores negativos que las tablas contabilizan. No se volvió a leer el test durante esta exploración. Como el bloque 2024–2025 ya se había observado antes, usarlo posteriormente tampoco constituirá una prueba ciega. La evaluación y su análisis final permanecen desactivados en el último punto del notebook mientras decidimos con qué configuración continuar.
"""
    return texto,cambios


def guardar_desarrollo(ruta,pred,general,municipal,ranking,catalogo,figuras):
    ruta=Path(ruta)
    texto,cambios=conclusiones_desarrollo(ranking,municipal,catalogo)
    cambios.to_csv(ruta/'tablas/cambios_frente_referencia.csv',index=False,encoding='utf-8-sig')
    resumen=resumir_validacion(ranking);resumen.to_csv(ruta/'tablas/mejores_por_historia_familia.csv',index=False,encoding='utf-8-sig')
    (ruta/'DISCUSION_VALIDACION.md').write_text('# Conclusiones provisionales: solo entrenamiento y validación\n\n'+texto,encoding='utf-8')
    enlaces='\n'.join(f'- [{f}](figuras/{f}.png)' for f in figuras)
    (ruta/'README.md').write_text('''# Exploración temporal: LSTM, GRU y baselines

Se comparan 36 combinaciones recurrentes (dos familias, seis estructuras y tres historias) más referencias anteriores y baselines: 50 configuraciones en total. Se entrenaron 31 redes y dos XGBoost nuevos; los demás ajustes se reutilizaron. Solo se utilizaron entrenamiento (2007–2021) y validación (2022–2023), con los mismos 3.800/505 objetivos en todas las configuraciones.

El candidato actual se elige por MAE de validación. RMSE y métricas municipales se informan por separado. La exploración está abierta: no se leyó ni evaluó el test. La evaluación queda desactivada en el último bloque del notebook; el test ya se había observado históricamente y no debe presentarse como intacto.

## Archivos

- `tablas/ranking_validacion.csv`: las 50 configuraciones ordenadas por MAE y RMSE.
- `tablas/metricas_desarrollo.csv`: métricas de entrenamiento y validación.
- `tablas/metricas_municipales.csv`: errores por municipio, modelo y conjunto.
- `tablas/cambios_frente_referencia.csv`: cambios de arquitectura frente a la referencia de cada familia e historia.
- `tablas/predicciones_training.csv` y `predicciones_validation.csv`: pronósticos individuales y fechas.
- `catalogo_modelos.json`: configuración, archivos y procedencia de cada modelo. Los modelos nuevos están en `modelos/`; las referencias conservan sus rutas originales.
- `seleccion_validacion.json`: candidato provisional y criterio fijado antes de cualquier evaluación posterior.
- `configuracion_previa.json`: recetas y protocolo definidos antes de entrenar.
- `insumos_desarrollo.joblib`: solo entrenamiento, validación y preprocesamiento congelado.
- `DISCUSION_VALIDACION.md`: interpretación calculada a partir de estas métricas.
- `verificacion_entrenamiento.json`: guardas y comprobaciones de los ajustes.

## Figuras: solo desarrollo

'''+enlaces,encoding='utf-8')
    shutil.copy2(Path(__file__),ruta/'exploracion_temporal.py')
    return texto,cambios,resumen


def evaluar_al_cierre(ruta,datos_desarrollo,prep,catalogo,ranking,seleccion):
    """Solo llamar tras activar explícitamente el bloque FINAL del notebook.

    El test ya es conocido históricamente. No es una nueva prueba ciega.
    """
    ruta=Path(ruta)
    ids=[seleccion['ID_ELEGIDO']]
    for familia in ['LSTM','GRU','Regresión lineal','Random Forest','XGBoost']:
        g=ranking.loc[ranking['FAMILIA'].eq(familia)]
        ids.append(g.loc[g['MAE'].idxmin(),'ID'])
    ids=list(dict.fromkeys(ids))
    salida=ruta/'evaluacion_al_cierre'/f'ejecucion_{uuid4().hex[:8]}'
    salida.mkdir(parents=True,exist_ok=False)
    specs=[next(r for r in catalogo if r['ID']==i) for i in ids]
    # Registrar candidatos antes de abrir datos de evaluación.
    (salida/'seleccion_previa.json').write_text(json.dumps({'IDs':ids,'criterio':'MAE de validación','test_historicamente_conocido':True},ensure_ascii=False,indent=2),encoding='utf-8')
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
