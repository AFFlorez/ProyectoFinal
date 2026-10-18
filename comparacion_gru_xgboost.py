"""Ampliación controlada: GRU y XGBoost globales con historia de 16 semanas.

Entrenamiento y elección de checkpoints solo con desarrollo. El test ya abierto
se consulta después de congelar ambos ajustes y se informa como exploratorio.
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
import tensorflow as tf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from evaluacion_final_dengue import calcular_metricas

REFERENCIA = Path('ResultadosModelos/evaluacion_final_t4/ejecucion_7456e638')
DESARROLLO = Path('ResultadosModelos/historia_12_16_20_t4/ejecucion_ba352a18/ventana_16')
ARQUITECTURAS = Path('ResultadosModelos/arquitecturas_lstm_global_t4/ejecucion_1db6e884')
MODELOS = ['Regresión lineal','Random Forest','LSTM global','GRU global','XGBoost']
CONJUNTOS = ['training','validation','evaluation']
COLORES = dict(zip(MODELOS,['#176B93','#238A76','#8D4CA6','#D97628','#BE4256']))
CONFIG_GRU = {'semilla':42,'unidades_gru':32,'unidades_dense':16,'dropout':0.0,
    'learning_rate':.001,'batch_size':32,'epochs_max':100,'paciencia_early_stopping':10,
    'paciencia_reduce_lr':5,'factor_reduce_lr':.5,'learning_rate_min':1e-5}
CONFIG_XGB = {'n_estimators':800,'max_depth':3,'learning_rate':.03,'min_child_weight':5,
    'subsample':.8,'colsample_bytree':.8,'reg_alpha':0.0,'reg_lambda':1.0,
    'objective':'reg:squarederror','eval_metric':'rmse','early_stopping_rounds':50,
    'random_state':42,'n_jobs':4,'tree_method':'hist','device':'cpu'}


def sha(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def preparar_desarrollo():
    """Cargar solo train y validación; no abrir el archivo de tensores del test."""
    fuente = joblib.load(DESARROLLO/'insumos_control.joblib')
    prep = joblib.load(REFERENCIA/'modelo_exportado/preprocesamiento.joblib')
    datos = {}
    for nombre,sufijo in [('training','train'),('validation','val')]:
        X=fuente[f'X_{sufijo}'].copy()
        y=fuente[f'y_{sufijo}'].copy()
        meta=fuente[f'metadata_{sufijo}'].copy(deep=True)
        meta['COD_DANE']=meta['COD_DANE'].astype('string')
        aux=prep['codificador_municipios'].transform(meta[['COD_DANE']])
        if hasattr(aux,'toarray'):aux=aux.toarray()
        matriz=np.concatenate([X.reshape(len(X),-1).astype(float),aux],axis=1)
        assert X.shape==(3800 if nombre=='training' else 505,16,7)
        assert y.shape==(len(X),1) and aux.shape==(len(X),5) and matriz.shape==(len(X),117)
        assert meta['CONJUNTO'].eq(nombre).all() and meta['HORIZONTE'].eq(4).all()
        assert meta['FECHA_OBJETIVO'].eq(meta['FECHA_ULTIMA_ENTRADA']+pd.Timedelta(weeks=4)).all()
        assert np.isfinite(X).all() and np.isfinite(y).all()
        assert np.isin(aux,[0,1]).all() and np.equal(aux.sum(axis=1),1).all()
        datos[nombre]={'X':X,'y':y,'metadata':meta,'aux':aux,'matriz':matriz}
    assert datos['training']['metadata']['FECHA_OBJETIVO'].max()<datos['validation']['metadata']['FECHA_OBJETIVO'].min()
    return datos,prep


def iniciar_experimento(datos,prep):
    ruta=Path('ResultadosModelos/gru_xgboost_t4')/f'ejecucion_{uuid4().hex[:8]}'
    for p in [ruta/'modelos',ruta/'tablas',ruta/'figuras']:
        p.mkdir(parents=True,exist_ok=False)
    config={'fecha_utc':datetime.now(timezone.utc).isoformat(),'lookback':16,'horizonte':4,'n_salidas':1,
        'gru':CONFIG_GRU,'xgboost':CONFIG_XGB,'columnas_temporales':prep['columnas_temporales'],
        'referencia_desarrollo':str(DESARROLLO),'referencia_evaluacion':str(REFERENCIA),
        'test_ya_conocido':True,'uso_test':'Comparación exploratoria posterior al cierre de los dos ajustes. No usado en fit, callbacks o elección.',
        'criterio_checkpoint_gru':'Mínimo MSE de validación en escala transformada',
        'criterio_checkpoint_xgboost':'Mínimo RMSE de validación en escala transformada',
        'comparacion_modelos':'MAE y RMSE en casos originales; sin redondeo ni recorte',
        'nuevos_ajustes':['GRU global','XGBoost'],'modelos_anteriores_reentrenados':False,
        'versiones':{p:importlib.metadata.version(p) for p in ['tensorflow','xgboost','scikit-learn','numpy','pandas','joblib']},
        'insumos_sha256':{'desarrollo':sha(DESARROLLO/'insumos_control.joblib'),
                         'preprocesamiento':sha(REFERENCIA/'modelo_exportado/preprocesamiento.joblib')}}
    (ruta/'configuracion_previa.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    joblib.dump(prep,ruta/'preprocesamiento.joblib',compress=3)
    joblib.dump(datos,ruta/'insumos_desarrollo.joblib',compress=3)
    return ruta


def dataset_gru(d,con_objetivo=True):
    entradas={'historia':d['X'],'municipio':d['aux']}
    opciones=tf.data.Options()
    opciones.experimental_deterministic=True
    opciones.threading.private_threadpool_size=1
    opciones.threading.max_intra_op_parallelism=1
    elementos=(entradas,d['y']) if con_objetivo else entradas
    return tf.data.Dataset.from_tensor_slices(elementos).batch(CONFIG_GRU['batch_size']).with_options(opciones)


def entrenar_gru(datos,ruta):
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(CONFIG_GRU['semilla'])
    historia=tf.keras.Input(shape=(16,7),name='historia')
    municipio=tf.keras.Input(shape=(5,),name='municipio')
    representacion=tf.keras.layers.GRU(CONFIG_GRU['unidades_gru'],reset_after=True,dropout=CONFIG_GRU['dropout'],recurrent_dropout=0.0,name='gru')(historia)
    combinado=tf.keras.layers.Concatenate(name='historia_y_municipio')([representacion,municipio])
    oculta=tf.keras.layers.Dense(CONFIG_GRU['unidades_dense'],activation='relu',name='dense_16')(combinado)
    salida=tf.keras.layers.Dense(1,activation='linear',name='casos_std_t4')(oculta)
    modelo=tf.keras.Model({'historia':historia,'municipio':municipio},salida,name='gru_global_t4_v16')
    modelo.compile(optimizer=tf.keras.optimizers.Adam(CONFIG_GRU['learning_rate']),loss='mse',metrics=['mae'])
    archivo=Path(ruta)/'modelos/gru_global_t4.keras'
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=CONFIG_GRU['paciencia_early_stopping'],restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',mode='min',patience=CONFIG_GRU['paciencia_reduce_lr'],factor=CONFIG_GRU['factor_reduce_lr'],min_lr=CONFIG_GRU['learning_rate_min']),
        tf.keras.callbacks.ModelCheckpoint(str(archivo),monitor='val_loss',mode='min',save_best_only=True),
        tf.keras.callbacks.CSVLogger(str(Path(ruta)/'tablas/gru_log_entrenamiento.csv')),
        tf.keras.callbacks.TerminateOnNaN()]
    print(f'Entrenando GRU global: {modelo.count_params():,} parámetros, 16 semanas y una salida t+4...',flush=True)
    historia=modelo.fit(dataset_gru(datos['training']),validation_data=dataset_gru(datos['validation']),
        epochs=CONFIG_GRU['epochs_max'],callbacks=callbacks,shuffle=False,verbose=0)
    h=pd.DataFrame(historia.history);h.insert(0,'EPOCA',np.arange(1,len(h)+1))
    assert np.isfinite(h[['loss','val_loss','mae','val_mae']].to_numpy()).all()
    h.to_csv(Path(ruta)/'tablas/historial_gru.csv',index=False,encoding='utf-8-sig')
    mejor=tf.keras.models.load_model(archivo,compile=False)
    epoca=int(h.loc[h['val_loss'].idxmin(),'EPOCA'])
    print(f'GRU: {len(h)} épocas; checkpoint de la época {epoca}.',flush=True)
    return {'modelo':mejor,'historial':h,'mejor_epoca':epoca,'archivo':archivo}


def entrenar_xgboost(datos,ruta):
    modelo=XGBRegressor(**CONFIG_XGB)
    t,v=datos['training'],datos['validation']
    print('Entrenando XGBoost: 117 predictores, hasta 800 rondas y una salida t+4...',flush=True)
    modelo.fit(t['matriz'],t['y'].ravel(),eval_set=[(t['matriz'],t['y'].ravel()),(v['matriz'],v['y'].ravel())],verbose=False)
    resultados=modelo.evals_result()
    h=pd.DataFrame({'RMSE_TRAIN_STD':resultados['validation_0']['rmse'],'RMSE_VAL_STD':resultados['validation_1']['rmse']})
    h.insert(0,'ITERACION',np.arange(1,len(h)+1))
    assert np.isfinite(h.to_numpy()).all()
    h.to_csv(Path(ruta)/'tablas/historial_xgboost.csv',index=False,encoding='utf-8-sig')
    archivo=Path(ruta)/'modelos/xgboost_t4.ubj'
    modelo.save_model(archivo)
    cargado=XGBRegressor();cargado.load_model(archivo)
    assert cargado.best_iteration==modelo.best_iteration
    print(f'XGBoost: {len(h)} rondas; se usan las primeras {modelo.best_iteration+1}.',flush=True)
    return {'modelo':cargado,'historial':h,'mejor_iteracion':int(modelo.best_iteration),
        'arboles_entrenados':modelo.get_booster().num_boosted_rounds(),'archivo':archivo}


def predecir_nuevos(resultados,d):
    gru=resultados['GRU global']['modelo']
    partes=[]
    for i in range(0,len(d['X']),32):
        partes.append(gru({'historia':d['X'][i:i+32],'municipio':d['aux'][i:i+32]},training=False).numpy())
    xgb=resultados['XGBoost']['modelo']
    return {'GRU global':np.concatenate(partes).astype(float),
        'XGBoost':np.asarray(xgb.predict(d['matriz'],iteration_range=(0,xgb.best_iteration+1)),dtype=float).reshape(-1,1)}


def invertir(valores,meta,prep):
    z=np.asarray(valores,dtype=float).reshape(-1,1)
    assert z.shape==(len(meta),1) and np.isfinite(z).all()
    salida=np.empty(len(z))
    for codigo,pos in meta.groupby('COD_DANE',observed=True).indices.items():
        salida[pos]=np.expm1(prep['escaladores_casos'][codigo].inverse_transform(z[pos])).ravel()
    assert np.isfinite(salida).all()
    return salida


def comparar_periodo(split,d,resultados,prep):
    anteriores=pd.read_csv(REFERENCIA/f'tablas/predicciones_{split}.csv',dtype={'COD_DANE':'string'},parse_dates=['FECHA_EMISION','FECHA_OBJETIVO','FECHA_ULTIMA_ENTRADA','FECHA_INICIO_HISTORICO'],encoding='utf-8-sig')
    referencia=anteriores.loc[anteriores['MODELO'].eq('LSTM global')].sort_values('FILA_TENSOR').reset_index(drop=True)
    meta=d['metadata']
    for c in ['COD_DANE','FECHA_EMISION','FECHA_OBJETIVO']:
        np.testing.assert_array_equal(referencia[c],meta[c])
    reales=invertir(d['y'],meta,prep)
    np.testing.assert_allclose(reales,referencia['CASOS_REALES'],rtol=1e-10,atol=1e-8)
    pred=predecir_nuevos(resultados,d)
    tablas=[anteriores]
    for nombre,z in pred.items():
        tabla=referencia.copy(deep=True)
        tabla['MODELO']=nombre
        tabla['CASOS_REALES']=reales
        tabla['CASOS_PREDICHOS']=invertir(z,meta,prep)
        tabla['ERROR']=tabla['CASOS_PREDICHOS']-tabla['CASOS_REALES']
        tabla['PRONOSTICO_NEGATIVO']=tabla['CASOS_PREDICHOS'].lt(0)
        tablas.append(tabla)
    salida=pd.concat(tablas,ignore_index=True)
    for _,g in salida.groupby('MODELO'):
        assert len(g)==len(meta) and not g.duplicated(['COD_DANE','FECHA_OBJETIVO']).any()
    return salida,pred


def metricas_completas(tabla):
    general=calcular_metricas(tabla,['CONJUNTO','MODELO'])
    municipal=calcular_metricas(tabla,['CONJUNTO','MODELO','COD_DANE','MUNICIPIO'])
    macro=municipal.groupby(['CONJUNTO','MODELO'],observed=True).agg(MAE_PROMEDIO_MUNICIPAL=('MAE','mean'),RMSE_PROMEDIO_MUNICIPAL=('RMSE','mean')).reset_index()
    return general.merge(macro,on=['CONJUNTO','MODELO'],validate='one_to_one'),municipal


def comparar_desarrollo(datos,prep,resultados,ruta):
    tablas=[];pred_std={}
    for split in ['training','validation']:
        tabla,p=comparar_periodo(split,datos[split],resultados,prep)
        tablas.append(tabla);pred_std[split]=p
    predicciones=pd.concat(tablas,ignore_index=True)
    general,municipal=metricas_completas(predicciones)
    v=general.loc[general['CONJUNTO'].eq('validation')]
    eleccion={metrica:{'modelo':v.loc[v[metrica].idxmin(),'MODELO'],'error':float(v[metrica].min())} for metrica in ['MAE','RMSE']}
    eleccion.update({'conjunto':'validation','modelos_comparados':MODELOS,'test_ya_conocido':True,
        'solo_eleccion_de_desarrollo':True,'modelos_congelados':{nombre:{'archivo':str(r['archivo']),'sha256':sha(r['archivo'])} for nombre,r in resultados.items()}})
    (Path(ruta)/'seleccion_validacion_antes_comparacion_test.json').write_text(json.dumps(eleccion,ensure_ascii=False,indent=2),encoding='utf-8')
    for nombre,t in [('predicciones_desarrollo',predicciones),('metricas_desarrollo',general),('metricas_desarrollo_municipio',municipal)]:
        t.to_csv(Path(ruta)/'tablas'/f'{nombre}.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    for split,p in pred_std.items():
        np.savez_compressed(Path(ruta)/f'predicciones_std_{split}.npz',**{k.replace(' ','_'):z for k,z in p.items()})
    return predicciones,general,municipal,eleccion


def comparar_test_exploratorio(datos_desarrollo,prep,resultados,ruta):
    registro=Path(ruta)/'seleccion_validacion_antes_comparacion_test.json'
    assert registro.is_file(),'Cerrar los dos ajustes y registrar validación antes de consultar el test.'
    for nombre,r in resultados.items():
        assert json.loads(registro.read_text(encoding='utf-8'))['modelos_congelados'][nombre]['sha256']==sha(r['archivo'])
    existentes=joblib.load(REFERENCIA/'insumos_evaluacion.joblib')
    for split in ['training','validation']:
        for k,previo in [('X','X'),('y','y')]:
            np.testing.assert_array_equal(datos_desarrollo[split][k],existentes[split][previo])
    fuente=existentes['evaluation']
    X=fuente['X'].copy();meta=fuente['metadata'].copy(deep=True)
    meta['COD_DANE']=meta['COD_DANE'].astype('string')
    assert X.shape==(510,16,7) and fuente['y'].shape==(510,1)
    assert meta['FECHA_EMISION'].ge(pd.Timestamp('2023-12-30')).all()
    assert meta['FECHA_OBJETIVO'].eq(meta['FECHA_ULTIMA_ENTRADA']+pd.Timedelta(weeks=4)).all()
    aux=prep['codificador_municipios'].transform(meta[['COD_DANE']])
    if hasattr(aux,'toarray'):aux=aux.toarray()
    d={'X':X,'y':fuente['y'].copy(),'metadata':meta,'aux':aux,
       'matriz':np.concatenate([X.reshape(len(X),-1).astype(float),aux],axis=1)}
    tabla,pred= comparar_periodo('evaluation',d,resultados,prep)
    tabla['USO']='Exploratorio: test ya observado en una etapa anterior'
    tabla.to_csv(Path(ruta)/'tablas/predicciones_test_exploratorio.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    np.savez_compressed(Path(ruta)/'predicciones_std_evaluation.npz',**{k.replace(' ','_'):p for k,p in pred.items()})
    return tabla,d


def consolidar_comparacion(pred_desarrollo,pred_test,resultados,ruta):
    tabla=pd.concat([pred_desarrollo,pred_test],ignore_index=True)
    tabla['TIPO_ANALISIS']=tabla['CONJUNTO'].map({'training':'Ajuste dentro de muestra','validation':'Desarrollo y selección','evaluation':'Comparación exploratoria; test ya conocido'})
    general,municipal=metricas_completas(tabla)
    # Contexto de las demás estructuras LSTM de 16 semanas, sin nuevos ajustes.
    adicionales=pd.read_csv(ARQUITECTURAS/'predicciones_validacion.csv',dtype={'COD_DANE':'string'},parse_dates=['FECHA_EMISION','FECHA_OBJETIVO'],encoding='utf-8-sig')
    nombres=['LSTM 64','LSTM 32 + Dropout 0,20','LSTM 32 + LSTM 32','LSTM local']
    extendida=pd.concat([tabla.loc[tabla['CONJUNTO'].eq('validation')],adicionales.loc[adicionales['MODELO'].isin(nombres)]],ignore_index=True)
    ref=extendida.loc[extendida['MODELO'].eq('LSTM global')].sort_values('FILA_TENSOR')
    for nombre,g in extendida.groupby('MODELO'):
        g=g.sort_values('FILA_TENSOR')
        assert len(g)==505
        for c in ['COD_DANE','FECHA_EMISION','FECHA_OBJETIVO']:np.testing.assert_array_equal(g[c],ref[c])
        np.testing.assert_allclose(g['CASOS_REALES'],ref['CASOS_REALES'],rtol=1e-10,atol=1e-8)
    metricas_ext,municipales_ext=metricas_completas(extendida)
    resumen=pd.DataFrame([
        {'MODELO':'GRU global','PARAMETROS_RED':resultados['GRU global']['modelo'].count_params(),
         'PASOS_EJECUTADOS':len(resultados['GRU global']['historial']),'MEJOR_PASO':resultados['GRU global']['mejor_epoca'],
         'CRITERIO':'MSE de validación transformado','VALOR_MINIMO':resultados['GRU global']['historial']['val_loss'].min()},
        {'MODELO':'XGBoost','PARAMETROS_RED':np.nan,
         'PASOS_EJECUTADOS':len(resultados['XGBoost']['historial']),'MEJOR_PASO':resultados['XGBoost']['mejor_iteracion']+1,
         'CRITERIO':'RMSE de validación transformado','VALOR_MINIMO':resultados['XGBoost']['modelo'].best_score}])
    for nombre,t in [('predicciones_cinco_modelos',tabla),('metricas_cinco_modelos',general),('metricas_municipales',municipal),
        ('validacion_ampliada_nueve_enfoques',metricas_ext),('validacion_ampliada_municipal',municipales_ext),('resumen_entrenamiento',resumen)]:
        t.to_csv(Path(ruta)/'tablas'/f'{nombre}.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    return tabla,general,municipal,metricas_ext,resumen


def graficar_comparacion(tabla,general,municipal,extendida,resultados,ruta):
    nombres=[]
    def guardar(fig,nombre):
        for ext in ['png','svg']:
            fig.savefig(Path(ruta)/'figuras'/f'{nombre}.{ext}',dpi=300,bbox_inches='tight')
        nombres.append(nombre)
        plt.show()
    with plt.rc_context({'font.size':10,'svg.fonttype':'none'}):
        fig,ejes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
        g=resultados['GRU global'];h=g['historial']
        ejes[0].plot(h['EPOCA'],h['loss'],label='Entrenamiento',color='#176B93')
        ejes[0].plot(h['EPOCA'],h['val_loss'],label='Validación',color='#D97628')
        ejes[0].axvline(g['mejor_epoca'],linestyle='--',color='#777777',label='Mejor época')
        ejes[0].set(title='GRU global',xlabel='Época',ylabel='MSE de CASOS_std')
        x=resultados['XGBoost'];h=x['historial']
        ejes[1].plot(h['ITERACION'],h['RMSE_TRAIN_STD'],label='Entrenamiento',color='#176B93')
        ejes[1].plot(h['ITERACION'],h['RMSE_VAL_STD'],label='Validación',color='#D97628')
        ejes[1].axvline(x['mejor_iteracion']+1,linestyle='--',color='#777777',label='Mejor ronda')
        ejes[1].set(title='XGBoost',xlabel='Ronda de boosting',ylabel='RMSE de CASOS_std')
        for eje in ejes:eje.legend(fontsize=8);eje.grid(alpha=.2)
        fig.suptitle('Dos ajustes nuevos · 16 semanas · t+4\nCada panel conserva su criterio en escala transformada')
        guardar(fig,'01_curvas_gru_xgboost')
        fig,ejes=plt.subplots(3,2,figsize=(12,12),constrained_layout=True)
        etiquetas={'training':'Entrenamiento: ajuste','validation':'Validación: desarrollo','evaluation':'Test conocido: exploratorio'}
        for fila,split in enumerate(CONJUNTOS):
            g=general.loc[general['CONJUNTO'].eq(split)].set_index('MODELO').reindex(MODELOS)
            for eje,metrica in zip(ejes[fila],['MAE','RMSE']):
                b=eje.barh(MODELOS,g[metrica],color=[COLORES[m] for m in MODELOS])
                eje.bar_label(b,fmt='%.2f',padding=4)
                eje.set(title=f'{etiquetas[split]} · {metrica}',xlabel=f'{metrica} (casos)',xlim=(0,max(g[metrica].max()*1.23,1)))
                eje.grid(axis='x',alpha=.2)
        fig.suptitle('Cinco modelos globales · Misma historia de 16 semanas · Objetivo único t+4')
        guardar(fig,'02_metricas_cinco_modelos')
        for numero,(codigo,g) in enumerate(tabla.groupby('COD_DANE',observed=True,sort=True),3):
            fig,ejes=plt.subplots(3,1,figsize=(13,9.5),constrained_layout=True)
            for eje,split in zip(ejes,CONJUNTOS):
                periodo=g.loc[g['CONJUNTO'].eq(split)]
                real=periodo.loc[periodo['MODELO'].eq('LSTM global')].sort_values('FECHA_OBJETIVO')
                eje.plot(real['FECHA_OBJETIVO'],real['CASOS_ORIGINALES'],label='Observado',color='#252525',linewidth=1.6)
                for modelo in MODELOS:
                    p=periodo.loc[periodo['MODELO'].eq(modelo)].sort_values('FECHA_OBJETIVO')
                    eje.plot(p['FECHA_OBJETIVO'],p['CASOS_PREDICHOS'],label=modelo,color=COLORES[modelo],linewidth=1.05,alpha=.9)
                loc=mdates.AutoDateLocator(minticks=5,maxticks=9)
                eje.xaxis.set_major_locator(loc);eje.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
                eje.set(title=etiquetas[split],ylabel='Casos semanales',xlabel='Inicio de la semana objetivo t+4')
                eje.grid(alpha=.2)
            handles,labels=ejes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='outside lower center',ncol=3)
            fig.suptitle(f'{g["MUNICIPIO"].iloc[0]} · Observados y cinco modelos · t+4\nEscalas temporal y vertical independientes en cada panel')
            guardar(fig,f'{numero:02d}_series_{codigo}_t4')
        fig,ejes=plt.subplots(1,2,figsize=(14,7),constrained_layout=True)
        test=municipal.loc[municipal['CONJUNTO'].eq('evaluation')]
        for eje,metrica in zip(ejes,['MAE','RMSE']):
            test.pivot(index='MUNICIPIO',columns='MODELO',values=metrica).reindex(columns=MODELOS).plot.barh(ax=eje,color=[COLORES[m] for m in MODELOS])
            eje.set(title=metrica,xlabel=f'{metrica} (casos)',ylabel='');eje.grid(axis='x',alpha=.2)
        handles,labels=ejes[0].get_legend_handles_labels()
        for eje in ejes:eje.get_legend().remove()
        fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle('Comparación municipal de cinco modelos · Test ya conocido: exploratorio · t+4')
        guardar(fig,'08_metricas_municipales_test_exploratorio')
        fig,ejes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
        orden=extendida.sort_values('MAE')['MODELO'].tolist()
        g=extendida.set_index('MODELO').loc[orden]
        for eje,metrica in zip(ejes,['MAE','RMSE']):
            b=eje.barh(orden,g[metrica],color=[COLORES.get(m,'#788D9D') for m in orden])
            eje.bar_label(b,fmt='%.2f',padding=4)
            eje.set(title=metrica,xlabel=f'{metrica} (casos)',xlim=(0,g[metrica].max()*1.23))
            eje.grid(axis='x',alpha=.2)
        fig.suptitle('Validación ampliada · Nueve enfoques · Misma historia de 16 semanas y mismos 505 objetivos')
        guardar(fig,'09_validacion_nueve_enfoques')
    return nombres


def discutir_resultados(general,municipal,extendida,resultados):
    v=general.loc[general['CONJUNTO'].eq('validation')].set_index('MODELO')
    test=general.loc[general['CONJUNTO'].eq('evaluation')].set_index('MODELO')
    ganador_v_mae,ganador_v_rmse=v['MAE'].idxmin(),v['RMSE'].idxmin()
    ganador_t_mae,ganador_t_rmse=test['MAE'].idxmin(),test['RMSE'].idxmin()
    mejoras=[]
    for nuevo in ['GRU global','XGBoost']:
        cambios=[]
        for metrica in ['MAE','RMSE']:
            delta=100*(v.loc['LSTM global',metrica]-v.loc[nuevo,metrica])/v.loc['LSTM global',metrica]
            cambios.append(f"{metrica}: {abs(delta):.2f}% de {'reducción' if delta>=0 else 'aumento'}")
        mejoras.append(f"{nuevo} ({'; '.join(cambios)})")
    detalle=[]
    for nombre,g in municipal.loc[municipal['CONJUNTO'].eq('validation')].groupby('MUNICIPIO',sort=True):
        a,b=g.loc[g['MAE'].idxmin()],g.loc[g['RMSE'].idxmin()]
        detalle.append(f"{nombre}: {a['MODELO']} en MAE y {b['MODELO']} en RMSE")
    mejor_ext=extendida.loc[extendida['MAE'].idxmin()]
    g=resultados['GRU global'];x=resultados['XGBoost']
    return f"""**Qué se probó.** Se entrenaron únicamente una GRU global y un XGBoost con 16 semanas y una salida t+4. Regresión lineal, Random Forest y la LSTM global conservan sus ajustes anteriores. Todos se comparan sobre las mismas 3.800 ventanas de entrenamiento, 505 de validación y 510 del test ya conocido. No se reajustaron escaladores ni codificadores y el test no intervino en los entrenamientos o callbacks.

**Validación: comparación principal.** Entre los cinco modelos, el menor MAE corresponde a **{ganador_v_mae}** ({v.loc[ganador_v_mae,'MAE']:.3f} casos) y el menor RMSE a **{ganador_v_rmse}** ({v.loc[ganador_v_rmse,'RMSE']:.3f} casos). Frente a la LSTM original: {'; '.join(mejoras)}. En la tabla ampliada de nueve enfoques de 16 semanas, el menor MAE corresponde a {mejor_ext['MODELO']} ({mejor_ext['MAE']:.3f}). Son mínimos observados sobre la misma validación de desarrollo, no diferencias estadísticamente demostradas ni garantías para otros periodos.

**Complejidad y entrenamiento.** La GRU tiene {g['modelo'].count_params():,} parámetros frente a los 5.745 de la LSTM original, conservando Dense de 16 y el one-hot municipal. Tener menos parámetros no garantiza menor error. La GRU seleccionó la época {g['mejor_epoca']} de {len(g['historial'])} ejecutadas; XGBoost utiliza {x['mejor_iteracion']+1} rondas de {len(x['historial'])} ejecutadas. Los criterios de parada se calcularon en escala transformada. Las métricas finales se miden en casos, y pueden ordenar los modelos de otra forma. Los árboles no se comparan mediante un conteo de parámetros equivalente al de una red.

**Municipios.** Los mínimos de validación entre los cinco modelos son: {'; '.join(detalle)}. La ventaja agrupada no implica una ventaja en todos los municipios. Elegir retrospectivamente un modelo diferente para cada municipio requeriría evaluar esa nueva estrategia por separado.

**Test ya abierto: análisis exploratorio.** Los mínimos observados son de {ganador_t_mae} en MAE ({test.loc[ganador_t_mae,'MAE']:.3f}) y {ganador_t_rmse} en RMSE ({test.loc[ganador_t_rmse,'RMSE']:.3f}). La GRU obtiene MAE {test.loc['GRU global','MAE']:.3f} y RMSE {test.loc['GRU global','RMSE']:.3f}; XGBoost, MAE {test.loc['XGBoost','MAE']:.3f} y RMSE {test.loc['XGBoost','RMSE']:.3f}. Estos resultados amplían la comparación después de haber observado ese test en el punto 20. No se presentan como una nueva confirmación independiente de selección ni se utiliza ese mínimo para sustituir automáticamente el modelo exportado. La historia, la semilla y las configuraciones no se cambiaron después de observar estos nuevos errores.

**Alcance.** Se pronostican casos registrados de una semana específica t+4, no un acumulado mensual. La historia se actualiza con observaciones hasta t, sin clima futuro y con pesos fijos. Las 510 ventanas respetan la emisión posterior o igual al cierre de validación (30 de diciembre de 2023); las tres primeras semanas por municipio siguen excluidas. Las métricas conservan predicciones continuas sin recorte; los valores negativos se contabilizan en las tablas. El ajuste en entrenamiento no es evaluación independiente. Falta estudiar otras semillas y cortes temporales, referencias de persistencia o estacionales y un periodo independiente para confirmar cualquier nueva selección. No se atribuye causalidad al clima, significación estadística o capacidad formal de detectar brotes.
"""


def guardar_cierre(ruta,datos,prep,resultados,general,municipal,extendida,figuras):
    ruta=Path(ruta)
    texto=discutir_resultados(general,municipal,extendida,resultados)
    (ruta/'DISCUSION_Y_CONCLUSIONES.md').write_text('# GRU y XGBoost · Historia de 16 semanas · t+4\n\n'+texto,encoding='utf-8')
    # Ejemplo pequeño que permite comprobar la carga nativa sin repetir ajustes.
    meta=datos['validation']['metadata']
    posiciones=np.array([np.flatnonzero(meta['COD_DANE'].eq(c))[0] for c in sorted(meta['COD_DANE'].unique())])
    d=datos['validation']
    muestra={k:d[k][posiciones].copy() for k in ['X','aux','matriz']}
    z=predecir_nuevos(resultados,muestra)
    np.savez_compressed(ruta/'ejemplo_inferencia.npz',historia=muestra['X'],municipio=muestra['aux'],matriz_117=muestra['matriz'],
        codigos=meta.iloc[posiciones]['COD_DANE'].to_numpy(dtype='U5'),gru_std=z['GRU global'],xgboost_std=z['XGBoost'])
    cfg={'gru':{'parametros':resultados['GRU global']['modelo'].count_params(),'mejor_epoca':resultados['GRU global']['mejor_epoca']},
         'xgboost':{'mejor_iteracion_base_cero':resultados['XGBoost']['mejor_iteracion'],'rondas_usadas':resultados['XGBoost']['mejor_iteracion']+1,
                    'rondas_ejecutadas':len(resultados['XGBoost']['historial'])},'test_uso':'Exploratorio; test ya abierto antes de esta ampliación'}
    (ruta/'resumen_ajustes.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    tabla=['| Conjunto | Modelo | MAE | RMSE |','| --- | --- | ---: | ---: |']
    for r in general.itertuples():tabla.append(f'| {r.CONJUNTO} | {r.MODELO} | {r.MAE:.3f} | {r.RMSE:.3f} |')
    enlaces='\n'.join(f'- [{nombre}](figuras/{nombre}.png)' for nombre in figuras)
    readme='''# Comparación de GRU y XGBoost con LSTM y baselines

La comparación principal contiene cinco modelos con historia de 16 semanas y objetivo único t+4. Se entrenaron solo GRU y XGBoost, sin modificar los ajustes o exportaciones anteriores. Validación se usa para detener el entrenamiento y comparar; el test ya fue observado en el punto 20 y esta ampliación lo informa como exploratorio.

## Métricas

'''+ '\n'.join(tabla)+'''

## Archivos

- `modelos/gru_global_t4.keras`: checkpoint GRU elegido por val_loss.
- `modelos/xgboost_t4.ubj`: modelo XGBoost en formato nativo, con best_iteration. Para pronosticar se usa iteration_range=(0, best_iteration+1).
- `preprocesamiento.joblib`: los mismos escaladores y one-hot, sin nuevos ajustes.
- `insumos_desarrollo.joblib`: entrenamiento y validación reutilizados.
- `configuracion_previa.json`: configuraciones fijadas antes de entrenar.
- `seleccion_validacion_antes_comparacion_test.json`: resultados de desarrollo y huellas de los modelos, guardados antes de volver a consultar el test.
- `tablas/`: métricas y predicciones por conjunto y municipio; comparación ampliada de nueve enfoques en validación; historiales.
- `DISCUSION_Y_CONCLUSIONES.md`: interpretación generada desde los resultados.
- `ejemplo_inferencia.npz`: cinco ejemplos, uno por municipio, con las predicciones estandarizadas para verificar la carga de los dos modelos.

La GRU recibe un diccionario con historia (N,16,7) y municipio (N,5). XGBoost recibe las 117 columnas: las siete variables históricas aplanadas por semana, más los cinco indicadores municipales. Ambos entregan CASOS_std. Para recuperar casos: inverse_transform del escalador de ese municipio y luego expm1. No se redondea o recorta.

## Gráficas

'''+enlaces+'''

Cada figura se conserva en PNG a 300 dpi y SVG. Las series tienen un panel por conjunto; las figuras del test indican explícitamente su carácter exploratorio.

## Referencias de implementación

- [GRU, documentación oficial de TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU).
- [XGBRegressor, documentación oficial de XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor).

El modelo exportado en el punto 20 permanece intacto. Cualquier nueva elección a partir de esta ampliación requiere confirmación en un periodo independiente; el test conocido no vuelve a considerarse intacto por cambiarle el nombre.
'''
    (ruta/'README.md').write_text(readme,encoding='utf-8')
    for nombre in ['comparacion_gru_xgboost.py','evaluacion_final_dengue.py']:
        shutil.copy2(Path(__file__).with_name(nombre),ruta/nombre)
    (ruta/'requirements.txt').write_text('\n'.join(f'{p}=={importlib.metadata.version(p)}' for p in ['tensorflow','xgboost','numpy','pandas','scikit-learn','matplotlib','joblib'])+'\n',encoding='utf-8')
    manifest={'test_ya_conocido':True,'test_usado_en_fit':False,'ajustes_nuevos':2,'referencias':{'desarrollo':str(DESARROLLO),'evaluacion_previa':str(REFERENCIA)},
        'archivos_sha256':{str(p.relative_to(ruta)):sha(p) for p in ruta.rglob('*') if p.is_file()}}
    (ruta/'manifiesto.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return texto
