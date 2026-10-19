from pathlib import Path
import json
import joblib
import numpy as np
import tensorflow as tf
from xgboost import XGBRegressor

carpeta = Path(__file__).resolve().parent
config = json.loads((carpeta/'configuracion.json').read_text(encoding='utf-8'))
prep = joblib.load(carpeta/'preprocesamiento.joblib')
with np.load(carpeta/'entradas_ejemplo_validacion.npz') as ejemplo:
    X, aux, matriz, codigos = (ejemplo[k] for k in ['historia','municipio','matriz','codigos'])
    if config['tipo']=='local':
        z = np.empty((len(X),1))
        for codigo,archivo in config['archivos'].items():
            pos = np.flatnonzero(codigos==codigo)
            if len(pos):
                modelo = tf.keras.models.load_model(carpeta/archivo,compile=False)
                z[pos] = modelo({'historia':X[pos]},training=False).numpy()
    elif config['tipo']=='red':
        modelo = tf.keras.models.load_model(carpeta/config['archivos'],compile=False)
        z = modelo({'historia':X,'municipio':aux},training=False).numpy()
    elif config['tipo']=='xgboost':
        modelo = XGBRegressor()
        modelo.load_model(carpeta/config['archivos'])
        z = modelo.predict(matriz,iteration_range=(0,modelo.best_iteration+1)).reshape(-1,1)
    else:
        modelo = joblib.load(carpeta/config['archivos'])
        z = modelo.predict(matriz).reshape(-1,1)
    casos = np.array([np.expm1(prep['escaladores_casos'][codigo].inverse_transform(z[i:i+1]))[0,0]
        for i,codigo in enumerate(codigos)])
print(casos)
