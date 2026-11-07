"""Cargar en MLflow la comparación de modelos y registrar el modelo elegido.

Usa los archivos versionados en modelos/experimentos/ (los genera
notebooks/exportar_experimentos.py) y el paquete de modelos/. No entrena nada:
reproduce en MLflow los 100 enfoques evaluados en el notebook.

    python infra/mlflow/registrar_experimentos.py --tracking http://127.0.0.1:5000 --modelos modelos
"""
import argparse
import json
import math
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

EXPERIMENTO = 'pronostico-dengue-t4'
MODELO_REGISTRADO = 'pronostico-dengue-t4'
COMUNES = {
    'horizonte_semanas': 4,
    'objetivo': 'casos de la semana t+4',
    'municipios': 'Cali, Buga, Palmira, Tuluá, Yumbo',
    'periodo_entrenamiento': '2007-2021',
    'periodo_validacion': '2022-2023',
    'variables': 'casos, temp. media/máx/mín, precipitación, seno/coseno de la semana',
}
ENTRENAMIENTO_LSTM = {
    'optimizador': 'Adam', 'tasa_aprendizaje_inicial': 0.001, 'tamano_lote': 32, 'epocas_max': 200,
    'early_stopping_paciencia': 20, 'reduce_lr_paciencia': 5, 'semilla': 42, 'barajar': False,
    'perdida': 'MSE sobre log1p(casos) estandarizado',
}
BASELINES = {
    'Regresión lineal': {'implementacion': 'scikit-learn LinearRegression, valores predeterminados'},
    'Random Forest': {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 3},
    'XGBoost': {'n_estimators_max': 800, 'max_depth': 3, 'learning_rate': 0.03, 'early_stopping_rounds': 50},
}
FIGURAS_GENERALES = ['baselines_validacion', 'lstm64_comparacion', 'lstm64_curvas', 'locales_comparacion']


def numero(valor):
    try:
        valor = float(valor)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(valor) else valor


def parametros(fila):
    familia = fila['FAMILIA']
    p = {**COMUNES, 'familia': familia, 'historia_semanas': int(fila['HISTORIA'])}
    if familia.startswith('LSTM'):
        p.update(ENTRENAMIENTO_LSTM)
        p.update({'unidades_lstm': str(fila['UNIDADES']).replace(' ', ''),
                  'dense_intermedia': int(numero(fila['DENSE_INTERMEDIA']) or 0),
                  'dropout': numero(fila['DROPOUT']) or 0.0,
                  'parametros_por_red': int(fila['PARAMETROS_RED']),
                  'alcance': 'una red por municipio (5 redes)' if familia == 'LSTM local' else 'una red para los 5 municipios (one-hot)'})
    else:
        p.update(BASELINES[familia])
        p['alcance'] = 'un modelo para los 5 municipios (one-hot)'
    return p


def main():
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument('--tracking', default='http://127.0.0.1:5000')
    args.add_argument('--modelos', type=Path, help='Carpeta del paquete del modelo (por defecto, modelos/ del repositorio)')
    args = args.parse_args()
    args.modelos = args.modelos or Path(__file__).resolve().parents[2] / 'modelos'

    carpeta = args.modelos / 'experimentos'
    figuras = carpeta / 'figuras'
    config = json.loads((args.modelos / 'configuracion.json').read_text(encoding='utf-8-sig'))
    ranking = pd.read_csv(carpeta / 'ranking_validacion.csv')
    baselines = pd.read_csv(carpeta / 'metricas_baselines.csv')
    tres = pd.read_csv(carpeta / 'metricas_tres_conjuntos.csv').set_index('MODELO')
    prueba = pd.read_csv(carpeta / 'evaluacion_prueba.csv').set_index('MODELO')
    elegido = config['nombre']

    mlflow.set_tracking_uri(args.tracking)
    experimento = mlflow.set_experiment(EXPERIMENTO)
    # Abrir el experimento en la vista de entrenamiento de modelos (no en la de IA generativa).
    MlflowClient().set_experiment_tag(experimento.experiment_id, 'mlflow.experimentKind', 'custom_model_development')
    MlflowClient().set_experiment_tag(experimento.experiment_id, 'mlflow.note.content', (
        'Pronóstico de casos de dengue en la semana t+4 para cinco municipios del Valle del Cauca. '
        'Cada corrida hija es uno de los 100 enfoques comparados en el notebook; el modelo elegido '
        '(menor MAE de validación) está registrado en Models como pronostico-dengue-t4.'))

    with mlflow.start_run(run_name=f'Comparación de 100 enfoques ({config["seleccion"]["ejecucion"]})') as padre:
        mlflow.log_params({**{k: v for k, v in COMUNES.items() if k != 'objetivo'},
                           'enfoques': len(ranking), 'ajustes_totales': 160,
                           'criterio_seleccion': config['seleccion']['criterio_principal'],
                           'modelo_elegido': elegido})
        mlflow.log_metrics({'mejor_val_mae': float(ranking['MAE'].min()), 'mejor_val_rmse': float(ranking['RMSE'].min())})
        mlflow.set_tags({'mlflow.note.content': (
            'Selección por MAE de validación 2022-2023. El periodo 2024-2025 se evaluó una sola vez al cerrar '
            'la selección para el modelo elegido y las mejores referencias de cada familia.'), 'tipo': 'comparacion'})
        for archivo in sorted(carpeta.glob('*.csv')):
            mlflow.log_artifact(str(archivo), 'tablas')
        for nombre in FIGURAS_GENERALES:
            mlflow.log_artifact(str(figuras / f'{nombre}.png'), 'figuras')

        for _, fila in ranking.iterrows():
            modelo = fila['MODELO']
            with mlflow.start_run(run_name=modelo, nested=True) as corrida:
                mlflow.log_params(parametros(fila))
                metricas = {'val_mae': fila['MAE'], 'val_rmse': fila['RMSE'],
                            'val_pronosticos_negativos': fila['PRONOSTICOS_NEGATIVOS'], 'posicion_ranking': fila['POSICION']}
                entrenamiento = baselines.loc[baselines['MODELO'].eq(modelo) & baselines['CONJUNTO'].eq('training')]
                if not entrenamiento.empty:
                    metricas.update(train_mae=entrenamiento['MAE'].iloc[0], train_rmse=entrenamiento['RMSE'].iloc[0])
                if modelo in tres.index:
                    metricas.update(train_mae=tres.loc[modelo, 'MAE_TRAINING'], train_rmse=tres.loc[modelo, 'RMSE_TRAINING'])
                if modelo in prueba.index:
                    metricas.update(test_mae=prueba.loc[modelo, 'MAE'], test_rmse=prueba.loc[modelo, 'RMSE'],
                                    test_sesgo_medio=prueba.loc[modelo, 'SESGO_MEDIO'])
                mlflow.log_metrics({k: float(v) for k, v in metricas.items()})
                mlflow.set_tags({'familia': fila['FAMILIA'], 'elegido': str(modelo == elegido).lower(),
                                 'evaluado_en_prueba': str(modelo in prueba.index).lower()})
                if modelo == elegido:
                    registrar_elegido(args.modelos, figuras, config, corrida.info.run_id)
    print(f'Registradas {len(ranking)} corridas en el experimento {EXPERIMENTO}.')


def registrar_elegido(modelos, figuras, config, run_id):
    mlflow.set_tag('mlflow.note.content', (
        f'Modelo elegido: {config["nombre"]}. Menor MAE de validación entre los 100 enfoques. '
        'En prueba 2024-2025 (brote mayor que los de entrenamiento) el MAE sube a 38,1 y tiende a subestimar picos.'))
    for figura in sorted(figuras.glob('*.png')):
        if figura.stem not in FIGURAS_GENERALES:
            mlflow.log_artifact(str(figura), 'figuras')
    for archivo in ['modelo.keras', 'preprocesamiento.joblib', 'configuracion.json', 'sha256.json', 'requirements-lock.txt']:
        mlflow.log_artifact(str(modelos / archivo), 'modelo')
    cliente = MlflowClient()
    cliente.create_registered_model(MODELO_REGISTRADO, description=(
        'LSTM global (64 unidades + Dense 16) que pronostica los casos de dengue de la semana t+4 a partir de '
        '20 semanas de casos y clima. Lo sirve la aplicación desplegada (app/pronosticador.py).'))
    version = cliente.create_model_version(
        MODELO_REGISTRADO, source=f'{mlflow.get_artifact_uri()}/modelo', run_id=run_id,
        description=f'Exportado por el notebook (punto 20.4), ejecución {config["seleccion"]["ejecucion"]}. '
                    'Verificación de integridad con sha256.json.')
    cliente.set_registered_model_alias(MODELO_REGISTRADO, 'produccion', version.version)
    for clave, valor in {'historia_semanas': config['historia_semanas'], 'parametros': config['parametros'],
                         'tensorflow': config['versiones']['tensorflow'], 'val_mae': round(config['seleccion']['MAE']['MAE'], 3),
                         'test_mae': round(config['metricas_test']['MAE'], 3)}.items():
        cliente.set_model_version_tag(MODELO_REGISTRADO, version.version, clave, str(valor))


if __name__ == '__main__':
    main()
