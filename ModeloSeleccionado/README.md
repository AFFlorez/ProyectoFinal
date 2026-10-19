# Modelo seleccionado por MAE de validación

LSTM 64 · Dense16 · D0,0 · 20s

Ejecuta predecir_ejemplo.py para reproducir cinco pronósticos de validación, uno por municipio. La configuración contiene el tipo de modelo, archivos, historia, variables y orden del codificador. preprocesamiento.joblib conserva los escaladores y el one-hot ajustados en entrenamiento. Los pesos se evaluaron sin volver a ajustar con test. Para nuevas entradas conserva el orden y las transformaciones del notebook.
