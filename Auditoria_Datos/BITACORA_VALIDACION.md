    # Bitácora — Auditoría inicial de los datos de dengue

    Fecha (UTC): 2026-11-04T10:56:37.619247+00:00

    ## Objetivo
    Comprobar que los CSV anuales se pueden utilizar antes de preparar las series del modelo.

    ## Procedimiento
    Se abrió cada CSV entre 2007 y 2025 conservando códigos como texto. Se comprobaron columnas, departamento, municipio, año, rango semanal, fechas vacías, duplicados y huellas SHA-256 del manifiesto.
    No se modificó ni eliminó ningún registro.

    ## Resultados
    - Archivos legibles: 19 de 19.
    - Registros: 290,069.
    - Fechas INI_SIN vacías: 12.
    - Errores estructurales o de integridad: 0.
    - Avisos para revisión: 23.
    - Notas informativas: 19.

    ## Limitaciones
    - Validación estructural: no demuestra exactitud epidemiológica ni correspondencia entre semana y fecha.
    - Semanas de 1 a 53: solo rango; no se verificó el calendario epidemiológico de cada año.
    - Los duplicados exactos se reportan pero no se eliminan: podrían representar casos distintos.
    - Los archivos originales Excel no están en el ZIP: solo se verifica la huella del CSV contra el manifiesto.
    - Una huella idéntica después de normalizar CRLF a LF prueba equivalencia de bytes normalizados, no identidad byte a byte del archivo ZIP.
    - No se emplean datos de test para selección de modelos; este script no entrena ni selecciona modelos.

    ## Archivos de evidencia
    - `validacion_por_ano.csv`: una fila por año con cada comprobación.
    - `incidencias.csv`: avisos y problemas detectados.
    - `resumen_auditoria.json`: resumen de la ejecución.

    ## Decisión
    No se detectaron errores estructurales en los CSV. Se puede continuar a la siguiente validación.

    ##Validacion estructural de la data
    e revisaron los 19 archivos CSV correspondientes a los años 2007–2025, con un total de 290.069 registros.

Todos los archivos pudieron leerse correctamente y no se identificaron errores estructurales.

Se encontraron 12 registros sin fecha de inicio de síntomas (INI_SIN), distribuidos en los años 2010 (4), 2011 (1), 2013 (5) y 2016 (2).

También se identificaron filas con información idéntica. Estas se conservaron porque no es posible confirmar que representen casos duplicados y eliminarlas podría alterar el número de casos registrados.

Por último, se observaron diferencias en los saltos de línea de los archivos (CRLF y LF), sin diferencias en el contenido después de su normalización.
