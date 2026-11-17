
/***************************************************
 CLIMA SEMANAL MUNICIPAL - DENGUE

 Municipios: Cali, Palmira, Tulua,
 Yumbo y Guadalajara de Buga

 Periodo: 2007-2025
 CHIRPS + ERA5-Land
 Semanas epidemiologicas domingo-sabado
***************************************************/

// ============================================
// 1. CONFIGURACION
// ============================================

var anioInicial = 2007;
var anioFinal = 2025;

var codigos = [
  '76001', // Cali
  '76520', // Palmira
  '76834', // Tulua
  '76892', // Yumbo
  '76111'  // Guadalajara de Buga
];

// ============================================
// 2. MUNICIPIOS DANE
// ============================================

var dane = ee.FeatureCollection(
  'projects/salonbodasdecanan/assets/SHP_MGN2018_INTGRD_MPIO'
);

var municipios = dane.filter(
  ee.Filter.inList('MPIO_CDPMP', codigos)
);

print('Municipios encontrados:', municipios.size());
print('Nombres:', municipios.aggregate_array('MPIO_CNMBR'));

Map.centerObject(municipios, 9);

Map.addLayer(
  municipios.style({
    color: 'red',
    fillColor: '00000000',
    width: 2
  }),
  {},
  'Municipios DANE'
);

// ============================================
// 3. COLECCIONES CLIMATICAS
// ============================================

var chirps = ee.ImageCollection(
  'UCSB-CHG/CHIRPS/DAILY'
).select('precipitation');

var era5 = ee.ImageCollection(
  'ECMWF/ERA5_LAND/DAILY_AGGR'
).select([
  'temperature_2m',
  'temperature_2m_max',
  'temperature_2m_min'
]);

// ============================================
// 4. CALENDARIO EPIDEMIOLOGICO
// ============================================

// Convencion: semana 1 contiene el 4 de
// enero; domingo a sabado.
// Validar con calendario INS del año.

function inicioEpidemiologico(year) {
  var fecha = new Date(Date.UTC(year, 0, 4));

  fecha.setUTCDate(
    fecha.getUTCDate() - fecha.getUTCDay()
  );

  return fecha;
}

function formatoFecha(fecha) {
  return fecha.toISOString().slice(0, 10);
}

var msDia = 24 * 60 * 60 * 1000;
var msSemana = 7 * msDia;

// ============================================
// 5. FUNCION PARA UN AÑO
// ============================================

function generarAnio(anio) {

  var inicio = inicioEpidemiologico(anio);
  var siguiente = inicioEpidemiologico(anio + 1);

  var totalSemanas = Math.round(
    (siguiente - inicio) / msSemana
  );

  var semanas = [];

  for (var i = 0; i < totalSemanas; i++) {

    var fechaInicio = new Date(
      inicio.getTime() + i * msSemana
    );

    var fechaFinExclusiva = new Date(
      fechaInicio.getTime() + msSemana
    );

    var fechaSabado = new Date(
      fechaFinExclusiva.getTime() - msDia
    );

    semanas.push({
      numero: i + 1,
      inicio: formatoFecha(fechaInicio),
      fin: formatoFecha(fechaSabado),
      finExclusivo: formatoFecha(fechaFinExclusiva)
    });
  }

  // Calculos por municipio.
  var colecciones = codigos.map(function(codigo) {

    var municipio = ee.Feature(
      municipios.filter(
        ee.Filter.eq('MPIO_CDPMP', codigo)
      ).first()
    );

    var geometria = municipio.geometry();

    var nombre = municipio.get('MPIO_CNMBR');

    var filas = semanas.map(function(s) {

      // ------------------------------------
      // CHIRPS: lluvia semanal
      // ------------------------------------

      var lluviaDiaria = chirps.filterDate(
        s.inicio, s.finExclusivo
      );

      var lluvia = lluviaDiaria
        .sum()
        .rename('PRECIPITACION_MM');

      var p = lluvia.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geometria,
        scale: 5566,
        maxPixels: 1e9,
        tileScale: 4
      });

      // ------------------------------------
      // ERA5: temperatura semanal
      // ------------------------------------

      var tempDiaria = era5.filterDate(
        s.inicio, s.finExclusivo
      );

      var media = tempDiaria
        .select('temperature_2m')
        .mean()
        .subtract(273.15)
        .rename('TEMP_MEDIA_C');

      var maxima = tempDiaria
        .select('temperature_2m_max')
        .max()
        .subtract(273.15)
        .rename('TEMP_MAX_C');

      var minima = tempDiaria
        .select('temperature_2m_min')
        .min()
        .subtract(273.15)
        .rename('TEMP_MIN_C');

      var temperatura = media
        .addBands(maxima)
        .addBands(minima);

      var t = temperatura.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: geometria,
        scale: 11132,
        maxPixels: 1e9,
        tileScale: 4
      });

      // ------------------------------------
      // REGISTRO
      // ------------------------------------

      return ee.Feature(null, {

        COD_DANE: codigo,
        MUNICIPIO: nombre,

        ANIO: anio,
        SEMANA_EPI: s.numero,

        FECHA_INICIO: s.inicio,
        FECHA_FIN: s.fin,

        PRECIPITACION_MM:
          p.get('PRECIPITACION_MM'),

        TEMP_MEDIA_C:
          t.get('TEMP_MEDIA_C'),

        TEMP_MAX_C:
          t.get('TEMP_MAX_C'),

        TEMP_MIN_C:
          t.get('TEMP_MIN_C'),

        DIAS_CHIRPS: lluviaDiaria.size(),
        DIAS_ERA5: tempDiaria.size()

      });

    });

    return ee.FeatureCollection(filas);
  });

  return ee.FeatureCollection(
    colecciones
  ).flatten();
}

// ============================================
// 6. EXPORTAR CADA AÑO
// ============================================

for (
  var anio = anioInicial;
  anio <= anioFinal;
  anio++
) {

  var resultado = generarAnio(anio);

  var nombreArchivo =
    'CLIMA_VALLE_SIVIGILA_' + anio;

  Export.table.toDrive({
    collection: resultado,

    description: nombreArchivo,

    folder: 'CLIMA_DENGUE_VALLE',

    fileNamePrefix: nombreArchivo,

    fileFormat: 'CSV',

    selectors: [
      'COD_DANE',
      'MUNICIPIO',
      'ANIO',
      'SEMANA_EPI',
      'FECHA_INICIO',
      'FECHA_FIN',
      'PRECIPITACION_MM',
      'TEMP_MEDIA_C',
      'TEMP_MAX_C',
      'TEMP_MIN_C',
      'DIAS_CHIRPS',
      'DIAS_ERA5'
    ]
  });

  print('Exportacion preparada para:', anio);
}

print('Todas las tareas han sido creadas.');
