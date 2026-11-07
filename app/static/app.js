'use strict';

const graficos = {};
const COLORES = { casos: '#94a3b8', posteriores: '#cbd5e1', pronostico: '#4f46e5', real: '#0f766e', lluvia: '#38bdf8', temp: '#f97316' };
const $ = (id) => document.getElementById(id);
const formato = new Intl.NumberFormat('es-CO', { maximumFractionDigits: 1 });
const fechaCorta = (iso) => new Date(iso + 'T00:00:00').toLocaleDateString('es-CO', { day: 'numeric', month: 'short', year: 'numeric' });
const semanaTexto = (s) => `${s.ano}-S${String(s.semana).padStart(2, '0')}`;
const rangoTexto = (s) => `${fechaCorta(s.fecha_inicio)} – ${fechaCorta(s.fecha_fin)}`;
const insignia = (nivel) => `<span class="insignia ${nivel}">${nivel}</span>`;

async function api(ruta, opciones) {
  const respuesta = await fetch(ruta, opciones);
  const cuerpo = await respuesta.json().catch(() => ({}));
  if (!respuesta.ok) {
    let detalle = cuerpo.detail ?? `Error ${respuesta.status}`;
    if (Array.isArray(detalle)) detalle = detalle.map((d) => d.msg).join('; ');
    throw new Error(detalle);
  }
  return cuerpo;
}

function avisar(mensaje, tipo = 'error') {
  const caja = $('aviso-global');
  caja.textContent = mensaje;
  caja.className = `aviso ${tipo}`;
  caja.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
const limpiarAviso = () => $('aviso-global').classList.add('oculto');

async function conBoton(boton, tarea) {
  limpiarAviso();
  boton.disabled = true;
  try { await tarea(); } catch (error) { avisar(error.message); } finally { boton.disabled = false; }
}

function dibujar(id, configuracion) {
  if (!window.Chart) return;
  graficos[id]?.destroy();
  graficos[id] = new Chart($(id), configuracion);
}

// ---- Pestañas -------------------------------------------------------------
document.querySelectorAll('.pestana').forEach((boton) => boton.addEventListener('click', () => {
  document.querySelectorAll('.pestana').forEach((b) => { b.classList.toggle('activa', b === boton); b.setAttribute('aria-selected', b === boton); });
  document.querySelectorAll('.seccion').forEach((s) => s.classList.toggle('activa', s.id === boton.dataset.seccion));
  limpiarAviso();
}));

// ---- 1. Pronóstico --------------------------------------------------------
let municipios = [];

function actualizarRango() {
  const m = municipios.find((x) => x.codigo === $('p-municipio').value);
  if (m?.primera_emision) {
    $('p-rango').textContent = `Semanas de emisión disponibles para ${m.nombre}: de ${semanaTexto(m.primera_emision)} a ${semanaTexto(m.ultima_emision)}.`;
  }
}

async function pronosticar(ano, semana) {
  const codigo = $('p-municipio').value;
  const consulta = new URLSearchParams({ codigo });
  if (ano && semana) { consulta.set('ano', ano); consulta.set('semana', semana); }
  const r = await api(`/api/pronostico?${consulta}`);
  $('p-ano').value = r.emision.ano;
  $('p-semana').value = r.emision.semana;

  $('r-casos').textContent = `≈ ${r.casos_predichos_redondeados}`;
  $('r-objetivo').textContent = `${r.municipio.nombre} · semana ${semanaTexto(r.objetivo)} (${rangoTexto(r.objetivo)})`;
  $('r-alerta').textContent = r.nivel_alerta;
  $('r-alerta').className = `insignia ${r.nivel_alerta}`;
  $('r-umbrales').textContent = `Moderado desde ${formato.format(r.umbrales_alerta.moderado)} casos; alto desde ${formato.format(r.umbrales_alerta.alto)}.`;
  if (r.casos_reales === null) {
    $('r-reales').textContent = 'Aún no disponible';
    $('r-error').textContent = 'La semana objetivo está fuera de los datos cargados.';
  } else {
    $('r-reales').textContent = r.casos_reales;
    $('r-error').textContent = `Error absoluto: ${formato.format(r.error_absoluto)} casos (pronóstico sin redondear: ${formato.format(r.casos_predichos)}).`;
  }
  $('r-conjunto').textContent = `Emitido con datos hasta la semana ${semanaTexto(r.emision)} (${rangoTexto(r.emision)}). `
    + `Periodo de la semana objetivo: ${r.conjunto}.`;
  $('p-resultado').classList.remove('oculto');
  graficarPronostico(r);
  await cargarResumen(r.emision.ano, r.emision.semana);
}

function graficarPronostico(r) {
  const etiquetas = [...r.historia, ...r.semanas_posteriores].map(semanaTexto);
  const objetivo = semanaTexto(r.objetivo);
  if (!etiquetas.includes(objetivo)) {
    for (let k = r.semanas_posteriores.length + 1; k <= r.horizonte_semanas; k += 1) etiquetas.push(k === r.horizonte_semanas ? objetivo : `t+${k}`);
  }
  const relleno = (n) => Array(n).fill(null);
  const n = r.historia.length;
  const posteriores = r.semanas_posteriores.map((s) => s.casos);
  const punto = relleno(etiquetas.length); punto[etiquetas.indexOf(objetivo)] = r.casos_predichos;
  dibujar('g-casos', {
    data: {
      labels: etiquetas,
      datasets: [
        { type: 'bar', label: 'Casos observados (entrada)', data: [...r.historia.map((s) => s.casos), ...relleno(etiquetas.length - n)], backgroundColor: COLORES.casos },
        { type: 'bar', label: 'Casos reales posteriores', data: [...relleno(n), ...posteriores], backgroundColor: COLORES.posteriores },
        { type: 'line', label: 'Pronóstico t+4', data: punto, borderColor: COLORES.pronostico, backgroundColor: COLORES.pronostico, pointRadius: 8, pointStyle: 'rectRot', showLine: false },
      ],
    },
    options: { maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, scales: { y: { beginAtZero: true, title: { display: true, text: 'Casos' } } } },
  });
  dibujar('g-clima', {
    data: {
      labels: r.historia.map(semanaTexto),
      datasets: [
        { type: 'bar', label: 'Precipitación (mm)', data: r.historia.map((s) => s.precipitacion_mm), backgroundColor: COLORES.lluvia, yAxisID: 'lluvia' },
        { type: 'line', label: 'Temperatura media (°C)', data: r.historia.map((s) => s.temp_media_c), borderColor: COLORES.temp, backgroundColor: COLORES.temp, yAxisID: 'temp', tension: 0.3 },
      ],
    },
    options: {
      maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      scales: { lluvia: { position: 'left', beginAtZero: true, title: { display: true, text: 'mm' } }, temp: { position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: '°C' } } },
    },
  });
}

async function cargarResumen(ano, semana) {
  const { resultados } = await api(`/api/resumen?${new URLSearchParams({ ano, semana })}`);
  $('t-resumen').querySelector('tbody').innerHTML = resultados.map((r) => `<tr>
    <td>${r.municipio.nombre}</td><td>${semanaTexto(r.objetivo)}</td>
    <td class="num">${r.casos_predichos_redondeados}</td><td>${insignia(r.nivel_alerta)}</td>
    <td class="num">${r.casos_reales ?? '—'}</td><td class="num">${r.error_absoluto === null ? '—' : formato.format(r.error_absoluto)}</td></tr>`).join('');
}

$('form-pronostico').addEventListener('submit', (e) => {
  e.preventDefault();
  conBoton(e.submitter, () => pronosticar($('p-ano').value, $('p-semana').value));
});
$('p-ultima').addEventListener('click', (e) => conBoton(e.currentTarget, () => pronosticar()));
$('p-municipio').addEventListener('change', actualizarRango);

// ---- 2. Desempeño ---------------------------------------------------------
$('form-desempeno').addEventListener('submit', (e) => {
  e.preventDefault();
  conBoton(e.submitter, async () => {
    const consulta = new URLSearchParams({ codigo: $('d-municipio').value, desde: $('d-desde').value, hasta: $('d-hasta').value });
    const r = await api(`/api/desempeno?${consulta}`);
    dibujar('g-desempeno', {
      type: 'line',
      data: {
        labels: r.serie.map(semanaTexto),
        datasets: [
          { label: 'Casos reales', data: r.serie.map((s) => s.real), borderColor: COLORES.real, backgroundColor: COLORES.real, pointRadius: 0, borderWidth: 2 },
          { label: 'Pronóstico (4 semanas antes)', data: r.serie.map((s) => s.predicho), borderColor: COLORES.pronostico, backgroundColor: COLORES.pronostico, pointRadius: 0, borderWidth: 2, borderDash: [6, 3] },
        ],
      },
      options: { maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, scales: { y: { beginAtZero: true, title: { display: true, text: 'Casos semanales' } }, x: { ticks: { maxTicksLimit: 14 } } } },
    });
    $('t-metricas').querySelector('tbody').innerHTML = Object.entries(r.metricas).map(([periodo, m]) => `<tr>
      <td>${periodo === 'Total' ? '<strong>Total del rango</strong>' : periodo}</td><td class="num">${m.n}</td>
      <td class="num">${formato.format(m.mae)}</td><td class="num">${formato.format(m.rmse)}</td><td class="num">${formato.format(m.sesgo_medio)}</td></tr>`).join('');
    $('d-resultado').classList.remove('oculto');
  });
});

// ---- 3. Datos propios -----------------------------------------------------
$('form-archivo').addEventListener('submit', (e) => {
  e.preventDefault();
  conBoton(e.submitter, async () => {
    const datos = new FormData();
    datos.append('archivo', $('archivo').files[0]);
    const { resultados } = await api('/api/pronostico/archivo', { method: 'POST', body: datos });
    $('t-archivo').querySelector('tbody').innerHTML = resultados.map((r) => `<tr>
      <td>${r.municipio.nombre}</td><td>${semanaTexto(r.emision)} (${rangoTexto(r.emision)})</td>
      <td>${semanaTexto(r.objetivo)} (${rangoTexto(r.objetivo)})</td>
      <td class="num"><strong>${r.casos_predichos_redondeados}</strong></td><td>${insignia(r.nivel_alerta)}</td></tr>`).join('');
    const paleta = ['#0f766e', '#4f46e5', '#f97316', '#db2777', '#65a30d'];
    const etiquetas = [...resultados[0].historia.map((_, i) => `t-${resultados[0].historia.length - 1 - i}`).map((t) => (t === 't-0' ? 't' : t)), 't+1', 't+2', 't+3', 't+4'];
    dibujar('g-archivo', {
      type: 'line',
      data: {
        labels: etiquetas,
        datasets: resultados.flatMap((r, i) => [
          { label: r.municipio.nombre, data: [...r.historia.map((s) => s.casos), null, null, null, null], borderColor: paleta[i % 5], backgroundColor: paleta[i % 5], pointRadius: 2 },
          { label: `${r.municipio.nombre} (pronóstico)`, data: [...Array(r.historia.length + 3).fill(null), r.casos_predichos], borderColor: paleta[i % 5], backgroundColor: paleta[i % 5], pointRadius: 7, pointStyle: 'rectRot', showLine: false },
        ]),
      },
      options: { maintainAspectRatio: false, interaction: { mode: 'nearest', intersect: false }, scales: { y: { beginAtZero: true, title: { display: true, text: 'Casos' } } },
        plugins: { legend: { labels: { filter: (item) => !item.text.includes('(pronóstico)') } } } },
    });
    $('a-resultado').classList.remove('oculto');
  });
});

// ---- 4. Modelo ------------------------------------------------------------
async function cargarModelo() {
  const m = await api('/api/modelo');
  const ficha = {
    'Modelo elegido': m.nombre, 'Parámetros entrenables': m.parametros.toLocaleString('es-CO'),
    'Historia de entrada': `${m.historia_semanas} semanas`, 'Horizonte': `${m.horizonte_semanas} semanas (t+4)`,
    'Criterio de selección': m.criterio_seleccion, 'TensorFlow / Keras': `${m.versiones.tensorflow} / ${m.versiones.keras}`,
  };
  $('m-ficha').innerHTML = Object.entries(ficha).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  const filas = { 'Validación 2022-2023': m.metricas.validacion_2022_2023, 'Prueba 2024-2025': m.metricas.prueba_2024_2025 };
  $('t-modelo').querySelector('tbody').innerHTML = Object.entries(filas).map(([k, v]) => `<tr><td>${k}</td><td class="num">${v.n}</td>
    <td class="num">${formato.format(v.mae)}</td><td class="num">${formato.format(v.rmse)}</td></tr>`).join('');
}

// ---- Inicio -----------------------------------------------------------------
async function iniciar() {
  try {
    const [conf, lista] = await Promise.all([api('/api/configuracion'), api('/api/municipios')]);
    $('titulo').textContent = conf.titulo;
    document.querySelectorAll('.p-mod').forEach((e) => { e.textContent = conf.percentil_alerta_moderada; });
    document.querySelectorAll('.p-alto').forEach((e) => { e.textContent = conf.percentil_alerta_alta; });
    municipios = lista.municipios;
    const opciones = municipios.map((m) => `<option value="${m.codigo}">${m.nombre} (${m.codigo})</option>`).join('');
    $('p-municipio').innerHTML = opciones;
    $('d-municipio').innerHTML = opciones;
    cargarModelo().catch((e) => avisar(e.message));
    if (lista.historico_disponible) {
      actualizarRango();
      await pronosticar();
    } else {
      avisar(`${lista.aviso} Usa la pestaña «3. Usar mis datos».`, 'info');
      document.querySelectorAll('#form-pronostico button, #form-desempeno button').forEach((b) => { b.disabled = true; });
    }
  } catch (error) {
    avisar(`No fue posible conectar con el servicio: ${error.message}`);
  }
}
iniciar();
