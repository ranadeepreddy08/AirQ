/* â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
const API = 'http://localhost:8000';
const POLL_LABELS = {
  pm2_5: 'PM2.5 (Âµg/mÂ³)', pm10: 'PM10 (Âµg/mÂ³)',
  nitrogen_dioxide: 'NOâ‚‚ (Âµg/mÂ³)', sulphur_dioxide: 'SOâ‚‚ (Âµg/mÂ³)',
  ozone: 'Oâ‚ƒ (Âµg/mÂ³)', us_aqi: 'US AQI'
};
const POLLUTANTS = Object.keys(POLL_LABELS);

function aqiClass(cat) {
  const m = {
    Good: 'aqi-good', Moderate: 'aqi-moderate',
    'Unhealthy for Sensitive Groups': 'aqi-sensitive',
    'Unhealthy for Sensitive': 'aqi-sensitive',
    Unhealthy: 'aqi-unhealthy', 'Very Unhealthy': 'aqi-very',
    Hazardous: 'aqi-hazardous'
  };
  return m[cat] || 'aqi-unknown';
}

/* â”€â”€ State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
let state = {
  lat: 13.0827, lon: 80.2707,
  mode: 'days',         // 'days' | 'range'
  days: 7,
  startDate: '', endDate: '',
  horizon: 24,
  // fetched data
  data: null,           // raw rows
  trends: null,
  forecasts: {},        // pollutant â†’ forecast response
  issues: null,
  compliance: null,
  recommendations: null,
};
let trendChart = null, forecastChart = null, complianceChart = null;

/* â”€â”€ Map setup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
const map = L.map('map', { zoomControl: true }).setView([state.lat, state.lon], 11);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: 'Â© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 19
}).addTo(map);

let mapMarker = L.marker([state.lat, state.lon], {
  icon: L.divIcon({
    className: '', html: `
    <div style="width:14px;height:14px;background:#5e6ad2;border:2px solid #f7f8f8;
    border-radius:50%;box-shadow:0 2px 8px rgba(94,106,210,0.6)"></div>`,
    iconSize: [14, 14], iconAnchor: [7, 7]
  })
}).addTo(map);

map.on('click', e => {
  state.lat = +e.latlng.lat.toFixed(4);
  state.lon = +e.latlng.lng.toFixed(4);
  updateMarker();
  // clear city select
  document.getElementById('city-select').value = '';
});

function updateMarker() {
  mapMarker.setLatLng([state.lat, state.lon]);
  document.getElementById('coord-badge').textContent =
    `${state.lat.toFixed(4)}Â°N Â· ${state.lon.toFixed(4)}Â°E`;
}

/* â”€â”€ City preset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
document.getElementById('city-select').addEventListener('change', function () {
  const opt = this.options[this.selectedIndex];
  if (!opt.value) return;
  state.lat = +opt.dataset.lat;
  state.lon = +opt.dataset.lon;
  map.setView([state.lat, state.lon], 11);
  updateMarker();
});

/* â”€â”€ Time mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function setMode(m) {
  state.mode = m;
  document.getElementById('ctrl-days').style.display = m === 'days' ? '' : 'none';
  document.getElementById('ctrl-range').style.display = m === 'range' ? '' : 'none';
  document.getElementById('pill-days').classList.toggle('active', m === 'days');
  document.getElementById('pill-range').classList.toggle('active', m === 'range');
}

// Initialise date inputs to today / 14 days ago
(function () {
  const today = new Date();
  const fmt = d => d.toISOString().slice(0, 10);
  const twoWeeks = new Date(today); twoWeeks.setDate(today.getDate() - 14);
  document.getElementById('date-end').value = fmt(today);
  document.getElementById('date-start').value = fmt(twoWeeks);
  document.getElementById('date-end').max = fmt(today);
  document.getElementById('date-start').max = fmt(today);
})();

/* â”€â”€ Tab switching â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
let activeTab = 'overview';
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b => {
    if (b.getAttribute('onclick') === `showTab('${name}')`) b.classList.add('active');
  });
  activeTab = name;
  if (name === 'trends' && state.data) renderTrendChart();
  if (name === 'prediction' && state.data) renderForecastChart();
  if (name === 'issues' && state.data) renderIssuesTab();
}

/* â”€â”€ Banner helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function showLoading(msg) {
  document.getElementById('banner-loading-text').textContent = msg || 'Fetching dataâ€¦';
  document.getElementById('banner-loading').classList.remove('hidden');
  document.getElementById('banner-error').classList.add('hidden');
}
function hideLoading() { document.getElementById('banner-loading').classList.add('hidden'); }
function showError(msg) {
  document.getElementById('banner-error-text').textContent = msg;
  document.getElementById('banner-error').classList.remove('hidden');
  hideLoading();
}

/* â”€â”€ Build query params â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function timeParams() {
  if (state.mode === 'days') {
    state.days = +document.getElementById('days-slider').value;
    return `days=${state.days}`;
  } else {
    const s = document.getElementById('date-start').value;
    const e = document.getElementById('date-end').value;
    if (!s || !e) { showError('Please enter both start and end dates.'); return null; }
    if (s > e) { showError('Start date must be before end date.'); return null; }
    return `start_date=${s}&end_date=${e}`;
  }
}

/* â”€â”€ Main fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
async function fetchAll() {
  const btn = document.getElementById('fetch-btn');
  btn.disabled = true;
  showLoading('Fetching air-quality data from Open-Meteoâ€¦');

  const tp = timeParams();
  if (!tp) { btn.disabled = false; return; }
  state.horizon = +document.getElementById('horizon-select').value;
  const base = `lat=${state.lat}&lon=${state.lon}&${tp}`;

  try {
    // Fetch data + trends + issues + compliance + recommendations in parallel
    const [dataRes, trendsRes, issuesRes, complianceRes, recsRes] = await Promise.all([
      apiFetch(`/api/data?${base}`),
      apiFetch(`/api/trends?${base}`),
      apiFetch(`/api/issues?lat=${state.lat}&lon=${state.lon}`),
      apiFetch(`/api/compliance?${base}`),
      apiFetch(`/api/recommendations?${base}`),
    ]);

    if (!dataRes.ok) throw new Error(dataRes.error || 'Data fetch failed');
    if (!trendsRes.ok) throw new Error(trendsRes.error || 'Trends failed');
    if (!complianceRes.ok) throw new Error(complianceRes.error || 'Compliance failed');

    state.data = dataRes;
    state.trends = trendsRes;
    state.issues = issuesRes;
    state.compliance = complianceRes;
    state.recommendations = recsRes;
    state.forecasts = {};  // clear cached forecasts

    hideLoading();
    renderOverview();
    if (activeTab === 'trends') renderTrendChart();
    if (activeTab === 'prediction') renderForecastChart();
    if (activeTab === 'issues') renderIssuesTab();

  } catch (err) {
    showError(`Error: ${err.message}. Check internet connection or try again.`);
  } finally {
    btn.disabled = false;
  }
}

async function apiFetch(path) {
  const r = await fetch(API + path);
  if (!r.ok && r.status !== 404) {
    const t = await r.text();
    throw new Error(`HTTP ${r.status}: ${t.slice(0, 120)}`);
  }
  return r.json();
}

/* â”€â”€ OVERVIEW TAB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function renderOverview() {
  const t = state.trends;
  if (!t) return;

  // Hero AQI
  const aqi = t.current_aqi;
  const cat = t.aqi_category || 'Unknown';
  document.getElementById('hero-aqi-number').textContent = aqi !== null ? Math.round(aqi) : 'â€”';
  const badge = document.getElementById('hero-aqi-badge');
  badge.textContent = cat;
  badge.className = 'aqi-badge ' + aqiClass(cat);
  document.getElementById('hero-coord').textContent =
    `${state.lat.toFixed(4)}Â°N Â· ${state.lon.toFixed(4)}Â°E`;

  // Needle (0-500 scale mapped to 0-100%)
  if (aqi !== null) {
    const pct = Math.min(100, (aqi / 500) * 100);
    document.getElementById('aqi-needle').style.left = pct + '%';
  }

  // Pollutant cards
  const grid = document.getElementById('pollutant-grid');
  grid.innerHTML = '';
  for (const pol of POLLUTANTS) {
    if (pol === 'us_aqi') continue;
    const s = t.trends?.[pol] || {};
    const mean = s.mean;
    const label = POLL_LABELS[pol];
    const parts = label.split('(');
    const name = parts[0].trim();
    const unit = parts[1] ? parts[1].replace(')', '').trim() : '';

    const trendClass = s.trend?.includes('+') ? 'trend-rising' :
      s.trend?.includes('-') ? 'trend-falling' : 'trend-stable';
    const trendIcon = s.trend?.includes('+') ? 'â†‘' :
      s.trend?.includes('-') ? 'â†“' : 'â†’';

    const card = document.createElement('div');
    card.className = 'pol-card';
    if (mean !== null && mean !== undefined) {
      card.innerHTML = `
        <div class="pol-card-label">${name}</div>
        <div class="pol-card-value">${mean.toFixed(1)}<span class="pol-card-unit">${unit}</span></div>
        <div class="pol-card-peak">Peak: ${(s.peak || 0).toFixed(1)} ${unit}</div>
        <div class="pol-card-trend ${trendClass}">${trendIcon} ${s.trend || ''}</div>`;
    } else {
      card.innerHTML = `
        <div class="pol-card-label">${name}</div>
        <div style="margin-top:6px"><span class="na-badge">Not available</span></div>`;
    }
    grid.appendChild(card);
  }

  // Recent 24 rows table
  buildReadingsTable();

  // Render other tabs if active
  if (activeTab === 'issues') renderIssuesTab();
}

function buildReadingsTable() {
  const rows = (state.data?.data || []).slice(-24);
  const head = document.getElementById('readings-head');
  const body = document.getElementById('readings-body');

  const cols = ['time', ...POLLUTANTS];
  head.innerHTML = cols.map(c => `<th>${POLL_LABELS[c] || 'Time'}</th>`).join('');
  body.innerHTML = rows.map(r =>
    '<tr>' + cols.map(c => {
      if (c === 'time') return `<td>${r.time ? r.time.slice(0, 16).replace('T', ' ') : 'â€”'}</td>`;
      const v = r[c];
      return v !== null && v !== undefined
        ? `<td>${(+v).toFixed(1)}</td>`
        : `<td class="table-na">N/A</td>`;
    }).join('') + '</tr>'
  ).join('');
}

/* â”€â”€ TRENDS TAB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function renderTrendChart() {
  const pol = document.getElementById('trend-pol-select').value;
  const rows = state.data?.data || [];
  const valid = rows.filter(r => r[pol] !== null && r[pol] !== undefined);

  if (trendChart) { trendChart.destroy(); trendChart = null; }

  const ctx = document.getElementById('trend-chart').getContext('2d');

  if (!valid.length) {
    // draw empty state message
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = '#8a8f98';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('No data available for this pollutant', ctx.canvas.width / 2, 80);
    document.getElementById('trend-stats').innerHTML = '';
    return;
  }

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: valid.map(r => r.time?.slice(0, 16).replace('T', ' ')),
      datasets: [{
        label: POLL_LABELS[pol],
        data: valid.map(r => r[pol]),
        borderColor: '#5e6ad2',
        backgroundColor: 'rgba(94,106,210,0.08)',
        borderWidth: 2,
        pointRadius: valid.length > 100 ? 0 : 2,
        fill: true, tension: 0.3,
      }]
    },
    options: chartOptions(POLL_LABELS[pol])
  });

  // Stat pills
  const s = state.trends?.trends?.[pol] || {};
  const pills = document.getElementById('trend-stats');
  if (s.mean !== null && s.mean !== undefined) {
    pills.innerHTML = `
      <div class="stat-pill">Mean <strong>${s.mean.toFixed(1)}</strong></div>
      <div class="stat-pill">Peak <strong>${(s.peak || 0).toFixed(1)}</strong></div>
      <div class="stat-pill">Peak time <strong>${s.peak_time ? s.peak_time.slice(0, 16).replace('T', ' ') : 'â€”'}</strong></div>
      <div class="stat-pill">Trend <strong>${s.trend || 'â€”'}</strong></div>`;
  } else {
    pills.innerHTML = '<div class="stat-pill" style="color:var(--ink-subtle)">No trend data available</div>';
  }
}

/* â”€â”€ PREDICTION TAB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
async function renderForecastChart() {
  const pol = document.getElementById('pred-pol-select').value;
  const tp = timeParams();
  if (!tp || !state.data) return;

  document.getElementById('pred-loading').classList.remove('hidden');
  document.getElementById('pred-error').classList.add('hidden');
  document.getElementById('mae-card').style.display = 'none';
  if (forecastChart) { forecastChart.destroy(); forecastChart = null; }

  const base = `lat=${state.lat}&lon=${state.lon}&${tp}`;

  try {
    // Use cached forecast if available
    if (!state.forecasts[pol]) {
      state.forecasts[pol] = await apiFetch(
        `/api/forecast?${base}&pollutant=${pol}&horizon=${state.horizon}`
      );
    }
    const res = state.forecasts[pol];
    document.getElementById('pred-loading').classList.add('hidden');

    if (res.error && !res.forecast?.length) {
      document.getElementById('pred-error-text').textContent = res.error;
      document.getElementById('pred-error').classList.remove('hidden');
      return;
    }

    // History
    const histRows = (state.data?.data || []).filter(r =>
      r[pol] !== null && r[pol] !== undefined);
    const histLabels = histRows.map(r => r.time?.slice(0, 16).replace('T', ' '));
    const histVals = histRows.map(r => r[pol]);

    // Forecast
    const fcastRows = res.forecast || [];
    const fcastLabels = fcastRows.map(r => (r.time || '').slice(0, 16).replace('T', ' '));
    const fcastVals = fcastRows.map(r => r.predicted);

    const allLabels = [...histLabels, ...fcastLabels];
    const histData = [...histVals, ...new Array(fcastRows.length).fill(null)];
    const fcastData = [...new Array(histRows.length).fill(null), ...fcastVals];

    const ctx = document.getElementById('forecast-chart').getContext('2d');
    forecastChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'History',
            data: histData,
            borderColor: '#5e6ad2',
            backgroundColor: 'rgba(94,106,210,0.06)',
            borderWidth: 1.8,
            pointRadius: histRows.length > 100 ? 0 : 2,
            fill: true, tension: 0.3, spanGaps: false,
          },
          {
            label: `Forecast (${state.horizon}h)`,
            data: fcastData,
            borderColor: '#e05c5c',
            backgroundColor: 'rgba(224,92,92,0.08)',
            borderWidth: 2.2,
            borderDash: [6, 3],
            pointRadius: 4,
            pointBackgroundColor: '#e05c5c',
            fill: true, tension: 0.3, spanGaps: false,
          }
        ]
      },
      options: chartOptions(POLL_LABELS[pol])
    });

    // MAE
    if (res.mae !== null && res.mae !== undefined) {
      document.getElementById('mae-value').textContent = res.mae.toFixed(2);
      const unit = (POLL_LABELS[pol] || '').match(/\(([^)]+)\)/)?.[1] || '';
      document.getElementById('mae-unit').textContent = unit;
      document.getElementById('mae-card').style.display = 'inline-flex';
    }

  } catch (err) {
    document.getElementById('pred-loading').classList.add('hidden');
    document.getElementById('pred-error-text').textContent = `Could not load forecast: ${err.message}`;
    document.getElementById('pred-error').classList.remove('hidden');
  }
}

/* â”€â”€ ISSUES TAB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function renderIssuesTab() {
  // Issues
  const iss = state.issues;
  if (iss) {
    document.getElementById('issues-headline').textContent = iss.headline || '';
    const autoBadge = document.getElementById('issues-auto-badge');
    autoBadge.classList.toggle('hidden', !iss.auto_geocoded);
    const list = document.getElementById('issues-list');
    list.innerHTML = (iss.issues || []).map(i =>
      `<li>${i.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')}</li>`
    ).join('');
  }

  // Recommendations
  const recs = state.recommendations;
  if (recs) {
    const aqi = recs.current_aqi;
    const cat = recs.aqi_category || 'Unknown';
    const cls = aqiClass(cat);
    document.getElementById('rec-aqi-line').innerHTML =
      `Current AQI: <strong>${aqi !== null ? Math.round(aqi) : 'â€”'}</strong>
       &nbsp;<span class="aqi-badge ${cls}" style="font-size:11px">${cat}</span>`;
    document.getElementById('rec-exposure').innerHTML =
      (recs.exposure || []).map(t => `<li>${t}</li>`).join('');
    document.getElementById('rec-improve').innerHTML =
      (recs.improvement || []).map(t => `<li>${t}</li>`).join('');
  }

  renderComplianceChart();
}

/* â”€â”€ COMPLIANCE CHART â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function renderComplianceChart() {
  const rows = (state.compliance?.rows || []).filter(r =>
    typeof r['% Hours Exceeding'] === 'number');

  if (complianceChart) { complianceChart.destroy(); complianceChart = null; }
  if (!rows.length) return;

  const labels = rows.map(r => `${r.Pollutant} (${r.Standard})`);
  const vals = rows.map(r => r['% Hours Exceeding']);
  const colors = vals.map(v => v > 50 ? '#e05c5c' : v > 20 ? '#d97706' : '#5e6ad2');

  const ctx = document.getElementById('compliance-chart').getContext('2d');
  complianceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: '% Hours Exceeding',
        data: vals,
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      ...chartOptions('% Hours Exceeding'),
      plugins: {
        ...basePlugins(),
        datalabels: undefined,
      },
      scales: {
        x: xAxis({ rotation: -30, maxRotation: -30 }),
        y: yAxis({ max: Math.max(100, ...vals) + 10 }),
      }
    }
  });
}

/* â”€â”€ Chart.js base helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
function basePlugins() {
  return {
    legend: {
      labels: { color: '#d0d6e0', font: { size: 12, family: 'Inter' }, boxWidth: 12 }
    },
    tooltip: {
      backgroundColor: '#141516', borderColor: '#5e6ad2', borderWidth: 1,
      titleColor: '#f7f8f8', bodyColor: '#d0d6e0',
      titleFont: { size: 12, family: 'Inter' },
      bodyFont: { size: 12, family: 'Inter' },
    }
  };
}
function xAxis(extra = {}) {
  return {
    ticks: { color: '#8a8f98', font: { size: 11, family: 'Inter' }, maxTicksLimit: 8, ...extra },
    grid: { color: '#23252a' },
    border: { color: '#23252a' },
  };
}
function yAxis(extra = {}) {
  return {
    ticks: { color: '#8a8f98', font: { size: 11, family: 'Inter' } },
    grid: { color: '#23252a' },
    border: { color: '#23252a' },
    ...extra,
  };
}
function chartOptions(yLabel) {
  return {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: basePlugins(),
    scales: {
      x: xAxis(),
      y: { ...yAxis(), title: { display: true, text: yLabel, color: '#8a8f98', font: { size: 11 } } }
    },
    interaction: { intersect: false, mode: 'index' },
  };
}

/* â”€â”€ Initial load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
window.addEventListener('load', () => {
  // Ensure Leaflet tiles render after layout settles
  setTimeout(() => map.invalidateSize(), 300);
  // Wire up poll selects to re-render on tab switch
  document.getElementById('trend-pol-select').addEventListener('change', renderTrendChart);
  document.getElementById('pred-pol-select').addEventListener('change', renderForecastChart);
  fetchAll();
});

/* ── Chat ──────────────────────────────────────────────────────────────── */
const chatHistory = [];   // [{role:'user'|'model', content:str}]

function renderMarkdown(src) {
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const inline = s => esc(s)
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
  const cells = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
  const lines = src.split('\n');
  let html = '', inList = false, i = 0;
  const closeList = () => { if (inList) { html += '</ul>'; inList = false; } };

  while (i < lines.length) {
    const line = lines[i];
    if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      closeList();
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|')) { rows.push(cells(lines[i])); i++; }
      html += '<table><thead><tr>' + head.map(h => `<th>${inline(h)}</th>`).join('') +
        '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + r.map(c => `<td>${inline(c)}</td>`).join('') + '</tr>').join('') +
        '</tbody></table>';
      continue;
    }
    const li = line.match(/^\s*[-*•]\s+(.*)/);
    if (li) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inline(li[1])}</li>`; i++; continue;
    }
    closeList();
    if (line.trim() === '---') html += '<hr>';
    else if (line.trim()) {
      const h = line.match(/^#{1,4}\s+(.*)/);
      html += h ? `<p><strong>${inline(h[1])}</strong></p>` : `<p>${inline(line)}</p>`;
    }
    i++;
  }
  closeList();
  return html;
}

function appendChatMessage(role, text) {
  const list = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = `chat-msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  if (role === 'model') bubble.innerHTML = renderMarkdown(text);   // HTML is escaped first
  else bubble.textContent = text;
  row.appendChild(bubble);
  list.appendChild(row);
  list.scrollTop = list.scrollHeight;
}

async function sendChat() {
  const input   = document.getElementById('chat-input');
  const btn     = document.getElementById('chat-send-btn');
  const typing  = document.getElementById('chat-typing');
  const question = input.value.trim();
  if (!question) return;

  appendChatMessage('user', question);
  chatHistory.push({ role: 'user', content: question });
  input.value = '';
  btn.disabled = true;
  typing.classList.remove('hidden');
  document.getElementById('chat-messages').scrollTop = 9999;

  // Exact UI values — no fallbacks that mask missing data
  const isRange  = document.getElementById('pill-range').classList.contains('active');
  const days     = isRange ? null : parseInt(document.getElementById('days-slider').value, 10);
  const startDate = isRange ? document.getElementById('date-start').value || null : null;
  const endDate   = isRange ? document.getElementById('date-end').value   || null : null;
  const cityEl   = document.getElementById('city-select');
  const city     = cityEl.value || null;

  const payload = {
    question,
    lat:        state.lat,
    lon:        state.lon,
    days,
    start_date: startDate,
    end_date:   endDate,
    city,
    history:    chatHistory.slice(-11, -1),   // last 10 turns (excluding current)
  };

  try {
    const res  = await fetch(`${API}/api/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    typing.classList.add('hidden');
    const answer = data.answer || '⚠️ No answer received.';
    appendChatMessage('model', answer);
    chatHistory.push({ role: 'model', content: answer });
  } catch (err) {
    typing.classList.add('hidden');
    const errMsg = `⚠️ Could not reach the API. Is the server running? (${err.message})`;
    appendChatMessage('model', errMsg);
    chatHistory.push({ role: 'model', content: errMsg });
  }

  btn.disabled = false;
  input.focus();
}

function chatSuggest(text) {
  document.getElementById('chat-input').value = text;
  sendChat();
}