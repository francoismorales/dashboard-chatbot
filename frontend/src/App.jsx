// src/App.jsx
import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar,  Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ReferenceLine,
} from "recharts";
import { api } from "./api";
import ChatBot from "./ChatBot";
import ForecastChart from "./ForecastChart";
import "./App.css";

// Paleta: rojo-Rappi para crecimiento, azul para decrecimiento
const COLOR_UP = "#FF441F";
const COLOR_DOWN = "#1F6FFF";

const fmtInt = new Intl.NumberFormat("es-CO");
const fmtDec = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 1 });

const fmtDateTime = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("es-CO", {
    day: "2-digit", month: "short",
    hour: "2-digit", minute: "2-digit",
  });
};

const fmtShortTime = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("es-CO", {
    day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit",
  });
};

function KpiCard({ label, value, hint, accent }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value" style={{ color: accent }}>{value}</div>
      {hint && <div className="kpi-hint">{hint}</div>}
    </div>
  );
}

function Section({ title, subtitle, right, children }) {
  return (
    <section className="panel">
      <header className="panel-header">
        <div>
          <h2>{title}</h2>
          {subtitle && <p className="panel-subtitle">{subtitle}</p>}
        </div>
        {right}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

// ---------- Heatmap bidireccional (rojo ↔ azul) ----------
function Heatmap({ data, absMax }) {
  const days = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"];
  const hours = Array.from({ length: 24 }, (_, i) => i);

  const matrix = {};
  data.forEach((d) => {
    if (!matrix[d.dow_num]) matrix[d.dow_num] = {};
    matrix[d.dow_num][d.hour] = d.net_flow;
  });

  // Rojo si positivo (aperturas dominan), azul si negativo (cierres), blanco en el 0
  const colorFor = (val) => {
    if (val == null) return "#f5f5f5";
    const pct = Math.min(1, Math.abs(val) / (absMax || 1));
    if (val >= 0) {
      // Rojo: de blanco a #FF441F
      const r = 255;
      const g = Math.round(255 - (255 - 68) * pct);
      const b = Math.round(255 - (255 - 31) * pct);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Azul: de blanco a #1F6FFF
      const r = Math.round(255 - (255 - 31) * pct);
      const g = Math.round(255 - (255 - 111) * pct);
      const b = 255;
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  const cells = [];
  cells.push(<div key="corner" className="hm-corner" />);
  hours.forEach((h) => cells.push(
    <div key={`h-${h}`} className="hm-hhead">{h}</div>
  ));
  days.forEach((day, dowIdx) => {
    cells.push(<div key={`d-${dowIdx}`} className="hm-dhead">{day}</div>);
    hours.forEach((h) => {
      const val = matrix[dowIdx]?.[h];
      const label = val == null
        ? "sin datos"
        : val >= 0
          ? `+${fmtDec.format(val)} /s · saldo crece`
          : `−${fmtDec.format(Math.abs(val))} /s · saldo decrece`;
      cells.push(
        <div
          key={`c-${dowIdx}-${h}`}
          className="hm-cell"
          style={{ background: colorFor(val) }}
          title={`${day} ${String(h).padStart(2, "0")}:00 — ${label}`}
        />
      );
    });
  });

  return (
    <div className="hm-wrap">
      <div className="hm-grid">{cells}</div>
      <div className="hm-legend">
        <span>Saldo decrece</span>
        <div className="hm-scale-bi" />
        <span>Saldo crece</span>
      </div>
    </div>
  );
}

// ---------- App ----------
export default function App() {
  const [summary, setSummary] = useState(null);
  const [series, setSeries] = useState([]);
  const [granularity, setGranularity] = useState("auto");
  const [peaks, setPeaks] = useState(null);
  const [hourly, setHourly] = useState([]);
  const [daily, setDaily] = useState([]);
  const [heatmap, setHeatmap] = useState(null);
  const [anomalies, setAnomalies] = useState(null);
  const [threshold, setThreshold] = useState(6);
  const [filterHour, setFilterHour] = useState("");
  const [filterDow, setFilterDow] = useState("");
  const [byDate, setByDate] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.summary(),
      api.timeseries({ granularity }),
      api.peaks(10),
      api.hourlyPattern(),
      api.byDate(),
      api.heatmap(),
      api.anomalies({ threshold: 6, topN: 15 }),
    ])
      .then(([s, ts, p, h, bd, hm, an]) => {
        setSummary(s);
        setSeries(ts.data);
        setPeaks(p);
        setHourly(h.data);
        setByDate(bd.data);
        setHeatmap(hm);
        setAnomalies(an);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!summary) return;
    api.timeseries({ granularity }).then((r) => setSeries(r.data));
  }, [granularity]);

  useEffect(() => {
    if (!summary) return;
    api.anomalies({
      threshold,
      topN: 15,
      hour: filterHour === "" ? undefined : Number(filterHour),
      dow:  filterDow  === "" ? undefined : Number(filterDow),
    }).then(setAnomalies);
}, [threshold, filterHour, filterDow]);

  if (loading) return <div className="loading">⏳ Cargando datos…</div>;
  if (error) return <div className="error">❌ {error}<br/>¿Está corriendo el backend en localhost:8000?</div>;

  const metricKey = Object.keys(summary.metrics)[0];
  const m = summary.metrics[metricKey];

  // Cálculo del offset del cero para el gradiente bicolor de la línea principal.
  const flowValues = series.map((r) => r.net_flow ?? 0);
  const flowMax = flowValues.length ? Math.max(...flowValues) : 0;
  const flowMin = flowValues.length ? Math.min(...flowValues) : 0;
  const zeroOffset = flowMax === flowMin
    ? 0.5
    : flowMax / (flowMax - flowMin);

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>📊 Rappi · Disponibilidad de Tiendas</h1>
          <p className="subtitle">
            Flujo neto agregado de tiendas visibles · {fmtDateTime(summary.period.start)} → {fmtDateTime(summary.period.end)}
            {" · "}
            <strong>{fmtInt.format(summary.period.total_points)}</strong> muestras cada 10s · datos disponibles ~18h/día
          </p>
        </div>
      </header>

      {/* -------- KPI Cards (con lenguaje operacional correcto) -------- */}
      <div className="kpi-grid">
        <KpiCard
          label="Último valor observado"
          value={fmtInt.format(m.value_last_observation)}
          hint={`pico del período: ${fmtInt.format(m.value_max)} · ${fmtShortTime(m.value_max_at)}`}
          accent="#0E0B08"
        />
        <KpiCard
          label="Mayor crecimiento neto"
          value={`+${fmtDec.format(m.max_net_growth_per_sec)} /s`}
          hint={fmtShortTime(m.peak_growth_at)}
          accent={COLOR_UP}
        />
        <KpiCard
          label="Mayor decrecimiento neto"
          value={`−${fmtDec.format(m.max_net_attrition_per_sec)} /s`}
          hint={fmtShortTime(m.peak_attrition_at)}
          accent={COLOR_DOWN}
        />
        <KpiCard
          label="Cobertura del dataset"
          value={`${fmtInt.format(summary.period.total_points)} muestras`}
          hint={`${summary.period.duration_hours.toFixed(0)} h · 06h–00h cada día`}
          accent="#0E0B08"
        />
      </div>

      {/* -------- Serie temporal del flujo neto -------- */}
      <Section
        title="Flujo neto de tiendas visibles"
        subtitle="Saldo neto por intervalo · positivo: el saldo crece · negativo: el saldo decrece."
        right={
          <select
            className="selector"
            value={granularity}
            onChange={(e) => setGranularity(e.target.value)}
          >
            <option value="auto">Auto</option>
            <option value="hour">Por hora</option>
            <option value="minute">Por minuto</option>
          </select>
        }
      >
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="netFlowGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor={COLOR_UP} />
                <stop offset={zeroOffset} stopColor={COLOR_UP} />
                <stop offset={zeroOffset} stopColor={COLOR_DOWN} />
                <stop offset="1" stopColor={COLOR_DOWN} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={fmtShortTime}
              minTickGap={50}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              tickFormatter={(v) => fmtInt.format(Math.abs(v))}
              tick={{ fontSize: 11 }}
              width={80}
            />
            <Tooltip
              labelFormatter={fmtShortTime}
              formatter={(v, name) => {
                if (v == null) return ["—", "Flujo neto"];
                const sign = v >= 0 ? "+" : "−";
                return [`${sign}${fmtDec.format(Math.abs(v))} /s`, "Flujo neto"];
              }}
            />
            <ReferenceLine y={0} stroke="#999" strokeWidth={1} />
            <Line
              type="monotone"
              dataKey="net_flow"
              name="Flujo neto"
              stroke="url(#netFlowGradient)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Section>

      {/* -------- Heatmap -------- */}
      <Section
        title="Mapa de calor · Día × Hora"
        subtitle="Saldo neto promedio por día de semana × hora · Rojo: el saldo crece · Azul: el saldo decrece"
      >
        {heatmap && <Heatmap data={heatmap.data} absMax={heatmap.abs_max} />}
      </Section>

      
      <div className="two-col">
        <Section title="Flujo neto promedio por hora del día" subtitle="Saldo neto promedio en cada hora · positivo: crece · negativo: decrece">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={hourly}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmtInt.format(v)} />
              <Tooltip
                formatter={(v) => {
                if (v == null) return ["—", "Flujo neto"];
                const sign = v >= 0 ? "+" : "−";
                return [`${sign}${fmtDec.format(Math.abs(v))} /s`, "Flujo neto"];
              }}
              />
              <ReferenceLine y={0} stroke="#999" />
              <Bar dataKey="avg_net_flow" name="Flujo neto">
                {hourly.map((h, i) => (
                  <Cell key={i} fill={h.avg_net_flow >= 0 ? COLOR_UP : COLOR_DOWN} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>

        <Section title="Balance diario por fecha" subtitle="Flujo neto promedio en cada día del período">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={byDate}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmtInt.format(v)} />
              <Tooltip
               labelFormatter={(label, payload) => {
                  if (!payload?.[0]) return label;
                  return payload[0].payload.date;
                }}
                formatter={(v) => {
                  if (v == null) return ["—", "Flujo neto"];
                  const sign = v >= 0 ? "+" : "−";
                  return [`${sign}${fmtDec.format(Math.abs(v))} /s`, "Flujo neto"];
                }}
              />
              <ReferenceLine y={0} stroke="#999" />
              <Bar dataKey="avg_net_flow" name="Flujo neto">
                {byDate.map((d, i) => (
                  <Cell key={i} fill={d.avg_net_flow >= 0 ? COLOR_UP : COLOR_DOWN} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>
      </div>

      {/* -------- Top picos: 2 tablas lado a lado -------- */}
      <div className="two-col">
        <Section title="🔺 Top 10 picos de crecimiento neto" subtitle="Momentos de mayor saldo positivo en 10s · ⚠ marca posibles glitches de monitoreo">
          <table className="peaks-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th style={{ textAlign: "right" }}>Flujo /s</th>
              </tr>
            </thead>
            <tbody>
              {peaks?.top_growth?.map((p, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    {fmtDateTime(p.timestamp)}
                    {p.is_suspicious && (
                      <span title="Tiene un pico opuesto dentro de ±5 min — probable artefacto de monitoreo" style={{ marginLeft: 6, color: "#F59E0B" }}>⚠</span>
                    )}
                  </td>
                  <td style={{ textAlign: "right", color: COLOR_UP, fontWeight: 600 }}>
                    +{fmtDec.format(p.net_flow)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="🔻 Top 10 picos de decrecimiento neto" subtitle="Momentos de mayor saldo negativo en 10s · ⚠ marca posibles glitches de monitoreo">
          <table className="peaks-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th style={{ textAlign: "right" }}>Flujo /s</th>
              </tr>
            </thead>
            <tbody>
              {peaks?.top_attrition?.map((p, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    {fmtDateTime(p.timestamp)}
                    {p.is_suspicious && (
                      <span title="Tiene un pico opuesto dentro de ±5 min — probable artefacto de monitoreo" style={{ marginLeft: 6, color: "#F59E0B" }}>⚠</span>
                    )}
                  </td>
                  <td style={{ textAlign: "right", color: COLOR_DOWN, fontWeight: 600 }}>
                    {fmtDec.format(p.net_flow)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      </div>

      {/* -------- Anomalías -------- */}
      <Section
        title="🚨 Eventos anómalos detectados"
        subtitle={
          anomalies
            ? `${anomalies.count_filtered ?? anomalies.count} de ${anomalies.count} anomalías visibles (z robusto > ${anomalies.threshold})${
                filterDow !== "" || filterHour !== "" ? " · filtros activos" : ""
              } · ${fmtInt.format(anomalies.total_points)} muestras analizadas`
            : ""
        }
        right={
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <div className="sigma-control">
              <label>Día:</label>
              <select
                className="selector"
                value={filterDow}
                onChange={(e) => setFilterDow(e.target.value)}
              >
                <option value="">Todos</option>
                <option value="0">Lun</option>
                <option value="1">Mar</option>
                <option value="2">Mié</option>
                <option value="3">Jue</option>
                <option value="4">Vie</option>
                <option value="5">Sáb</option>
                <option value="6">Dom</option>
              </select>
            </div>
            <div className="sigma-control">
              <label>Hora:</label>
              <select
                className="selector"
                value={filterHour}
                onChange={(e) => setFilterHour(e.target.value)}
              >
                <option value="">Todas</option>
                {Array.from({ length: 24 }, (_, h) => (
                  <option key={h} value={h}>{String(h).padStart(2, "0")}h</option>
                ))}
              </select>
            </div>
            <div className="sigma-control">
              <label>Umbral z:</label>
              <select
                className="selector"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              >
                <option value="4">4 (más laxo)</option>
                <option value="6">6 (estándar)</option>
                <option value="8">8 (estricto)</option>
                <option value="10">10 (muy estricto)</option>
              </select>
            </div>
          </div>
        }
      >
        {anomalies && anomalies.top.length > 0 ? (
          <table className="peaks-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th style={{ textAlign: "center" }}>Día</th>
                <th style={{ textAlign: "center" }}>Hora</th>
                <th style={{ textAlign: "center" }}>Tipo</th>
                <th style={{ textAlign: "right" }}>Flujo</th>
                <th style={{ textAlign: "right" }}>z</th>
                <th style={{ textAlign: "right" }}>Mediana hora</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.top.map((a, i) => {
                const isGrowth = a.direction === "growth_spike";
                return (
                  <tr key={i}>
                    <td>{i + 1}</td>
                    <td>{fmtDateTime(a.timestamp)}</td>
                    <td style={{ textAlign: "center", color: "#5A5550" }}>{a.dow}</td>
                    <td style={{ textAlign: "center", fontFamily: "ui-monospace, monospace", color: "#5A5550" }}>
                      {String(a.hour).padStart(2, "0")}h
                    </td>
                    <td style={{ textAlign: "center" }}>
                      <span className={`badge ${isGrowth ? "badge-up" : "badge-down"}`}>
                        {isGrowth ? "▲ Crecimiento" : "▼ Decrecimiento"}
                      </span>
                    </td>
                    <td style={{ textAlign: "right", fontWeight: 600 }}>
                      {a.net_flow >= 0 ? "+" : ""}{fmtDec.format(a.net_flow)} /s
                    </td>
                    <td style={{ textAlign: "right", color: a.z_robust > 0 ? COLOR_UP : COLOR_DOWN, fontWeight: 600 }}>
                      {a.z_robust > 0 ? "+" : ""}{fmtDec.format(a.z_robust)}
                    </td>
                    <td style={{ textAlign: "right", color: "#5A5550" }}>
                      {fmtDec.format(a.hour_median)} /s
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <div className="no-data">No hay anomalías con el umbral actual.</div>
        )}
      </Section>

      {/* -------- Predicción 7 días -------- */}
      <ForecastChart />

      <footer className="foot">
        Construido por Francois Morales
      </footer>

      <ChatBot />
    </div>
  );
}
