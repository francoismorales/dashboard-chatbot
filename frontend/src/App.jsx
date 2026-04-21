// src/App.jsx
import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ReferenceLine,
} from "recharts";
import { api } from "./api";
import ChatBot from "./ChatBot";
import "./App.css";

// Paleta: rojo-Rappi para aperturas, azul para cierres
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
          ? `+${fmtDec.format(val)} aperturas/s neto`
          : `−${fmtDec.format(Math.abs(val))} cierres/s neto`;
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
        <span>Cierres dominan</span>
        <div className="hm-scale-bi" />
        <span>Aperturas dominan</span>
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
  const [sigma, setSigma] = useState(3);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.summary(),
      api.timeseries({ granularity }),
      api.peaks(10),
      api.hourlyPattern(),
      api.dailyPattern(),
      api.heatmap(),
      api.anomalies({ sigma: 3, topN: 15 }),
    ])
      .then(([s, ts, p, h, d, hm, an]) => {
        setSummary(s);
        setSeries(ts.data);
        setPeaks(p);
        setHourly(h.data);
        setDaily(d.data);
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
    api.anomalies({ sigma, topN: 15 }).then(setAnomalies);
  }, [sigma]);

  if (loading) return <div className="loading">⏳ Cargando datos…</div>;
  if (error) return <div className="error">❌ {error}<br/>¿Está corriendo el backend en localhost:8000?</div>;

  const metricKey = Object.keys(summary.metrics)[0];
  const m = summary.metrics[metricKey];

  // Para el gráfico bidireccional: invertimos rate_down para que vaya hacia abajo
  const bidirectionalSeries = series.map((r) => ({
    ...r,
    rate_down_neg: r.rate_down != null ? -r.rate_down : null,
  }));

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>📊 Rappi · Disponibilidad de Tiendas</h1>
          <p className="subtitle">
            Registro de cambios online/offline · {fmtDateTime(summary.period.start)} → {fmtDateTime(summary.period.end)}
            {" · "}
            <strong>{fmtInt.format(summary.period.total_points)}</strong> muestras cada 10s
          </p>
        </div>
      </header>

      {/* -------- KPI Cards (con lenguaje operacional correcto) -------- */}
      <div className="kpi-grid">
        <KpiCard
          label="Tiendas online (último registro)"
          value={fmtInt.format(m.value_end)}
          hint={`desde ${fmtInt.format(m.value_start)} al inicio del período`}
          accent="#0E0B08"
        />
        <KpiCard
          label="Mayor apertura masiva"
          value={`+${fmtInt.format(m.max_rate_up)}`}
          hint={`tiendas/seg · ${fmtShortTime(m.peak_up_at)}`}
          accent={COLOR_UP}
        />
        <KpiCard
          label="Mayor cierre masivo"
          value={`−${fmtInt.format(m.max_rate_down)}`}
          hint={`tiendas/seg · ${fmtShortTime(m.peak_down_at)}`}
          accent={COLOR_DOWN}
        />
        <KpiCard
          label="Eventos registrados"
          value={`${fmtInt.format(m.n_up_points)} / ${fmtInt.format(m.n_down_points)}`}
          hint="aperturas / cierres (muestras con cambio)"
          accent="#0E0B08"
        />
      </div>

      {/* -------- Serie temporal bidireccional -------- */}
      <Section
        title="Cambios de disponibilidad en el tiempo"
        subtitle="Arriba: tiendas que se pusieron online · Abajo: tiendas que pasaron a offline (tiendas por segundo)"
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
          <LineChart data={bidirectionalSeries} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
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
                if (name === "Aperturas/s") return [`+${fmtDec.format(v)}`, name];
                if (name === "Cierres/s") return [`−${fmtDec.format(Math.abs(v))}`, name];
                return [v, name];
              }}
            />
            <Legend />
            <ReferenceLine y={0} stroke="#999" strokeWidth={1} />
            <Line
              type="monotone" dataKey="rate_up" name="Aperturas/s"
              stroke={COLOR_UP} dot={false} strokeWidth={2}
            />
            <Line
              type="monotone" dataKey="rate_down_neg" name="Cierres/s"
              stroke={COLOR_DOWN} dot={false} strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </Section>

      {/* -------- Heatmap bidireccional -------- */}
      <Section
        title="Mapa de calor · Día × Hora"
        subtitle="Balance de tiendas online ganadas vs perdidas · Rojo: dominan aperturas · Azul: dominan cierres"
      >
        {heatmap && <Heatmap data={heatmap.data} absMax={heatmap.abs_max} />}
      </Section>

      {/* -------- Fila con 2 gráficos bidireccionales -------- */}
      <div className="two-col">
        <Section title="Patrón por hora del día" subtitle="Tiendas que abren (arriba) vs cierran (abajo) en promedio">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={hourly.map((h) => ({ ...h, avg_down_neg: -h.avg_down }))}
              stackOffset="sign">
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmtInt.format(Math.abs(v))} />
              <Tooltip
                formatter={(v, name) => {
                  if (name === "Aperturas") return [`+${fmtDec.format(v)} /s`, name];
                  if (name === "Cierres") return [`−${fmtDec.format(Math.abs(v))} /s`, name];
                  return [v, name];
                }}
              />
              <ReferenceLine y={0} stroke="#999" />
              <Bar dataKey="avg_up" name="Aperturas" stackId="a" fill={COLOR_UP} />
              <Bar dataKey="avg_down_neg" name="Cierres" stackId="a" fill={COLOR_DOWN} />
            </BarChart>
          </ResponsiveContainer>
        </Section>

        <Section title="Patrón por día de la semana" subtitle="Balance semanal de aperturas vs cierres de tiendas">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={daily.map((d) => ({ ...d, avg_down_neg: -d.avg_down }))}
              stackOffset="sign">
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="day" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmtInt.format(Math.abs(v))} />
              <Tooltip
                formatter={(v, name) => {
                  if (name === "Aperturas") return [`+${fmtDec.format(v)} /s`, name];
                  if (name === "Cierres") return [`−${fmtDec.format(Math.abs(v))} /s`, name];
                  return [v, name];
                }}
              />
              <ReferenceLine y={0} stroke="#999" />
              <Bar dataKey="avg_up" name="Aperturas" stackId="a" fill={COLOR_UP} />
              <Bar dataKey="avg_down_neg" name="Cierres" stackId="a" fill={COLOR_DOWN} />
            </BarChart>
          </ResponsiveContainer>
        </Section>
      </div>

      {/* -------- Top picos: 2 tablas lado a lado -------- */}
      <div className="two-col">
        <Section title="🔺 Top 10 aperturas masivas" subtitle="Momentos donde más tiendas se pusieron online en 10s">
          <table className="peaks-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th style={{ textAlign: "right" }}>Tiendas/s</th>
              </tr>
            </thead>
            <tbody>
              {peaks?.top_up?.map((p, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{fmtDateTime(p.timestamp)}</td>
                  <td style={{ textAlign: "right", color: COLOR_UP, fontWeight: 600 }}>
                    +{fmtDec.format(p.rate_per_sec)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>

        <Section title="🔻 Top 10 cierres masivos" subtitle="Momentos donde más tiendas pasaron a offline en 10s">
          <table className="peaks-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th style={{ textAlign: "right" }}>Tiendas/s</th>
              </tr>
            </thead>
            <tbody>
              {peaks?.top_down?.map((p, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{fmtDateTime(p.timestamp)}</td>
                  <td style={{ textAlign: "right", color: COLOR_DOWN, fontWeight: 600 }}>
                    −{fmtDec.format(p.rate_per_sec)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      </div>

      {/* -------- Anomalías bidireccionales -------- */}
      <Section
        title="🚨 Eventos anómalos detectados"
        subtitle={anomalies ? `${anomalies.count} cambios extremos fuera de ±${anomalies.sigma_threshold}σ del comportamiento normal de esa hora (de ${fmtInt.format(anomalies.total_points)} muestras)` : ""}
        right={
          <div className="sigma-control">
            <label>Umbral σ:</label>
            <select
              className="selector"
              value={sigma}
              onChange={(e) => setSigma(Number(e.target.value))}
            >
              <option value="2">2σ (más laxo)</option>
              <option value="3">3σ (estándar)</option>
              <option value="4">4σ (estricto)</option>
              <option value="5">5σ (muy estricto)</option>
            </select>
          </div>
        }
      >
        {anomalies && anomalies.top.length > 0 ? (
          <table className="peaks-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Timestamp</th>
                <th style={{ textAlign: "center" }}>Tipo</th>
                <th style={{ textAlign: "right" }}>Tasa</th>
                <th style={{ textAlign: "right" }}>z-score</th>
                <th style={{ textAlign: "right" }}>Promedio hora</th>
              </tr>
            </thead>
            <tbody>
              {anomalies.top.map((a, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{fmtDateTime(a.timestamp)}</td>
                  <td style={{ textAlign: "center" }}>
                    <span className={`badge ${a.direction === "up" ? "badge-up" : "badge-down"}`}>
                      {a.direction === "up" ? "▲ Apertura" : "▼ Cierre"}
                    </span>
                  </td>
                  <td style={{ textAlign: "right", fontWeight: 600 }}>
                    {a.rate_per_sec >= 0 ? "+" : ""}{fmtDec.format(a.rate_per_sec)} /s
                  </td>
                  <td style={{ textAlign: "right", color: a.z_score > 0 ? COLOR_UP : COLOR_DOWN, fontWeight: 600 }}>
                    {a.z_score > 0 ? "+" : ""}{fmtDec.format(a.z_score)}σ
                  </td>
                  <td style={{ textAlign: "right", color: "#5A5550" }}>
                    {fmtDec.format(a.hourly_mean)} /s
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="no-data">No hay anomalías con el umbral actual.</div>
        )}
      </Section>

      <footer className="foot">
        Construido con FastAPI + React · métrica bidireccional: aperturas vs cierres
      </footer>

      <ChatBot />
    </div>
  );
}
