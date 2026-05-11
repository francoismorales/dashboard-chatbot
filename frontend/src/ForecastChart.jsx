// src/ForecastChart.jsx
// Predicción del flujo neto para las próximas 48h.
// El modelo final fue elegido por comparación walk-forward de 4 candidatos:
// baseline (mediana por hora), Ridge, GradientBoosting, RandomForest.
// El panel muestra esa comparación para que la decisión sea verificable.

import { useEffect, useState } from "react";
import {
  ComposedChart, Area, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, ReferenceLine,
} from "recharts";
import { api } from "./api";
import "./ForecastChart.css";

const COLOR_UP = "#FF441F";
const COLOR_BAND = "rgba(120, 120, 120, 0.15)";

const fmtDec = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 1 });
const fmtPct = new Intl.NumberFormat("es-CO", { style: "percent", maximumFractionDigits: 1 });

const fmtDateShort = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("es-CO", {
    weekday: "short", day: "2-digit", month: "short", hour: "2-digit",
  });
};

function MetricBadge({ label, value, hint, color = "#0E0B08" }) {
  return (
    <div className="fc-metric">
      <div className="fc-metric-label">{label}</div>
      <div className="fc-metric-value" style={{ color }}>{value}</div>
      {hint && <div className="fc-metric-hint">{hint}</div>}
    </div>
  );
}

// Evaluación cualitativa del R²
function r2Quality(r2) {
  if (r2 >= 0.85) return { text: "Patrón fuerte capturado", color: "#0B8043" };
  if (r2 >= 0.65) return { text: "Patrón razonable", color: "#2E7D32" };
  if (r2 >= 0.40) return { text: "Patrón parcial", color: "#F59E0B" };
  return { text: "Patrón débil", color: "#B91C1C" };
}

export default function ForecastChart() {
  const [forecast, setForecast] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.forecast(), api.modelInfo()])
      .then(([fc, mi]) => {
        setForecast(fc.data);
        setModelInfo(mi);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="fc-loading">⏳ Cargando predicción…</div>;
  if (error) return <div className="fc-error">⚠️ Predicción no disponible: {error}</div>;

  const m = modelInfo.metrics_model;
  const b = modelInfo.metrics_baseline;
  const q = r2Quality(m.r2);
  const isBaseline = modelInfo.winner === "baseline";

  const data = forecast.map((r) => ({
    timestamp: r.timestamp,
    pred: r.pred_net_flow,
    band_top: r.pred_net_flow_high,
    band_bot: r.pred_net_flow_low,
  }));

  const featureImportance = m.feature_importance ?? {};
  const allCandidates = modelInfo.all_candidates ?? {};
  const winnerLabel = modelInfo.model_type;

  return (
    <section className="fc-panel">
      <header className="fc-header">
        <div>
          <h2>🔮 Predicción del flujo neto · próximas 48 horas</h2>
          <p className="fc-subtitle">
            Modelo seleccionado: <strong>{modelInfo.model_type}</strong> ·{" "}
            Entrenado con {modelInfo.trained_on.n_hours} horas históricas ·{" "}
            {isBaseline
              ? "Aplica la mediana histórica del flujo neto correspondiente a cada hora del día."
              : "Modelo paramétrico con features cíclicas + lag de 24h."}
          </p>
        </div>
      </header>

      {/* ---- Métricas del modelo ---- */}
      <div className="fc-metrics">
        <MetricBadge
          label="Calidad del modelo"
          value={q.text}
          hint={`R² = ${m.r2.toFixed(3)}`}
          color={q.color}
        />
        <MetricBadge
          label="MAE"
          value={`±${fmtDec.format(m.mae)} /s`}
          hint="error absoluto promedio"
        />
        <MetricBadge
          label="RMSE"
          value={fmtDec.format(m.rmse)}
          hint="raíz del error cuadrático"
        />
        <MetricBadge
          label="Intervalo 95%"
          value={`±${fmtDec.format(1.96 * m.residual_std)} /s`}
          hint="banda de confianza"
        />
      </div>

      {/* ---- Chart ---- */}
      <div className="fc-chart-wrap">
        <ResponsiveContainer width="100%" height={380}>
          <ComposedChart data={data} margin={{ top: 20, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={fmtDateShort}
              minTickGap={60}
              tick={{ fontSize: 10 }}
            />
            <YAxis tick={{ fontSize: 11 }} width={70} tickFormatter={(v) => fmtDec.format(v)} />
            <Tooltip
              labelFormatter={fmtDateShort}
              formatter={(v, name) => {
                if (v == null) return ["—", name];
                const labels = {
                  pred: "Predicción",
                  band_top: "Banda superior",
                  band_bot: "Banda inferior",
                };
                const sign = v >= 0 ? "+" : "−";
                return [`${sign}${fmtDec.format(Math.abs(v))} /s`, labels[name] || name];
              }}
            />
            <Legend
              payload={[
                { value: "Predicción de flujo neto", type: "line", color: COLOR_UP },
                { value: "Intervalo 95%", type: "rect", color: "#888" },
              ]}
            />

            <ReferenceLine y={0} stroke="#999" strokeWidth={1} strokeDasharray="2 2" />

            {/* Banda de confianza: high relleno y luego low en blanco */}
            <Area
              type="monotone"
              dataKey="band_top"
              stroke="none"
              fill={COLOR_BAND}
              fillOpacity={1}
              activeDot={false}
              legendType="none"
            />
            <Area
              type="monotone"
              dataKey="band_bot"
              stroke="none"
              fill="#FFFFFF"
              fillOpacity={1}
              activeDot={false}
              legendType="none"
            />

            {/* Línea de predicción */}
            <Line
              type="monotone"
              dataKey="pred"
              stroke={COLOR_UP}
              strokeWidth={2.5}
              strokeDasharray="6 3"
              dot={false}
              name="Predicción"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Footer con interpretación ---- */}
      <footer className="fc-footer">
        <div>
          💡 <strong>Cómo leerlo:</strong> la línea punteada es la predicción puntual del flujo neto
          (saldo de tiendas/segundo). La banda gris es el intervalo donde el valor real debería caer
          el 95% de las veces. La línea horizontal en 0 separa crecimiento neto (arriba) de
          decrecimiento neto (abajo).
        </div>
        <div className="fc-technical">
          <details>
            <summary>📊 Detalles técnicos del modelo y la comparativa</summary>
            <div className="fc-technical-content">
              {/* Métricas del modelo ganador */}
              <div>
                <strong>Métricas del modelo seleccionado (walk-forward, {m.n_folds} folds):</strong>
                <ul>
                  <li>MAE: <strong>{fmtDec.format(m.mae)}</strong> /s</li>
                  <li>RMSE: <strong>{fmtDec.format(m.rmse)}</strong> /s</li>
                  <li>R²: <strong>{m.r2.toFixed(3)}</strong></li>
                  <li>Desviación de residuos: <strong>{fmtDec.format(m.residual_std)}</strong> /s</li>
                  <li>Outliers excluidos del training: <strong>{modelInfo.trained_on.excluded_outlier_hours}</strong> horas (cluster 9-feb)</li>
                </ul>
                {isBaseline && (
                  <p style={{ marginTop: 10, color: "#5A5550" }}>
                    <strong>Nota:</strong> el modelo no usa lag, así que la calidad de la predicción
                    a la hora 48 es idéntica a la de la hora 1. El error no se compone con el horizonte.
                  </p>
                )}
              </div>

              {/* Importancia de features */}
              <div className="fc-features">
                <strong>{isBaseline ? "Horas más predictivas:" : "Importancia de features:"}</strong>
                <div className="fc-bars">
                  {Object.entries(featureImportance)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 8)
                    .map(([name, imp]) => (
                      <div key={name} className="fc-bar">
                        <span className="fc-bar-label">{name}</span>
                        <div className="fc-bar-track">
                          <div className="fc-bar-fill" style={{ width: `${imp * 100}%` }} />
                        </div>
                        <span className="fc-bar-value">{(imp * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                </div>
              </div>

              {/* Tabla comparativa de candidatos (ocupa las 2 columnas) */}
              <div>
                <strong>Comparativa de candidatos evaluados:</strong>
                <p style={{ marginTop: 4, color: "#5A5550", fontSize: 11 }}>
                  Cada modelo se evaluó en {m.n_folds} cortes temporales sucesivos.
                  Los valores son promedios entre folds. Menor MAE = mejor.
                </p>
                <table style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: 12,
                  marginTop: 10,
                }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #EAE7E3" }}>
                      <th style={{
                        textAlign: "left", padding: "6px 8px",
                        color: "#5A5550", fontWeight: 500,
                      }}>Modelo</th>
                      <th style={{
                        textAlign: "right", padding: "6px 8px",
                        color: "#5A5550", fontWeight: 500,
                      }}>MAE</th>
                      <th style={{
                        textAlign: "right", padding: "6px 8px",
                        color: "#5A5550", fontWeight: 500,
                      }}>RMSE</th>
                      <th style={{
                        textAlign: "right", padding: "6px 8px",
                        color: "#5A5550", fontWeight: 500,
                      }}>R²</th>
                      <th style={{
                        textAlign: "right", padding: "6px 8px",
                        color: "#5A5550", fontWeight: 500,
                      }}>Skill vs baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(allCandidates).map(([name, s]) => {
                      const isWinner = name === winnerLabel;
                      const skill = s.skill_vs_baseline;
                      return (
                        <tr key={name} style={{ borderBottom: "1px solid #F7F5F1" }}>
                          <td style={{
                            padding: "6px 8px",
                            fontWeight: isWinner ? 700 : 400,
                            color: isWinner ? "#FF441F" : "#0E0B08",
                          }}>
                            {isWinner ? "🏆 " : ""}{name}
                          </td>
                          <td style={{
                            textAlign: "right", padding: "6px 8px",
                            fontFamily: "ui-monospace, monospace",
                            fontWeight: isWinner ? 700 : 400,
                            color: isWinner ? "#FF441F" : "#0E0B08",
                          }}>
                            {fmtDec.format(s.mae)}
                          </td>
                          <td style={{
                            textAlign: "right", padding: "6px 8px",
                            fontFamily: "ui-monospace, monospace",
                            color: "#5A5550",
                          }}>
                            {fmtDec.format(s.rmse)}
                          </td>
                          <td style={{
                            textAlign: "right", padding: "6px 8px",
                            fontFamily: "ui-monospace, monospace",
                            color: "#5A5550",
                          }}>
                            {s.r2.toFixed(3)}
                          </td>
                          <td style={{
                            textAlign: "right", padding: "6px 8px",
                            fontFamily: "ui-monospace, monospace",
                            color: skill === null || skill === undefined
                              ? "#5A5550"
                              : skill > 0 ? "#0B8043" : "#B91C1C",
                          }}>
                            {skill === null || skill === undefined
                              ? "(referencia)"
                              : fmtPct.format(skill)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
                <p style={{ marginTop: 10, color: "#5A5550", fontSize: 11, lineHeight: 1.5 }}>
                  El ganador se eligió por menor MAE en walk-forward. Reportar la tabla completa
                  permite verificar la decisión y deja explícito que se evaluaron alternativas
                  antes de adoptar el modelo final.
                </p>
              </div>
            </div>
          </details>
        </div>
      </footer>
    </section>
  );
}