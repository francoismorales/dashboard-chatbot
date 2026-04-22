// src/ForecastChart.jsx
// Gráfico de predicción de próximos 7 días - modelo de CIERRES únicamente.
// El modelo de aperturas fue descartado por baja calidad predictiva (R²=0.25).
// Mantener solo lo que funciona bien es una decisión consciente: integridad > completitud.

import { useEffect, useState } from "react";
import {
  ComposedChart, Area, Line,
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, ReferenceLine,
} from "recharts";
import { api } from "./api";
import "./ForecastChart.css";

const COLOR_DOWN = "#1F6FFF";
const COLOR_BAND_DOWN = "rgba(31, 111, 255, 0.15)";

const fmtDec = new Intl.NumberFormat("es-CO", { maximumFractionDigits: 1 });

const fmtDateShort = (iso) => {
  if (!iso) return "";
  const d = new Date(iso);
  return d.toLocaleString("es-CO", {
    weekday: "short", day: "2-digit", month: "short", hour: "2-digit",
  });
};

// Interpretación de R² para usuario no técnico
function qualityLabel(r2) {
  if (r2 >= 0.8) return { text: "Excelente", color: "#0B8043" };
  if (r2 >= 0.6) return { text: "Bueno", color: "#2E7D32" };
  if (r2 >= 0.4) return { text: "Aceptable", color: "#F59E0B" };
  if (r2 >= 0.2) return { text: "Débil", color: "#D97706" };
  return { text: "Pobre", color: "#B91C1C" };
}

function MetricBadge({ label, value, hint, color = "#0E0B08" }) {
  return (
    <div className="fc-metric">
      <div className="fc-metric-label">{label}</div>
      <div className="fc-metric-value" style={{ color }}>{value}</div>
      {hint && <div className="fc-metric-hint">{hint}</div>}
    </div>
  );
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

  const m = modelInfo.metrics_down;
  const q = qualityLabel(m.r2);

  // Preparar datos: rate_down se grafica hacia abajo del eje 0 para consistencia visual
  // con el resto del dashboard (cierres siempre en la mitad inferior, color azul).
  const data = forecast.map((r) => ({
    timestamp: r.timestamp,
    pred_down_neg: -r.pred_down,
    pred_down_low_neg: -r.pred_down_high,   // invertidos porque al negar el alto
    pred_down_high_neg: -r.pred_down_low,   // queda abajo y el bajo queda arriba
    pred_down: r.pred_down,
    pred_down_low: r.pred_down_low,
    pred_down_high: r.pred_down_high,
  }));

  return (
    <section className="fc-panel">
      <header className="fc-header">
        <div>
          <h2>🔮 Predicción para los próximos 7 días</h2>
          <p className="fc-subtitle">
            Predicción horaria de <strong>cierres</strong> ·{" "}
            Modelo: <strong>{modelInfo.model_type}</strong> ·{" "}
            Entrenado con {modelInfo.trained_on.n_hours} horas
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
          label="Error promedio (MAE)"
          value={`±${fmtDec.format(m.mae)}`}
          hint="cierres/s de error típico"
          color={COLOR_DOWN}
        />
        <MetricBadge
          label="RMSE"
          value={fmtDec.format(m.rmse)}
          hint="raíz del error cuadrático medio"
        />
        <MetricBadge
          label="Intervalo de confianza"
          value="95%"
          hint={`±${fmtDec.format(1.96 * m.residual_std)} cierres/s`}
        />
      </div>

      {/* ---- Gráfico ---- */}
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
            <YAxis tick={{ fontSize: 11 }} width={70} />
            <Tooltip
              labelFormatter={fmtDateShort}
              formatter={(v, name) => {
                if (v == null) return ["—", name];
                const labels = {
                  "pred_down_neg": "Cierres (predicción)",
                  "pred_down_low_neg": "Banda inferior",
                  "pred_down_high_neg": "Banda superior",
                };
                return [fmtDec.format(Math.abs(v)), labels[name] || name];
              }}
            />
            <Legend
              payload={[
                { value: "Predicción de cierres", type: "line", color: COLOR_DOWN },
                { value: "Intervalo 95%", type: "rect", color: "#888" },
              ]}
            />

            <ReferenceLine y={0} stroke="#999" strokeWidth={1} />

            {/* CIERRES: área de banda + línea punteada */}
            <Area
              type="monotone"
              dataKey="pred_down_high_neg"
              stroke="none"
              fill={COLOR_BAND_DOWN}
              fillOpacity={1}
              activeDot={false}
              legendType="none"
            />
            <Area
              type="monotone"
              dataKey="pred_down_low_neg"
              stroke="none"
              fill="#FFFFFF"
              fillOpacity={1}
              activeDot={false}
              legendType="none"
            />
            <Line
              type="monotone"
              dataKey="pred_down_neg"
              stroke={COLOR_DOWN}
              strokeWidth={2.5}
              strokeDasharray="6 3"
              dot={false}
              name="Cierres"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ---- Pie con interpretación ---- */}
      <footer className="fc-footer">
        <div>
          💡 <strong>Cómo leerlo:</strong> la línea punteada es la predicción puntual.
          La zona sombreada es el intervalo donde el valor real debería caer el 95% de las veces.
          Cuanto más estrecha la banda, más certera la predicción para ese momento.
        </div>
        <div className="fc-technical">
          <details>
            <summary>📊 Detalles técnicos del modelo</summary>
            <div className="fc-technical-content">
              <div>
                <strong>Métricas en test set ({m.n_test} horas nunca vistas durante training):</strong>
                <ul>
                  <li>MAE (error absoluto promedio): <strong>{fmtDec.format(m.mae)}</strong></li>
                  <li>RMSE: <strong>{fmtDec.format(m.rmse)}</strong></li>
                  <li>R² (variabilidad capturada): <strong>{m.r2.toFixed(3)}</strong></li>
                  <li>Desviación de residuos: <strong>{fmtDec.format(m.residual_std)}</strong></li>
                </ul>
              </div>
              <div className="fc-features">
                <strong>Features más importantes:</strong>
                <div className="fc-bars">
                  {Object.entries(m.feature_importance)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 5)
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
            </div>
          </details>
        </div>
      </footer>
    </section>
  );
}
