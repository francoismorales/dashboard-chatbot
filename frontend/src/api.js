// src/api.js
// Capa fina sobre los endpoints de FastAPI.
// Un solo lugar donde cambiar si la URL del backend cambia o agregamos auth.

const BASE_URL = "http://localhost:8000";

async function fetchJSON(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`API ${path} falló: HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  summary: () => fetchJSON("/api/summary"),

  timeseries: ({ granularity = "auto", start, end } = {}) => {
    const params = new URLSearchParams({ granularity });
    if (start) params.append("start", start);
    if (end) params.append("end", end);
    return fetchJSON(`/api/timeseries?${params.toString()}`);
  },

  peaks: (topN = 10) => fetchJSON(`/api/peaks?top_n=${topN}`),

  hourlyPattern: () => fetchJSON("/api/hourly-pattern"),

  dailyPattern: () => fetchJSON("/api/daily-pattern"),

  byDate: () => fetchJSON("/api/by-date"),

  heatmap: () => fetchJSON("/api/heatmap"),

  anomalies: ({ threshold = 6, topN = 15, hour, dow } = {}) => {
    const params = new URLSearchParams({ threshold, top_n: topN });
    if (hour !== undefined && hour !== null && hour !== "") params.append("hour", hour);
    if (dow !== undefined && dow !== null && dow !== "") params.append("dow", dow);
    return fetchJSON(`/api/anomalies?${params.toString()}`);
},

  forecast: () => fetchJSON("/api/forecast"),

  modelInfo: () => fetchJSON("/api/model-info"),
};