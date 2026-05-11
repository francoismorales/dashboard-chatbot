"""
Chatbot semántico para el dashboard de Rappi.

Usa Groq (Llama 3.3 70B) con tool use para responder preguntas sobre los datos
del dashboard. El modelo decide qué tool invocar según la pregunta; las tools
ejecutan código real sobre los DataFrames de pandas y devuelven datos exactos.

Por qué tool use y no RAG:
    Los datos son numéricos/tabulares. RAG sirve para buscar texto similar, no
    para calcular estadísticas. Tool use = el LLM "llama funciones" que TÚ
    defines y ejecutas de forma segura. Así el LLM razona, tú provees la verdad.

Por qué Groq:
    - Free tier sin tarjeta
    - Llama 3.3 70B soporta tool use nativo
    - LPU hardware → respuestas en ~1s
"""

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from groq import Groq

# El cliente se crea lazy para que no explote si falta la API key al importar
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Falta GROQ_API_KEY. Configúrala en el archivo .env del backend."
            )
        _client = Groq(api_key=api_key)
    return _client

def _to_bogota(ts_series: pd.Series) -> pd.Series:
    """Convierte una serie de timestamps a hora Bogotá."""
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize("UTC")
    return ts_series.dt.tz_convert("America/Bogota")

# =============================================================================
# TOOLS: funciones que el LLM puede invocar.
# Cada una recibe el DataFrame como primer argumento + args del modelo.
# =============================================================================

def tool_describe_dataset(df: pd.DataFrame) -> dict:
    """Panorama general del dataset."""
    d = df.dropna(subset=["net_flow"])
    ts = _to_bogota(d["timestamp"])

    return {
        "metric": "synthetic_monitoring_visible_stores",
        "metric_description": (
            "Conteo agregado de tiendas visibles muestreado cada 10 segundos por SignalFx. "
            "Es un GAUGE instantáneo a nivel plataforma, no eventos discretos por tienda."
        ),
        "period_start": str(ts.min()),
        "period_end": str(ts.max()),
        "total_observations": int(len(d)),
        "value_max_observed": int(d["value"].max()),
        "value_max_at": str(_to_bogota(d.loc[d["value"].idxmax():d["value"].idxmax(), "timestamp"]).iloc[0]),
        "value_last_observed": int(d["value"].iloc[-1]),
        "value_mean": round(float(d["value"].mean()), 0),
        "max_net_growth_per_sec": round(float(d["net_flow"].max()), 2),
        "max_net_attrition_per_sec": round(float(-d["net_flow"].min()), 2),
        "avg_net_flow_per_sec": round(float(d["net_flow"].mean()), 2),
        "data_coverage_note": (
            "Aproximadamente 18 horas/día. Cada noche hay un gap sistemático de ~6h "
            "entre ~00:11 y ~06:11 sin observaciones."
        ),
        "dimensions_NOT_available": [
            "tienda individual (no hay store_id)",
            "ciudad (no hay city)",
            "vertical o categoría",
            "aperturas vs cierres separados (sólo conocemos el saldo neto)",
        ],
    }


def tool_get_hour_stats(df: pd.DataFrame, hour: int) -> dict:
    """Estadísticas del flujo neto para una hora específica del día (0-23)."""
    if not 0 <= hour <= 23:
        return {"error": "hour debe estar entre 0 y 23"}

    d = df.dropna(subset=["net_flow"]).copy()
    d["hour_local"] = _to_bogota(d["timestamp"]).dt.hour
    h = d[d["hour_local"] == hour]

    if len(h) == 0:
        return {
            "hour": hour,
            "sin_datos": True,
            "mensaje": f"No hay observaciones en la hora {hour}h (posiblemente cae en el gap nocturno).",
        }

    return {
        "hour": hour,
        "n_observations": int(len(h)),
        "net_flow_mean_per_sec": round(float(h["net_flow"].mean()), 2),
        "net_flow_median_per_sec": round(float(h["net_flow"].median()), 2),
        "net_flow_std_per_sec": round(float(h["net_flow"].std()), 2),
        "net_flow_max_per_sec": round(float(h["net_flow"].max()), 2),
        "net_flow_min_per_sec": round(float(h["net_flow"].min()), 2),
        "value_mean": round(float(h["value"].mean()), 0),
        "interpretation": (
            "net_flow positivo: el saldo de tiendas visibles crece en esa hora. "
            "Negativo: decrece. NO separa aperturas de cierres."
        ),
    }

def tool_get_dow_stats(df: pd.DataFrame, day: str) -> dict:
    """Estadísticas para un día de la semana ('lunes', 'martes', ...) agregadas sobre todas las fechas del histórico."""
    day_lower = day.lower().strip()
    days_map = {
        "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
        "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
    }
    if day_lower not in days_map:
        return {"error": f"No entiendo '{day}'. Usa 'lunes', 'martes', etc."}

    dow_num = days_map[day_lower]
    d = df.dropna(subset=["net_flow"]).copy()
    ts = _to_bogota(d["timestamp"])
    d["dow"] = ts.dt.dayofweek
    sub = d[d["dow"] == dow_num]

    if len(sub) == 0:
        return {"day": day, "sin_datos": True}

    n_unique_dates = _to_bogota(sub["timestamp"]).dt.date.nunique()
    warning = None
    if n_unique_dates <= 2:
        warning = (
            f"Sólo {n_unique_dates} fecha(s) calendario corresponden a {day} en el histórico. "
            "Las conclusiones estadísticas son débiles con tan pocas muestras."
        )

    return {
        "day": day,
        "n_observations": int(len(sub)),
        "n_calendar_dates_in_sample": int(n_unique_dates),
        "low_sample_warning": warning,
        "net_flow_mean_per_sec": round(float(sub["net_flow"].mean()), 2),
        "net_flow_median_per_sec": round(float(sub["net_flow"].median()), 2),
        "value_max_observed": int(sub["value"].max()),
        "value_mean": round(float(sub["value"].mean()), 0),
    }

def tool_get_date_stats(df: pd.DataFrame, date: str) -> dict:
    """Estadísticas para una fecha calendario específica (YYYY-MM-DD)."""
    try:
        target = pd.to_datetime(date).date()
    except Exception:
        return {"error": f"Fecha inválida: '{date}'. Usa YYYY-MM-DD."}

    d = df.dropna(subset=["net_flow"]).copy()
    ts = _to_bogota(d["timestamp"])
    d["date_local"] = ts.dt.date
    sub = d[d["date_local"] == target]

    if len(sub) == 0:
        return {"date": date, "sin_datos": True, "mensaje": "No hay observaciones para esa fecha."}

    sub_pos = sub[sub["net_flow"] > 0]
    sub_neg = sub[sub["net_flow"] < 0]
    peak_idx = sub["value"].idxmax()
    peak_ts = _to_bogota(sub.loc[peak_idx:peak_idx, "timestamp"]).iloc[0]

    return {
        "date": str(target),
        "n_observations": int(len(sub)),
        "value_max": int(sub["value"].max()),
        "value_max_at": str(peak_ts),
        "value_mean": round(float(sub["value"].mean()), 0),
        "net_flow_mean_per_sec": round(float(sub["net_flow"].mean()), 2),
        "net_flow_max_per_sec": round(float(sub["net_flow"].max()), 2),
        "net_flow_min_per_sec": round(float(sub["net_flow"].min()), 2),
        "n_samples_growing": int(len(sub_pos)),
        "n_samples_shrinking": int(len(sub_neg)),
    }


def tool_find_extreme_events(df: pd.DataFrame, direction: str = "up", top_n: int = 5) -> dict:
    """Picos extremos del flujo neto: 'growth' (más positivos) o 'attrition' (más negativos)."""
    if direction not in ("growth", "attrition"):
        return {"error": "direction debe ser 'growth' o 'attrition'"}
    if not 1 <= top_n <= 30:
        return {"error": "top_n debe estar entre 1 y 30"}

    d = df.dropna(subset=["net_flow"]).copy()
    if direction == "growth":
        extreme = d[d["net_flow"] > 0].nlargest(top_n, "net_flow")
    else:
        extreme = d[d["net_flow"] < 0].nsmallest(top_n, "net_flow")

    extreme = extreme.copy()
    extreme["timestamp"] = _to_bogota(extreme["timestamp"]).astype(str)

    return {
        "direction": direction,
        "interpretation": (
            "Estos NO son momentos de 'aperturas masivas' o 'cierres masivos'. "
            "Son los instantes donde el SALDO NETO cambió más bruscamente. "
            "El flujo neto = aperturas − cierres, y sólo conocemos la diferencia."
        ),
        "top": [
            {
                "timestamp": row["timestamp"],
                "net_flow_per_sec": round(float(row["net_flow"]), 2),
                "value_at_moment": int(row["value"]),
            }
            for _, row in extreme.iterrows()
        ],
    }


def tool_compare_two_periods(
    df: pd.DataFrame, period_a: str, period_b: str
) -> dict:
    """Compara dos días de la semana o dos fechas YYYY-MM-DD."""
    def _stats(period: str) -> dict:
        # Detectar si es día-de-semana o fecha
        if period.lower().strip() in (
            "lunes", "martes", "miércoles", "miercoles",
            "jueves", "viernes", "sábado", "sabado", "domingo",
        ):
            return tool_get_dow_stats(df, period)
        return tool_get_date_stats(df, period)

    a = _stats(period_a)
    b = _stats(period_b)
    if "error" in a or "error" in b:
        return {"error_a": a.get("error"), "error_b": b.get("error")}
    if a.get("sin_datos") or b.get("sin_datos"):
        return {"sin_datos_a": a.get("sin_datos"), "sin_datos_b": b.get("sin_datos")}

    return {
        "period_a": a,
        "period_b": b,
        "comparacion": {
            "mayor_net_flow_promedio": "A" if a.get("net_flow_mean_per_sec", 0) > b.get("net_flow_mean_per_sec", 0) else "B",
            "mayor_value_pico": "A" if a.get("value_max_observed", a.get("value_max", 0)) > b.get("value_max_observed", b.get("value_max", 0)) else "B",
        },
    }


def tool_rank_by_dimension(
    df: pd.DataFrame, dimension: str = "hour", metric: str = "pico_apertura", top_n: int = 5
) -> dict:
    """Rankea TODAS las horas (0-23) o TODOS los días de semana por una métrica."""
    if dimension not in ("hour", "dow"):
        return {"error": "dimension debe ser 'hour' o 'dow'"}

    valid_metrics = ["net_flow_mean", "net_flow_max", "net_flow_min", "value_max", "value_mean"]
    if metric not in valid_metrics:
        return {"error": f"metric debe ser uno de: {valid_metrics}"}

    d = df.dropna(subset=["net_flow"]).copy()
    ts = _to_bogota(d["timestamp"])

    if dimension == "hour":
        d["bucket"] = ts.dt.hour
        bucket_label = "hour"
        buckets = list(range(24))
        name_map = {i: f"{i:02d}h" for i in buckets}
    else:
        d["bucket"] = ts.dt.dayofweek
        bucket_label = "day"
        buckets = list(range(7))
        name_map = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
                    4: "Viernes", 5: "Sábado", 6: "Domingo"}

    rows = []
    for b in buckets:
        sub = d[d["bucket"] == b]
        if len(sub) == 0:
            continue
        value_map = {
            "net_flow_mean": float(sub["net_flow"].mean()),
            "net_flow_max":  float(sub["net_flow"].max()),
            "net_flow_min":  float(sub["net_flow"].min()),
            "value_max":     float(sub["value"].max()),
            "value_mean":    float(sub["value"].mean()),
        }
        rows.append({bucket_label: name_map[b], "valor": round(value_map[metric], 2)})

    # Para "min" rankeamos ascendente, para el resto descendente
    reverse = metric != "net_flow_min"
    rows.sort(key=lambda r: r["valor"], reverse=reverse)

    return {
        "dimension": dimension,
        "metric": metric,
        "ranking": rows[:top_n],
        "total_buckets_analizados": len(rows),
        "note": (
            "Para 'dow' con sólo 11 días de datos, cada día tiene 1-2 muestras calendario. "
            "Las conclusiones por día de semana son débiles."
            if dimension == "dow" else None
        ),
    }


def tool_get_date_range_stats(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """Estadísticas de un rango de fechas arbitrario (inclusivo, YYYY-MM-DD)."""
    try:
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
    except Exception:
        return {"error": "Fechas inválidas. Usa formato YYYY-MM-DD"}

    d = df.dropna(subset=["net_flow"]).copy()
    ts = _to_bogota(d["timestamp"])
    mask = (ts.dt.date >= start) & (ts.dt.date <= end)
    sub = d[mask]

    if len(sub) == 0:
        return {"sin_datos": True, "rango": f"{start} → {end}"}

    return {
        "range_start": str(start),
        "range_end": str(end),
        "n_observations": int(len(sub)),
        "value_max": int(sub["value"].max()),
        "value_mean": round(float(sub["value"].mean()), 0),
        "net_flow_mean_per_sec": round(float(sub["net_flow"].mean()), 2),
        "net_flow_max_per_sec": round(float(sub["net_flow"].max()), 2),
        "net_flow_min_per_sec": round(float(sub["net_flow"].min()), 2),
    }

def tool_get_anomalies(
    df: pd.DataFrame, threshold: float = 6.0, top_n: int = 10,
    hour: Optional[int] = None, dow: Optional[int] = None,
) -> dict:
    """Eventos anómalos del flujo neto detectados por z robusto (MAD por hora del día)."""
    d = df.dropna(subset=["net_flow"]).copy()
    ts = _to_bogota(d["timestamp"])
    d["hour_local"] = ts.dt.hour
    d["dow_local"] = ts.dt.dayofweek

    grouped = d.groupby("hour_local")["net_flow"]
    d["hour_median"] = grouped.transform("median")
    d["hour_mad"] = grouped.transform(lambda s: (s - s.median()).abs().median())
    d["z_robust"] = (d["net_flow"] - d["hour_median"]) / (1.4826 * d["hour_mad"]).replace(0, np.nan)
    anomalies = d[d["z_robust"].abs() > threshold].copy()

    if hour is not None:
        anomalies = anomalies[anomalies["hour_local"] == hour]
    if dow is not None:
        anomalies = anomalies[anomalies["dow_local"] == dow]

    anomalies = anomalies.reindex(anomalies["z_robust"].abs().sort_values(ascending=False).index)
    anomalies = anomalies.head(top_n)

    return {
        "threshold": threshold,
        "n_found": int(len(anomalies)),
        "filters": {"hour": hour, "dow": dow},
        "anomalies": [
            {
                "timestamp": str(_to_bogota(pd.Series([row["timestamp"]])).iloc[0]),
                "net_flow_per_sec": round(float(row["net_flow"]), 2),
                "z_robust": round(float(row["z_robust"]), 2),
                "hour": int(row["hour_local"]),
                "hour_typical_median": round(float(row["hour_median"]), 2),
                "type": "growth_spike" if row["z_robust"] > 0 else "attrition_spike",
            }
            for _, row in anomalies.iterrows()
        ],
        "note": (
            "z_robust mide cuán desviado está el punto de la mediana de su hora del día. "
            "z > 6 es muy anómalo. Picos con contraparte opuesta en ±5 min suelen ser glitches "
            "de monitoreo, no incidentes reales del negocio."
        ),
    }

# ----- Tools que requieren forecast_df + model_info -----

def tool_get_forecast(
    df, forecast_df, model_info: dict, hours_ahead: int = 24
) -> dict:
    """Predicción del flujo neto para las próximas N horas (1-48)."""
    if forecast_df is None or model_info is None:
        return {"error": "El forecast no está disponible. Corre ingest.py."}
    if not 1 <= hours_ahead <= 48:
        return {"error": "hours_ahead debe estar entre 1 y 48"}

    fc = forecast_df.head(hours_ahead).copy()
    return {
        "model_type": model_info.get("model_type"),
        "horizon_hours": int(len(fc)),
        "predictions": [
            {
                "timestamp": str(row["timestamp"]),
                "predicted_net_flow_per_sec": round(float(row["pred_net_flow"]), 2),
                "ci_95_low": round(float(row["pred_net_flow_low"]), 2),
                "ci_95_high": round(float(row["pred_net_flow_high"]), 2),
            }
            for _, row in fc.iterrows()
        ],
        "interpretation": (
            "Predicción del FLUJO NETO de tiendas visibles. Positivo = saldo crece, "
            "negativo = saldo decrece. Banda de confianza del 95% basada en residuos del modelo."
        ),
    }


def tool_get_model_info(
    df, forecast_df, model_info: dict
) -> dict:
    """Metadatos del modelo de predicción: tipo, métricas, candidatos comparados."""
    if model_info is None:
        return {"error": "No hay modelo entrenado."}
    m = model_info.get("metrics_model", {})
    b = model_info.get("metrics_baseline", {})
    return {
        "model_type": model_info.get("model_type"),
        "selection_note": model_info.get("model_selection_note"),
        "trained_on_n_hours": model_info.get("trained_on", {}).get("n_hours"),
        "mae": m.get("mae"),
        "rmse": m.get("rmse"),
        "r2": m.get("r2"),
        "baseline_mae": b.get("mae"),
        "candidates_compared": model_info.get("all_candidates", {}),
        "horizon_hours": model_info.get("forecast", {}).get("hours_ahead"),
    }


# =============================================================================
# ESQUEMA DE TOOLS para el LLM
# =============================================================================

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "describe_dataset",
            "description": (
                "Panorama general del dataset: período cubierto, valor máximo, KPIs globales y, "
                "críticamente, qué dimensiones NO existen (tienda, ciudad, vertical, aperturas vs cierres "
                "separados). USA esta tool al inicio si la pregunta es muy abierta o general."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hour_stats",
            "description": (
                "Estadísticas del flujo neto para una hora del día (0-23) agregadas sobre todos los días. "
                "Útil para '¿cómo se comporta la actividad a las 8 PM?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hora del día (0-23)"},
                },
                "required": ["hour"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dow_stats",
            "description": (
                "Estadísticas para un día de la semana ('lunes', 'martes', ...). "
                "ADVERTENCIA: con sólo 11 días de datos, cada día de semana tiene 1-2 fechas — conclusiones débiles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": "Día en español: lunes, martes, miércoles, etc."},
                },
                "required": ["day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date_stats",
            "description": (
                "Estadísticas para una fecha calendario específica (YYYY-MM-DD). "
                "Rango disponible: 2026-02-01 a 2026-02-11."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Fecha en formato YYYY-MM-DD"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_extreme_events",
            "description": (
                "Top N momentos con mayor flujo neto positivo ('growth') o negativo ('attrition'). "
                "IMPORTANTE: son momentos donde el SALDO NETO cambió más, NO 'aperturas' ni 'cierres' separados."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["growth", "attrition"],
                                  "description": "'growth' para mayor saldo positivo, 'attrition' para mayor saldo negativo"},
                    "top_n": {"type": "integer", "description": "Cuántos momentos devolver (1-30, default 5)"},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_two_periods",
            "description": (
                "Compara dos períodos: dos días de la semana ('lunes' vs 'domingo') "
                "o dos fechas específicas ('2026-02-06' vs '2026-02-08')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "period_a": {"type": "string", "description": "Primer período (día de semana o fecha YYYY-MM-DD)"},
                    "period_b": {"type": "string", "description": "Segundo período"},
                },
                "required": ["period_a", "period_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_by_dimension",
            "description": (
                "Rankea TODAS las horas (0-23) o TODOS los días de la semana por una métrica. "
                "USA siempre esta tool para preguntas tipo '¿en qué hora hay más crecimiento?' o "
                "'¿qué día tiene mayor valor pico?'. NO llames get_hour_stats una por una."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": ["hour", "dow"],
                                  "description": "'hour' para rankear las 24 horas, 'dow' para los 7 días"},
                    "metric": {"type": "string",
                               "enum": ["net_flow_mean", "net_flow_max", "net_flow_min", "value_max", "value_mean"],
                               "description": "Métrica a rankear"},
                    "top_n": {"type": "integer", "description": "Cuántos buckets devolver (default 5)"},
                },
                "required": ["dimension", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date_range_stats",
            "description": (
                "Estadísticas para un rango de fechas (YYYY-MM-DD). Útil para 'del 3 al 6 de feb' o 'la primera semana'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Fecha inicio YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "Fecha fin YYYY-MM-DD"},
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": (
                "Eventos anómalos del flujo neto detectados por z robusto. "
                "Soporta filtros opcionales por hora del día y día de la semana."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "description": "Umbral z robusto (default 6)"},
                    "top_n": {"type": "integer", "description": "Cuántas anomalías devolver (default 10)"},
                    "hour": {"type": "integer", "description": "Filtro opcional por hora del día (0-23)"},
                    "dow": {"type": "integer", "description": "Filtro opcional por día de semana (0=Lunes ... 6=Domingo)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": (
                "Predicción del flujo neto para las próximas N horas (1-48). "
                "Incluye banda de confianza del 95%."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours_ahead": {"type": "integer", "description": "Cuántas horas predecir (1-48, default 24)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": (
                "Metadatos del modelo de predicción: tipo, métricas (MAE/RMSE/R²), "
                "comparativa de candidatos evaluados, y nota de selección. "
                "Útil para '¿qué tan confiable es la predicción?' o '¿cómo se eligió el modelo?'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_DISPATCHER = {
    "describe_dataset": tool_describe_dataset,
    "get_hour_stats": tool_get_hour_stats,
    "get_dow_stats": tool_get_dow_stats,
    "get_date_stats": tool_get_date_stats,
    "find_extreme_events": tool_find_extreme_events,
    "compare_two_periods": tool_compare_two_periods,
    "rank_by_dimension": tool_rank_by_dimension,
    "get_date_range_stats": tool_get_date_range_stats,
    "get_anomalies": tool_get_anomalies,
    "get_forecast": tool_get_forecast,
    "get_model_info": tool_get_model_info,
}

TOOLS_REQUIRING_FORECAST = {"get_forecast", "get_model_info"}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def build_system_prompt(summary: dict, has_forecast: bool = False) -> str:
    """Construye el system prompt con el contexto actual del dataset."""
    metric_key = list(summary["metrics"].keys())[0]
    m = summary["metrics"][metric_key]
    p = summary["period"]

    forecast_section = ""
    if has_forecast:
        forecast_section = """

PREDICCIÓN (FORECAST):
- El dashboard incluye un modelo que predice el flujo neto para las próximas 48 horas.
- Se evaluaron 4 candidatos (baseline mediana por hora, Ridge, GradientBoosting, RandomForest)
  en walk-forward validation. El baseline ganó por menor MAE; ningún modelo paramétrico
  superó la mediana por hora del día con los datos disponibles.
- Si el usuario pregunta sobre la predicción, usa `get_forecast` para los valores horarios
  o `get_model_info` para métricas del modelo.
- Si preguntan "¿qué tan bueno es?" o "¿es confiable?", reporta el R² y el MAE reales usando
  `get_model_info` y aclara que se eligió por comparación honesta contra alternativas."""

    return f"""Eres un asistente analítico del dashboard de disponibilidad de tiendas de Rappi.

QUÉ ES LA MÉTRICA (LEER CON CUIDADO):
`synthetic_monitoring_visible_stores` es un conteo agregado de tiendas visibles muestreado
cada 10 segundos por el sistema de monitoreo de Rappi (SignalFx). Es un GAUGE — un valor
instantáneo del agregado, no eventos individuales por tienda. El número (millones) refleja
una agregación a nivel plataforma; no es directamente "cantidad de tiendas únicas".

CONCEPTOS CLAVE:
- `value` = nivel observado del agregado en un instante.
- `net_flow` (rate_per_sec) = derivada del value, en unidades de "/segundo". Representa
  el SALDO NETO: tiendas que entran al sistema menos tiendas que salen, por segundo.
- net_flow positivo = el saldo crece (más tiendas entrando que saliendo).
- net_flow negativo = el saldo decrece (más tiendas saliendo que entrando).

LO QUE NO PODEMOS RESPONDER (NUNCA INVENTES):
- Aperturas vs cierres por separado. Sólo conocemos el saldo neto (diferencia), no los
  componentes individuales. Si el usuario pregunta "¿cuántas tiendas abrieron?", aclara
  honestamente: "El dataset sólo permite ver el saldo neto. En ese momento el saldo creció
  en X tiendas/segundo, pero no sabemos cuántas abrieron ni cuántas cerraron por separado."
- Métricas por tienda individual: no hay store_id.
- Métricas por ciudad, vertical o categoría: no existen esas dimensiones.

COBERTURA DEL DATASET:
- Rango: {p['start']} → {p['end']} ({p['duration_hours']} horas).
- {p['total_points']:,} muestras (una cada 10 segundos).
- Aproximadamente 18 horas/día con datos. Cada noche hay un GAP sistemático de ~6 horas
  entre ~00:11 y ~06:11. Si te preguntan por horas en ese rango, indica que no hay datos.
- Valor máximo observado: {int(m['value_max']):,} ({m['value_max_at']}).
- Último valor observado: {int(m['value_last_observation']):,}.
- Mayor crecimiento neto: +{m['max_net_growth_per_sec']:.1f}/s ({m['peak_growth_at']}).
- Mayor decrecimiento neto: -{m['max_net_attrition_per_sec']:.1f}/s ({m['peak_attrition_at']}).{forecast_section}

REGLAS DE COMPORTAMIENTO:
1. Para CUALQUIER número específico, fecha o estadística, USA las tools. Nunca inventes
   cifras ni las aproximes de memoria.
2. Si una pregunta es muy abierta, empieza con `describe_dataset`.
3. Si una pregunta requiere comparar opciones (ej. "¿en qué hora hay más X?"), usa
   `rank_by_dimension` — NO llames `get_hour_stats` una por una.
4. Cuando reportes flujo neto, incluye unidades ("/s") y la interpretación correcta
   (positivo = crece, negativo = decrece).
5. Si te preguntan "cuántas aperturas / cuántos cierres", reformula honestamente al saldo
   neto. Ej: "El saldo creció X /s en ese momento. No podemos descomponer cuántas tiendas
   abrieron vs cerraron individualmente."
6. Sé conciso: 2-4 frases en preguntas simples, listas numeradas para rankings.
7. Tono profesional pero cercano, en español. Usa formato legible para los números
   (separador de miles, decimales sensatos).
8. Si una pregunta no se puede responder con el dataset, dilo claramente en lugar de inventar
   ("el dataset no contiene esa dimensión", "no hay observaciones en ese rango", etc.).
9. IMPORTANTE: responde SIEMPRE en texto plano. NO uses Markdown bajo ninguna circunstancia.
   - No uses tablas Markdown.
   - No uses listas con `#`, `*`, `-`, `|`, ni bloques de código.
   - No uses negritas, cursivas ni backticks.
   - No formatees respuestas como tablas ASCII.
   - Entrega respuestas limpias, naturales y legibles en texto simple.
10. Si necesitas enumerar elementos, usa formato simple:
    1. elemento
    2. elemento
    3. elemento
11. Nunca devuelvas contenido con símbolos típicos de Markdown como:
    `#`, `##`, `###`, `**`, `|---|`, triple backticks o tablas.

GUÍA RÁPIDA DE TOOLS:
- "panorama general / qué tenemos" → `describe_dataset`
- "qué pasa a las X horas" → `get_hour_stats`
- "cómo son los lunes / domingos" → `get_dow_stats`
- "qué pasó el día X / fecha específica" → `get_date_stats`
- "del día A al día B" → `get_date_range_stats`
- "cuándo hubo más / pico de crecimiento o decrecimiento" → `find_extreme_events`
- "compara X con Y" → `compare_two_periods`
- "¿en qué hora/día hay más X?" → `rank_by_dimension`
- "eventos anómalos / cosas raras / outliers" → `get_anomalies`
- "predicción / qué pasará / próximos días" → `get_forecast`
- "qué tan bueno es el modelo / cómo se entrenó" → `get_model_info`

Si la pregunta es ambigua, pide aclaración antes de llamar tools."""


# =============================================================================
# LOOP PRINCIPAL DEL AGENTE
# =============================================================================

def chat(
    user_message: str,
    conversation_history: list,
    df: pd.DataFrame,
    summary: dict,
    forecast_df=None,
    model_info: dict = None,
    model: str = "openai/gpt-oss-120b",
    max_iterations: int = 5,
) -> dict:
    """
    Ejecuta un turno de conversación con tool use.

    Flujo:
    1. LLM decide si llamar una tool o responder directo
    2. Si llama tool(s), las ejecutamos y devolvemos resultado al LLM
    3. LLM redacta respuesta final con los datos
    4. Loop hasta max_iterations (safety)

    Si el modelo primario falla (tool_use_failed), intenta con fallback.

    forecast_df y model_info son opcionales: si se proveen, las tools
    de forecast pueden invocarse. Si no, el LLM recibe error si intenta usarlas.
    """
    client = _get_client()
    system_prompt = build_system_prompt(summary, has_forecast=forecast_df is not None)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    tool_calls_log = []
    models_to_try = [model, "llama-3.3-70b-versatile"] if model != "llama-3.3-70b-versatile" else [model]

    last_error = None
    for current_model in models_to_try:
        try:
            return _run_agent_loop(
                client, current_model, messages, tool_calls_log,
                df, forecast_df, model_info, max_iterations
            )
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "tool_use_failed" in err_str or "invalid_request" in err_str:
                tool_calls_log.append({"name": "__model_fallback__", "from": current_model})
                continue
            raise

    raise last_error or RuntimeError("Todos los modelos fallaron")


def _run_agent_loop(client, model, messages, tool_calls_log, df, forecast_df, model_info, max_iterations):
    """Loop interno de tool use. Separado para permitir reintento con otro modelo."""
    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.2,
            max_completion_tokens=800,
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return {
                "response": msg.content,
                "tool_calls": tool_calls_log,
                "iterations": iteration + 1,
                "model_used": model,
            }

        # Ejecutar cada tool que pidió el LLM
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            tool_calls_log.append({"name": fn_name, "args": fn_args})

            if fn_name not in TOOL_DISPATCHER:
                result = {"error": f"Tool '{fn_name}' no existe"}
            else:
                try:
                    # Las tools de forecast reciben forecast_df + model_info además de df
                    if fn_name in TOOLS_REQUIRING_FORECAST:
                        result = TOOL_DISPATCHER[fn_name](df, forecast_df, model_info, **fn_args)
                    else:
                        result = TOOL_DISPATCHER[fn_name](df, **fn_args)
                except Exception as e:
                    result = {"error": f"Error ejecutando {fn_name}: {str(e)}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str, ensure_ascii=False),
            })

    return {
        "response": "Disculpa, no pude procesar tu pregunta. ¿Podrías reformularla?",
        "tool_calls": tool_calls_log,
        "iterations": max_iterations,
        "model_used": model,
    }