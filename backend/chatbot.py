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
from typing import Any

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


# =============================================================================
# TOOLS: funciones que el LLM puede invocar.
# Cada una recibe el DataFrame como primer argumento + args del modelo.
# =============================================================================

def tool_get_overview(df: pd.DataFrame) -> dict:
    """Panorama general del dataset."""
    d = df.dropna(subset=["rate_per_sec"])
    ts = d["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("America/Bogota")

    total_up_events = int((d["rate_per_sec"] > 0).sum())
    total_down_events = int((d["rate_per_sec"] < 0).sum())

    return {
        "period_start": str(ts.min()),
        "period_end": str(ts.max()),
        "total_samples": int(len(d)),
        "contador_al_inicio": int(d["value"].iloc[0]),
        "contador_al_final": int(d["value"].iloc[-1]),
        "crecimiento_neto_del_contador": int(d["value"].iloc[-1] - d["value"].iloc[0]),
        "n_samples_con_aumento": total_up_events,
        "n_samples_con_disminucion": total_down_events,
        "mayor_aumento_por_segundo": round(float(d["rate_per_sec"].max()), 1),
        "mayor_disminucion_por_segundo": round(float(d["rate_per_sec"].min()), 1),
        "aumento_promedio_por_segundo": round(float(d.loc[d["rate_per_sec"] > 0, "rate_per_sec"].mean()), 2),
        "disminucion_promedio_por_segundo": round(float(d.loc[d["rate_per_sec"] < 0, "rate_per_sec"].abs().mean()), 2),
    }


def tool_get_hour_stats(df: pd.DataFrame, hour: int) -> dict:
    """Estadísticas específicas de una hora del día (0-23)."""
    if not 0 <= hour <= 23:
        return {"error": "hour debe estar entre 0 y 23"}

    d = df.dropna(subset=["rate_per_sec"]).copy()
    ts = d["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("America/Bogota")
    d["hour"] = ts.dt.hour
    h = d[d["hour"] == hour]

    if len(h) == 0:
        return {"hour": hour, "sin_datos": True, "mensaje": f"No hay datos en la hora {hour}"}

    ups = h.loc[h["rate_per_sec"] > 0, "rate_per_sec"]
    downs = h.loc[h["rate_per_sec"] < 0, "rate_per_sec"].abs()

    return {
        "hour": hour,
        "total_samples": int(len(h)),
        "n_eventos_apertura": int(len(ups)),
        "n_eventos_cierre": int(len(downs)),
        "pico_apertura_s": round(float(ups.max()) if len(ups) else 0, 1),
        "pico_cierre_s": round(float(downs.max()) if len(downs) else 0, 1),
        "apertura_promedio_s": round(float(ups.mean()) if len(ups) else 0, 2),
        "cierre_promedio_s": round(float(downs.mean()) if len(downs) else 0, 2),
        "flujo_neto_promedio_s": round(float(h["rate_per_sec"].mean()), 2),
    }


def tool_get_day_stats(df: pd.DataFrame, day: str) -> dict:
    """Estadísticas de un día de la semana (lunes, martes, etc.) o fecha YYYY-MM-DD."""
    d = df.dropna(subset=["rate_per_sec"]).copy()
    ts = d["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("America/Bogota")

    day_lower = day.lower().strip()
    days_map = {
        "lunes": 0, "martes": 1, "miércoles": 2, "miercoles": 2,
        "jueves": 3, "viernes": 4, "sábado": 5, "sabado": 5, "domingo": 6,
    }

    if day_lower in days_map:
        dow = days_map[day_lower]
        mask = ts.dt.dayofweek == dow
        subset_label = f"todos los {day_lower}"
    else:
        # Intentar como fecha específica
        try:
            target = pd.to_datetime(day).date()
            mask = ts.dt.date == target
            subset_label = f"fecha {target}"
        except Exception:
            return {"error": f"No entiendo '{day}'. Usa 'lunes', 'martes'... o una fecha YYYY-MM-DD"}

    subset = d[mask]
    if len(subset) == 0:
        return {"day": day, "sin_datos": True, "mensaje": f"No hay datos para {subset_label}"}

    ups = subset.loc[subset["rate_per_sec"] > 0, "rate_per_sec"]
    downs = subset.loc[subset["rate_per_sec"] < 0, "rate_per_sec"].abs()

    return {
        "day": subset_label,
        "total_samples": int(len(subset)),
        "n_eventos_apertura": int(len(ups)),
        "n_eventos_cierre": int(len(downs)),
        "pico_apertura_s": round(float(ups.max()) if len(ups) else 0, 1),
        "pico_cierre_s": round(float(downs.max()) if len(downs) else 0, 1),
        "apertura_promedio_s": round(float(ups.mean()) if len(ups) else 0, 2),
        "cierre_promedio_s": round(float(downs.mean()) if len(downs) else 0, 2),
        "tiendas_al_inicio": int(subset["value"].iloc[0]),
        "tiendas_al_final": int(subset["value"].iloc[-1]),
        "crecimiento_neto_del_dia": int(subset["value"].iloc[-1] - subset["value"].iloc[0]),
    }


def tool_find_extreme_events(df: pd.DataFrame, direction: str = "up", top_n: int = 5) -> dict:
    """Los N momentos de mayor apertura o cierre masivo."""
    if direction not in ("up", "down"):
        return {"error": "direction debe ser 'up' o 'down'"}

    d = df.dropna(subset=["rate_per_sec"]).copy()

    if direction == "up":
        extreme = d[d["rate_per_sec"] > 0].nlargest(top_n, "rate_per_sec")
        label = "aperturas_masivas"
    else:
        extreme = d[d["rate_per_sec"] < 0].nsmallest(top_n, "rate_per_sec")
        extreme = extreme.copy()
        extreme["rate_per_sec"] = extreme["rate_per_sec"].abs()
        label = "cierres_masivos"

    ts = extreme["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    extreme = extreme.copy()
    extreme["timestamp"] = ts.dt.tz_convert("America/Bogota").astype(str)

    return {
        "tipo": label,
        "top": [
            {
                "timestamp": row["timestamp"],
                "tiendas_por_segundo": round(float(row["rate_per_sec"]), 1),
                "tiendas_online_en_ese_momento": int(row["value"]),
            }
            for _, row in extreme.iterrows()
        ],
    }


def tool_compare_two_periods(
    df: pd.DataFrame, period_a: str, period_b: str
) -> dict:
    """Compara dos períodos descritos como días ('lunes') o fechas ('2026-02-05')."""
    stats_a = tool_get_day_stats(df, period_a)
    stats_b = tool_get_day_stats(df, period_b)

    if "error" in stats_a or "error" in stats_b:
        return {"error_a": stats_a.get("error"), "error_b": stats_b.get("error")}

    return {
        "periodo_a": stats_a,
        "periodo_b": stats_b,
        "comparacion": {
            "mas_eventos_apertura_en": "A" if stats_a.get("n_eventos_apertura", 0) > stats_b.get("n_eventos_apertura", 0) else "B",
            "mas_eventos_cierre_en": "A" if stats_a.get("n_eventos_cierre", 0) > stats_b.get("n_eventos_cierre", 0) else "B",
            "mayor_crecimiento_neto_en": "A" if stats_a.get("crecimiento_neto_del_dia", 0) > stats_b.get("crecimiento_neto_del_dia", 0) else "B",
        },
    }


def tool_rank_by_dimension(
    df: pd.DataFrame, dimension: str = "hour", metric: str = "pico_apertura", top_n: int = 5
) -> dict:
    """
    Rankea TODAS las horas del día o TODOS los días de la semana por una métrica.
    Esencial para responder 'cuál es la hora con más aperturas' (hay que escanear las 24).
    """
    if dimension not in ("hour", "day"):
        return {"error": "dimension debe ser 'hour' o 'day'"}

    valid_metrics = ["pico_apertura", "pico_cierre", "promedio_apertura",
                     "promedio_cierre", "flujo_neto", "n_aperturas", "n_cierres"]
    if metric not in valid_metrics:
        return {"error": f"metric debe ser uno de: {valid_metrics}"}

    d = df.dropna(subset=["rate_per_sec"]).copy()
    ts = d["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("America/Bogota")

    if dimension == "hour":
        d["bucket"] = ts.dt.hour
        bucket_label = "hora"
        buckets = range(24)
    else:
        days_es = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
                   4: "Viernes", 5: "Sábado", 6: "Domingo"}
        d["bucket"] = ts.dt.dayofweek
        bucket_label = "día"
        buckets = range(7)

    results = []
    for b in buckets:
        sub = d[d["bucket"] == b]
        if len(sub) == 0:
            continue

        ups = sub.loc[sub["rate_per_sec"] > 0, "rate_per_sec"]
        downs = sub.loc[sub["rate_per_sec"] < 0, "rate_per_sec"].abs()

        value_map = {
            "pico_apertura": float(ups.max()) if len(ups) else 0,
            "pico_cierre": float(downs.max()) if len(downs) else 0,
            "promedio_apertura": float(ups.mean()) if len(ups) else 0,
            "promedio_cierre": float(downs.mean()) if len(downs) else 0,
            "flujo_neto": float(sub["rate_per_sec"].mean()),
            "n_aperturas": int(len(ups)),
            "n_cierres": int(len(downs)),
        }

        label = days_es[b] if dimension == "day" else f"{b}:00"
        results.append({bucket_label: label, "valor": round(value_map[metric], 2)})

    # Ordenar descendente por valor
    results.sort(key=lambda x: x["valor"], reverse=True)

    return {
        "dimension": dimension,
        "metrica_rankeada": metric,
        "ranking": results[:top_n],
        "total_buckets_analizados": len(results),
    }


def tool_get_date_range_stats(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """Estadísticas para un rango de fechas arbitrario (YYYY-MM-DD)."""
    d = df.dropna(subset=["rate_per_sec"]).copy()
    ts = d["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("America/Bogota")

    try:
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
    except Exception:
        return {"error": "Fechas inválidas. Usa formato YYYY-MM-DD"}

    mask = (ts.dt.date >= start) & (ts.dt.date <= end)
    subset = d[mask]

    if len(subset) == 0:
        return {"sin_datos": True, "rango": f"{start} → {end}"}

    ups = subset.loc[subset["rate_per_sec"] > 0, "rate_per_sec"]
    downs = subset.loc[subset["rate_per_sec"] < 0, "rate_per_sec"].abs()

    return {
        "rango": f"{start} → {end}",
        "total_samples": int(len(subset)),
        "n_eventos_apertura": int(len(ups)),
        "n_eventos_cierre": int(len(downs)),
        "pico_apertura_s": round(float(ups.max()) if len(ups) else 0, 1),
        "pico_cierre_s": round(float(downs.max()) if len(downs) else 0, 1),
        "contador_al_inicio_del_rango": int(subset["value"].iloc[0]),
        "contador_al_final_del_rango": int(subset["value"].iloc[-1]),
        "crecimiento_neto_del_rango": int(subset["value"].iloc[-1] - subset["value"].iloc[0]),
    }


# =============================================================================
# ESQUEMA DE TOOLS para el LLM (formato OpenAI/Groq)
# =============================================================================

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_overview",
            "description": "Panorama general del dataset: período, totales de eventos, picos globales, crecimiento neto. Úsalo para preguntas genéricas como '¿cómo se ve el panorama?' o '¿qué pasó en general?'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hour_stats",
            "description": "Estadísticas de una hora específica del día (0-23). Úsalo para preguntas como '¿qué pasa a las 8pm?', '¿cómo son las 3 de la mañana?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hora del día en formato 24h (0-23)"},
                },
                "required": ["hour"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_day_stats",
            "description": "Estadísticas de un día de la semana ('lunes', 'martes'...) o una fecha específica (formato YYYY-MM-DD). Úsalo para '¿cómo son los lunes?' o '¿qué pasó el 9 de febrero?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "description": "Día de la semana en español O fecha YYYY-MM-DD"},
                },
                "required": ["day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_extreme_events",
            "description": "Los momentos más extremos de apertura o cierre masivo de tiendas. Úsalo para '¿cuándo hubo más aperturas?' o '¿cuándo se cerraron más tiendas?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"], "description": "'up' para aperturas masivas, 'down' para cierres masivos"},
                    "top_n": {"type": "integer", "description": "Cuántos momentos devolver (default 5)"},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_two_periods",
            "description": "Compara dos días o fechas. Úsalo para '¿cómo se compara el lunes vs el martes?' o 'compara el 3-feb con el 5-feb'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period_a": {"type": "string", "description": "Primer período ('lunes' o 'YYYY-MM-DD')"},
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
            "description": "Rankea TODAS las horas del día (0-23) o TODOS los días de la semana por una métrica. ÚSALO SIEMPRE para preguntas como '¿en qué hora hay más aperturas?', '¿qué día tiene más cierres?', '¿cuál es la franja horaria con más actividad?'. NO uses get_hour_stats de a una.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": ["hour", "day"], "description": "'hour' para rankear las 24 horas, 'day' para rankear los 7 días"},
                    "metric": {"type": "string", "enum": ["pico_apertura", "pico_cierre", "promedio_apertura", "promedio_cierre", "flujo_neto", "n_aperturas", "n_cierres"], "description": "Métrica a rankear"},
                    "top_n": {"type": "integer", "description": "Cuántos buckets devolver en el top (default 5)"},
                },
                "required": ["dimension", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date_range_stats",
            "description": "Estadísticas de un rango de fechas (inicio y fin en formato YYYY-MM-DD). Úsalo para '¿cómo fueron del 3 al 5 de feb?' o 'resumen de la primera semana'.",
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
]

TOOL_DISPATCHER = {
    "get_overview": tool_get_overview,
    "get_hour_stats": tool_get_hour_stats,
    "get_day_stats": tool_get_day_stats,
    "find_extreme_events": tool_find_extreme_events,
    "compare_two_periods": tool_compare_two_periods,
    "rank_by_dimension": tool_rank_by_dimension,
    "get_date_range_stats": tool_get_date_range_stats,
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def build_system_prompt(summary: dict) -> str:
    """Construye el system prompt con el contexto actual del dataset."""
    m = list(summary["metrics"].values())[0]
    p = summary["period"]

    return f"""Eres un asistente analítico del dashboard de monitoreo de tiendas de Rappi.

QUÉ MIDE REALMENTE LA MÉTRICA (MUY IMPORTANTE):
La métrica `synthetic_monitoring_visible_stores` NO es el número de tiendas individuales.
Es el CONTEO DE CHEQUEOS de monitoreo sintético que respondieron "visible/online" en un
momento dado. El sistema sintético chequea cada tienda múltiples veces por segundo desde
distintas regiones/servidores, por lo que el número es mucho mayor que la cantidad real
de tiendas. Por ejemplo, 3.7 millones de "visibles" pueden corresponder a ~100 mil tiendas
reales chequeadas ~37 veces cada una.

Por lo tanto:
- NO digas "hay 3.7 millones de tiendas online". Di "se registraron 3.7 millones de chequeos
  exitosos de visibilidad" o "el contador sintético marcó 3.7M".
- NO digas "se abrieron 19.532 tiendas en un segundo". Di "el contador de chequeos aumentó
  en 19.532 unidades en ese segundo".
- Los picos grandes reflejan cambios de carga del sistema sintético o cambios coordinados
  de estado entre muchas tiendas — no necesariamente aperturas/cierres individuales.

CONTEXTO DEL DATASET:
- Rango: {p['start']} → {p['end']} ({p['duration_hours']} horas)
- Muestras: {p['total_points']:,} (una cada 10 segundos)
- Contador al inicio: {int(m['value_start']):,}
- Contador al final: {int(m['value_end']):,}
- Mayor aumento del contador en un segundo: +{int(m['max_rate_up']):,} ({m['peak_up_at']})
- Mayor disminución del contador en un segundo: -{int(m['max_rate_down']):,} ({m['peak_down_at']})

INTERPRETACIÓN OPERACIONAL:
- Aumentos en el contador = tiendas que pasaron a responder "visible"
- Disminuciones = tiendas que dejaron de responder "visible"
- Picos extremos coinciden con momentos de cambio de estado masivo (cierres/aperturas
  comerciales coordinadas).

REGLAS:
1. Para CUALQUIER número específico, SIEMPRE usa las tools. Nunca inventes cifras.
2. Responde en español, tono profesional pero cercano.
3. Sé conciso: 2-4 frases en la mayoría de casos. Para rankings, usa lista numerada.
4. Usa lenguaje cauto: "el contador muestra X", "se registraron Y cambios", en vez de
   "hay X tiendas" o "se abrieron Y tiendas".
5. Si detectas un patrón interesante en los datos, menciónalo.
6. Si el usuario pregunta cuántas tiendas reales tiene Rappi, aclara que los datos no
   permiten saberlo directamente porque miden chequeos de monitoreo, no tiendas únicas.

GUÍA PARA ELEGIR TOOLS:
- "¿qué hora / franja horaria / día tiene más X?" → usa `rank_by_dimension`, NO `get_hour_stats`
- "¿qué pasa a las 8pm?" / "stats de la hora X" → `get_hour_stats`
- "stats del lunes" / "cómo son los domingos" → `get_day_stats`
- "del 3 al 5 de feb" / "rango de fechas" → `get_date_range_stats`
- "cuándo hubo más / top / ranking de aperturas o cierres" → `find_extreme_events`
- "compara X con Y" → `compare_two_periods`
- "resumen general / qué tenemos" → `get_overview`

Si una pregunta es ambigua, pide aclaración antes de invocar una tool."""


# =============================================================================
# LOOP PRINCIPAL DEL AGENTE
# =============================================================================

def chat(
    user_message: str,
    conversation_history: list,
    df: pd.DataFrame,
    summary: dict,
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
    """
    client = _get_client()
    system_prompt = build_system_prompt(summary)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    tool_calls_log = []
    models_to_try = [model, "llama-3.3-70b-versatile"] if model != "llama-3.3-70b-versatile" else [model]

    last_error = None
    for current_model in models_to_try:
        try:
            return _run_agent_loop(
                client, current_model, messages, tool_calls_log, df, max_iterations
            )
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # Si es tool_use_failed o similar, intenta con el siguiente modelo
            if "tool_use_failed" in err_str or "invalid_request" in err_str:
                tool_calls_log.append({"name": "__model_fallback__", "from": current_model})
                continue
            # Otros errores (red, rate limit) no vale la pena reintentar con otro modelo
            raise

    raise last_error or RuntimeError("Todos los modelos fallaron")


def _run_agent_loop(client, model, messages, tool_calls_log, df, max_iterations):
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

        # Si no hay tool_calls, terminamos
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