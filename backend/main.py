"""
Backend FastAPI para el dashboard de disponibilidad de tiendas Rappi.

Interpretación de la métrica synthetic_monitoring_visible_stores:
    rate_per_sec = cambio neto de tiendas visibles por segundo
    rate_up (positivo) = aperturas netas
    rate_down (negativo en rate_per_sec, absoluto en rate_down) = cierres netos

El dashboard muestra ambos signos para revelar el ritmo bidireccional del negocio.
"""

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import chatbot  # nuestro módulo del agente

# Cargar variables de entorno desde .env
load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BACKEND_DIR.parent / "data" / "processed"

DATA: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Cargando datos en memoria...")
    DATA["raw"] = pd.read_parquet(PROCESSED_DIR / "data.parquet")
    DATA["by_minute"] = pd.read_parquet(PROCESSED_DIR / "by_minute.parquet")
    DATA["by_hour"] = pd.read_parquet(PROCESSED_DIR / "by_hour.parquet")
    with open(PROCESSED_DIR / "summary.json", "r", encoding="utf-8") as f:
        DATA["summary"] = json.load(f)

    # Cargar forecast y métricas si existen (puede no haberse entrenado aún)
    forecast_path = PROCESSED_DIR / "forecast.parquet"
    model_info_path = PROCESSED_DIR / "model_info.json"
    if forecast_path.exists() and model_info_path.exists():
        DATA["forecast"] = pd.read_parquet(forecast_path)
        with open(model_info_path, "r", encoding="utf-8") as f:
            DATA["model_info"] = json.load(f)
        print(f"  ✔ forecast:  {len(DATA['forecast']):,} horas futuras cargadas")
    else:
        DATA["forecast"] = None
        DATA["model_info"] = None
        print("  ⚠  forecast no disponible (corre ingest.py primero)")

    print(f"  ✔ raw:       {len(DATA['raw']):,} filas")
    print(f"  ✔ by_minute: {len(DATA['by_minute']):,} filas")
    print(f"  ✔ by_hour:   {len(DATA['by_hour']):,} filas")
    yield


app = FastAPI(title="Rappi Availability API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _df_to_records(df: pd.DataFrame) -> list:
    """DataFrame -> lista de dicts lista para JSON. Maneja timestamps y NaN."""
    df = df.copy()
    if "timestamp" in df.columns and len(df) > 0:
        ts = df["timestamp"]
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("UTC")
        df["timestamp"] = ts.dt.tz_convert("America/Bogota").astype(str)
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records


def _filter_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start, utc=True)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end, utc=True)]
    return df


def _localize(ts_series):
    """Asegura tz UTC y convierte a hora Bogotá."""
    if ts_series.dt.tz is None:
        ts_series = ts_series.dt.tz_localize("UTC")
    return ts_series.dt.tz_convert("America/Bogota")


# ----------------- Endpoints -----------------
@app.get("/")
def root():
    return {"status": "ok", "service": "rappi-availability-api"}


@app.get("/api/summary")
def get_summary():
    """KPIs globales con métricas bidireccionales (aperturas vs cierres)."""
    return DATA["summary"]


@app.get("/api/timeseries")
def get_timeseries(
    start: Optional[str] = None,
    end: Optional[str] = None,
    granularity: str = Query("auto", pattern="^(auto|minute|hour)$"),
):
    """Serie temporal bidireccional: rate_up (aperturas) y rate_down (cierres)."""
    if granularity == "auto":
        df = _filter_range(DATA["by_minute"], start, end)
        if len(df) > 3000:
            df = _filter_range(DATA["by_hour"], start, end)
            granularity_used = "hour"
        else:
            granularity_used = "minute"
    else:
        source = DATA["by_minute"] if granularity == "minute" else DATA["by_hour"]
        df = _filter_range(source, start, end)
        granularity_used = granularity

    if len(df) > 3000:
        step = len(df) // 3000 + 1
        df = df.iloc[::step]

    return {
        "granularity": granularity_used,
        "count": len(df),
        "data": _df_to_records(df),
    }


@app.get("/api/peaks")
def get_peaks(top_n: int = Query(10, ge=1, le=50)):
    """Top picos de apertura y de cierre, por separado."""
    df = DATA["raw"].dropna(subset=["rate_per_sec"])

    top_up = df[df["rate_per_sec"] > 0].nlargest(top_n, "rate_per_sec")
    top_up = top_up[["timestamp", "value", "rate_per_sec"]].copy()

    top_down = df[df["rate_per_sec"] < 0].nsmallest(top_n, "rate_per_sec")
    top_down = top_down[["timestamp", "value", "rate_per_sec"]].copy()
    top_down["rate_per_sec"] = top_down["rate_per_sec"].abs()

    return {
        "top_up": _df_to_records(top_up),
        "top_down": _df_to_records(top_down),
    }


@app.get("/api/hourly-pattern")
def get_hourly_pattern():
    """Promedio de aperturas y cierres por hora del día (0-23)."""
    df = DATA["raw"].copy().dropna(subset=["rate_per_sec"])
    df["hour"] = _localize(df["timestamp"]).dt.hour

    result = []
    for hour in range(24):
        h = df[df["hour"] == hour]
        if len(h) == 0:
            result.append({"hour": hour, "avg_up": 0, "avg_down": 0, "n_samples": 0})
            continue
        ups = h.loc[h["rate_per_sec"] > 0, "rate_per_sec"]
        downs = h.loc[h["rate_per_sec"] < 0, "rate_per_sec"].abs()
        result.append({
            "hour": hour,
            "avg_up": round(float(ups.mean()) if len(ups) else 0, 2),
            "avg_down": round(float(downs.mean()) if len(downs) else 0, 2),
            "n_samples": int(len(h)),
        })
    return {"data": result}


@app.get("/api/daily-pattern")
def get_daily_pattern():
    """Promedio de aperturas y cierres por día de la semana."""
    days_es = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    df = DATA["raw"].copy().dropna(subset=["rate_per_sec"])
    ts = _localize(df["timestamp"])
    df["dow_num"] = ts.dt.dayofweek

    result = []
    for dow in range(7):
        d = df[df["dow_num"] == dow]
        if len(d) == 0:
            continue
        ups = d.loc[d["rate_per_sec"] > 0, "rate_per_sec"]
        downs = d.loc[d["rate_per_sec"] < 0, "rate_per_sec"].abs()
        result.append({
            "dow_num": dow,
            "day": days_es[dow],
            "avg_up": round(float(ups.mean()) if len(ups) else 0, 2),
            "avg_down": round(float(downs.mean()) if len(downs) else 0, 2),
        })
    return {"data": result}


@app.get("/api/heatmap")
def get_heatmap():
    """
    Matriz día-de-semana × hora-del-día con net_flow (aperturas - cierres).
    Rojo cuando dominan aperturas, azul cuando dominan cierres.
    """
    days_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    df = DATA["raw"].copy().dropna(subset=["rate_per_sec"])
    ts = _localize(df["timestamp"])
    df["dow_num"] = ts.dt.dayofweek
    df["hour"] = ts.dt.hour

    pivot = (
        df.groupby(["dow_num", "hour"])["rate_per_sec"]
        .mean().round(2).reset_index()
    )
    pivot.rename(columns={"rate_per_sec": "net_flow"}, inplace=True)
    pivot["day"] = pivot["dow_num"].map(days_es)

    abs_max = float(pivot["net_flow"].abs().max()) if len(pivot) else 0

    return {
        "data": pivot.to_dict(orient="records"),
        "min": float(pivot["net_flow"].min()) if len(pivot) else 0,
        "max": float(pivot["net_flow"].max()) if len(pivot) else 0,
        "abs_max": abs_max,
    }


@app.get("/api/anomalies")
def get_anomalies(sigma: float = Query(3.0, ge=1.0, le=6.0), top_n: int = Query(15, ge=1, le=50)):
    """
    Anomalías en ambos sentidos (apertura y cierre).
    Z-score calculado contra el promedio de cada hora del día.
    """
    df = DATA["raw"].copy().dropna(subset=["rate_per_sec"])
    ts = _localize(df["timestamp"])
    df["hour"] = ts.dt.hour

    stats = df.groupby("hour")["rate_per_sec"].agg(hourly_mean="mean", hourly_std="std").reset_index()
    df = df.merge(stats, on="hour")
    df["z_score"] = (df["rate_per_sec"] - df["hourly_mean"]) / df["hourly_std"].replace(0, np.nan)
    df["is_anomaly"] = df["z_score"].abs() > sigma
    df["direction"] = np.where(df["z_score"] >= 0, "up", "down")

    count = int(df["is_anomaly"].sum())
    anomalies = df[df["is_anomaly"]].copy()
    anomalies = anomalies.reindex(anomalies["z_score"].abs().sort_values(ascending=False).index)
    anomalies = anomalies.head(top_n)
    anomalies = anomalies[["timestamp", "value", "rate_per_sec", "z_score", "hour", "hourly_mean", "direction"]]
    anomalies = anomalies.round(2)

    return {
        "sigma_threshold": sigma,
        "count": count,
        "total_points": len(df),
        "top": _df_to_records(anomalies),
    }


@app.get("/api/forecast")
def get_forecast():
    """
    Predicción horaria de los próximos 7 días para aperturas y cierres.
    Incluye banda de confianza 95% (±1.96 × desviación de residuos).
    """
    if DATA.get("forecast") is None:
        raise HTTPException(
            status_code=503,
            detail="Forecast no disponible. Corre `python ingest.py` para entrenar el modelo."
        )

    df = DATA["forecast"].copy()
    # Los timestamps están guardados como string en el parquet
    return {
        "count": len(df),
        "data": df.to_dict(orient="records"),
    }


@app.get("/api/model-info")
def get_model_info():
    """Métricas de evaluación del modelo (MAE, RMSE, R², MAPE, feature importance)."""
    if DATA.get("model_info") is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no entrenado. Corre `python ingest.py`."
        )
    return DATA["model_info"]


# =============================================================================
# CHATBOT ENDPOINT
# =============================================================================

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.post("/api/chat")
def post_chat(req: ChatRequest):
    """
    Endpoint del chatbot con tool use.
    Recibe: mensaje del usuario + historial conversacional
    Devuelve: respuesta del asistente + lista de tools invocadas (para debugging/UI)
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]

    try:
        result = chatbot.chat(
            user_message=req.message,
            conversation_history=history,
            df=DATA["raw"],
            summary=DATA["summary"],
            forecast_df=DATA.get("forecast"),
            model_info=DATA.get("model_info"),
        )
    except RuntimeError as e:
        # Falta API key u otro problema de config
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error del chatbot: {e}")

    return {
        "response": result["response"],
        "tool_calls": result["tool_calls"],
        "iterations": result["iterations"],
    }