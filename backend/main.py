

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

def _mark_suspicious(picos: pd.DataFrame, contrapartes: pd.DataFrame) -> pd.DataFrame:
    """Marca picos que tienen un pico opuesto dentro de ±5 min."""
    if len(picos) == 0:
        picos = picos.copy()
        picos["is_suspicious"] = False
        return picos

    ts_contra = contrapartes["timestamp"].reset_index(drop=True)
    delta_min = pd.Timedelta(minutes=5)

    suspicious = []
    for ts in picos["timestamp"]:
        close = ((ts_contra >= (ts - delta_min)) & (ts_contra <= (ts + delta_min))).any()
        suspicious.append(bool(close))

    picos = picos.copy()
    picos["is_suspicious"] = suspicious
    return picos

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
    """Serie temporal del flujo neto de tiendas visibles."""
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
    """Top picos de flujo neto."""
    df = DATA["raw"].dropna(subset=["net_flow"])

    top_growth = (df[df["net_flow"] > 0].nlargest(top_n, "net_flow")[["timestamp", "value", "net_flow"]])
    top_attrition = (df[df["net_flow"] < 0].nsmallest(top_n, "net_flow")[["timestamp", "value", "net_flow"]].copy())
    top_growth = _mark_suspicious(top_growth, top_attrition)
    top_attrition = _mark_suspicious(top_attrition, top_growth)

    cols = ["timestamp", "value", "net_flow", "is_suspicious"]

    return {
        "top_growth":    _df_to_records(top_growth[cols]),
        "top_attrition": _df_to_records(top_attrition[cols]),
    }


@app.get("/api/hourly-pattern")
def get_hourly_pattern():
    """Flujo neto promedio por hora del día (0-23)."""
    df = DATA["raw"].copy().dropna(subset=["net_flow"])
    df["hour"] = _localize(df["timestamp"]).dt.hour

    result = []
    for hour in range(24):
        h = df[df["hour"] == hour]
        if len(h) == 0:
            result.append({"hour": hour, "avg_net_flow": 0, "n_samples": 0})
            continue
        result.append({
            "hour": hour,
            "avg_net_flow": round(float(h["net_flow"].mean()), 2),
            "n_samples": int(len(h)),
        })
    return {"data": result}


@app.get("/api/daily-pattern")
def get_daily_pattern():
    """Flujo neto por día de la semana."""
    days_es = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    df = DATA["raw"].copy().dropna(subset=["net_flow"])
    ts = _localize(df["timestamp"])
    df["dow_num"] = ts.dt.dayofweek

    result = []
    for dow in range(7):
        d = df[df["dow_num"] == dow]
        if len(d) == 0:
            continue
        result.append({
            "dow_num": dow,
            "day": days_es[dow],
            "avg_net_flow": round(float(d["net_flow"].mean()), 2),
        })
    return {"data": result}

@app.get("/api/by-date")
def get_by_date():
    """
    Balance diario del flujo neto por fecha calendario.
    Devuelve una entrada por cada día con datos: fecha, flujo neto promedio.
    """
    df = DATA["raw"].copy().dropna(subset=["net_flow"])
    ts = _localize(df["timestamp"])
    df["date"] = ts.dt.date

    result = []
    for date, day in df.groupby("date"):
        flow = day["net_flow"]
        result.append({
            "date": date.isoformat(),
            "label": date.strftime("%d %b"),  # "01 feb"
            "avg_net_flow": round(float(flow.mean()), 2),
            "sum_growth": round(float(flow[flow > 0].sum()), 2),
            "sum_attrition": round(float(flow[flow < 0].sum()), 2),  # negativo
            "value_peak": float(day["value"].max()),
            "n_samples": int(len(day)),
        })
    return {"data": result}

@app.get("/api/heatmap")
def get_heatmap():
    """
    Matriz día-de-semana × hora-del-día con net_flow (aperturas - cierres).
    Rojo cuando dominan aperturas, azul cuando dominan cierres.
    """
    days_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    df = DATA["raw"].copy().dropna(subset=["net_flow"])
    ts = _localize(df["timestamp"])
    df["dow_num"] = ts.dt.dayofweek
    df["hour"] = ts.dt.hour

    pivot = (
        df.groupby(["dow_num", "hour"])["net_flow"]
        .mean().round(2).reset_index()
    )
    pivot["day"] = pivot["dow_num"].map(days_es)
    abs_values = pivot["net_flow"].abs()
    abs_max = float(abs_values.quantile(0.95)) if len(abs_values) else 0

    return {
        "data": pivot.to_dict(orient="records"),
        "min": float(pivot["net_flow"].min()) if len(pivot) else 0,
        "max": float(pivot["net_flow"].max()) if len(pivot) else 0,
        "abs_max": abs_max,
    }


@app.get("/api/anomalies")
def get_anomalies(threshold: float = Query(6.0, ge=1.0, le=12.0), top_n: int = Query(15, ge=1, le=50), hour: Optional[int] = Query(None, ge=0, le=23), dow: Optional[int] = Query(None, ge=0, le=6),):
    """
    Anomalías del flujo neto detectadas con z-score
    """
    df = DATA["raw"].copy().dropna(subset=["net_flow"])
    ts = _localize(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dow_num"] = ts.dt.dayofweek

    days_es = {0: "Lun", 1: "Mar", 2: "Mié", 3: "Jue", 4: "Vie", 5: "Sáb", 6: "Dom"}
    df["dow"] = df["dow_num"].map(days_es)

    grouped = df.groupby("hour")["net_flow"]
    df["hour_median"] = grouped.transform("median")
    df["hour_mad"] = grouped.transform(lambda s: (s - s.median()).abs().median())
    df["z_robust"] = (df["net_flow"] - df["hour_median"]) / (1.4826 * df["hour_mad"]).replace(0, np.nan)
    df["is_anomaly"] = df["z_robust"].abs() > threshold
    df["direction"] = np.where(df["z_robust"] >= 0, "growth_spike", "attrition_spike")

    count = int(df["is_anomaly"].sum())
    anomalies = df[df["is_anomaly"]].copy()

    if hour is not None:
        anomalies = anomalies[anomalies["hour"] == hour]
    if dow is not None:
        anomalies = anomalies[anomalies["dow_num"] == dow]
    
    count_filtered = len(anomalies)

    anomalies = anomalies.reindex(anomalies["z_robust"].abs().sort_values(ascending=False).index)
    anomalies = anomalies.head(top_n)
    cols = ["timestamp", "value", "net_flow", "z_robust", "hour", "dow", "hour_median", "direction"]
    anomalies = anomalies[cols].copy()
    numeric_cols = ["value", "net_flow", "z_robust", "hour_median"]
    anomalies[numeric_cols] = anomalies[numeric_cols].round(2)
    
    return {
        "threshold": threshold,
        "filters": {"hour": hour, "dow": dow},
        "count": count,
        "total_points": len(df),
        "count_filtered": count_filtered,
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