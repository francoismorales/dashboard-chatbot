"""
Modelo de predicción de aperturas y cierres por hora.

Entrena DOS RandomForestRegressor:
  - Uno predice rate_up (aperturas/s promedio por hora)
  - Otro predice rate_down (cierres/s promedio por hora)

Features: hour, day_of_week, is_weekend, hour_sin/cos, dow_sin/cos.
El encoding cíclico con seno/coseno permite al modelo entender que
la hora 23 está "cerca" de la hora 0, y el domingo cerca del lunes.

Validación: los últimos 2 días se reservan como test set.
Métricas reportadas: MAE, RMSE, R², MAPE, y residual std para banda de confianza.

Uso desde ingest.py al final:
    from model import train_and_save
    train_and_save(df, OUT_DIR)
"""

from pathlib import Path
import json
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def make_features(timestamps: pd.Series) -> pd.DataFrame:
    """
    Genera features temporales a partir de una serie de timestamps en hora Bogotá.

    Features:
      - hour: 0-23 (crudo)
      - day_of_week: 0-6 (crudo, 0=lunes)
      - is_weekend: 0/1
      - hour_sin, hour_cos: encoding cíclico de la hora (24 es periodo)
      - dow_sin, dow_cos: encoding cíclico del día (7 es periodo)

    El encoding cíclico es clave: sin él, el modelo pensaría que la hora 23
    está "lejos" de la hora 0, cuando en realidad están adyacentes en el ciclo.
    """
    ts = timestamps
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ts = ts.dt.tz_convert("America/Bogota")

    hour = ts.dt.hour
    dow = ts.dt.dayofweek

    return pd.DataFrame({
        "hour": hour,
        "day_of_week": dow,
        "is_weekend": (dow >= 5).astype(int),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    })


# ---------------------------------------------------------------------------
# PREPARACIÓN DE DATOS
# ---------------------------------------------------------------------------

def prepare_hourly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los datos raw a nivel hora: para cada hora calendario
    computa la MEDIANA de rate_up y rate_down.

    Por qué mediana y no promedio: los datos tienen picos extremos (eventos
    raros de ±19K/s) que contaminan los promedios. La mediana es robusta a
    estos outliers y captura mejor el "valor típico" de cada hora, que es lo
    que el modelo puede predecir con features temporales.

    Resultado: DataFrame con columnas
      timestamp (inicio de hora) | rate_up | rate_down
    """
    d = df.dropna(subset=["rate_per_sec"]).copy()

    ts = d["timestamp"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    d["timestamp"] = ts.dt.tz_convert("America/Bogota")

    # Separar en apertura / cierre
    d["rate_up_val"] = d["rate_per_sec"].where(d["rate_per_sec"] > 0, 0)
    d["rate_down_val"] = (-d["rate_per_sec"]).where(d["rate_per_sec"] < 0, 0)

    # Agregar a nivel hora usando MEDIANA (robusta a outliers)
    hourly = (
        d.set_index("timestamp")
        .resample("1h")
        .agg(rate_up=("rate_up_val", "median"),
             rate_down=("rate_down_val", "median"))
        .reset_index()
        .dropna()
    )

    return hourly


def add_lag_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features de lag: valor de hace 24h (mismo momento del día anterior).

    Por qué: el mejor predictor de una serie temporal es su propio pasado
    reciente. Si quiero predecir las 20:00 de mañana, el valor más informativo
    es el de las 20:00 de hoy.

    Nota: con solo ~10 días de datos no usamos lag de 168h (1 semana) porque
    quitaría demasiadas filas de training. Con más historia sería ideal agregarlo.
    """
    h = hourly.sort_values("timestamp").reset_index(drop=True).copy()

    # Solo lag de 24h + rolling. Lag semanal requeriría al menos 3-4 semanas de datos.
    for target in ["rate_up", "rate_down"]:
        h[f"{target}_lag_24h"] = h[target].shift(24)
        # Promedio móvil de las últimas 24 horas (tendencia reciente)
        h[f"{target}_rolling_24h"] = h[target].shift(1).rolling(window=24, min_periods=6).mean()

    return h


# ---------------------------------------------------------------------------
# ENTRENAMIENTO + EVALUACIÓN
# ---------------------------------------------------------------------------

def train_one_target(
    hourly_with_lags: pd.DataFrame, target: str
) -> Tuple[RandomForestRegressor, dict]:
    """
    Entrena UN modelo para predecir una columna (rate_up o rate_down).

    Usa features temporales (hora, día, encoding cíclico) + lag features
    (valor de hace 24h y 168h) para capturar patrones.

    Split temporal: últimos 2 días = test. El resto = train.
    Esta estrategia simula el escenario real: predecir el futuro viendo solo
    el pasado. NO usamos shuffle random porque sería information leak.

    Devuelve (modelo_entrenado, diccionario_metricas).
    """
    # Filtrar filas sin lags (las primeras 24 horas no tienen lag)
    h = hourly_with_lags.dropna(subset=[
        f"{target}_lag_24h", f"{target}_rolling_24h"
    ]).reset_index(drop=True).copy()

    if len(h) < 30:
        raise ValueError(
            f"Datos insuficientes después de calcular lags ({len(h)} filas). "
            f"Se necesita al menos 1-2 días de historia."
        )

    # Features: temporales + lags del mismo target
    X_temporal = make_features(h["timestamp"]).reset_index(drop=True)
    X_lags = h[[
        f"{target}_lag_24h",
        f"{target}_rolling_24h",
    ]].reset_index(drop=True)
    X = pd.concat([X_temporal, X_lags], axis=1)
    y = h[target].values

    # Split temporal: últimas 24h = test (reducido porque tenemos pocos datos)
    cutoff = h["timestamp"].max() - pd.Timedelta(days=1)
    train_mask = (h["timestamp"] < cutoff).to_numpy()
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # RandomForest: robusto con poco dato, no requiere escalado,
    # captura interacciones no lineales automáticamente.
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluar en test set (nunca visto durante training)
    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # MAPE: cuidado con divisiones por cero
    nonzero = y_test > 1e-6
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs((y_test[nonzero] - y_pred[nonzero]) / y_test[nonzero])) * 100)
    else:
        mape = None

    # Para la banda de confianza: usamos la desviación estándar de los residuos
    residuals = y_test - y_pred
    residual_std = float(np.std(residuals))

    feature_importance = dict(zip(X.columns, model.feature_importances_.astype(float)))

    metrics = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mape_percent": round(mape, 2) if mape is not None else None,
        "residual_std": round(residual_std, 3),
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "feature_importance": {k: round(v, 3) for k, v in feature_importance.items()},
        "target_mean_actual": round(float(np.mean(y_test)), 3),
    }

    return model, metrics


# ---------------------------------------------------------------------------
# PREDICCIÓN DEL FUTURO
# ---------------------------------------------------------------------------

def forecast_next_days(
    model_up: RandomForestRegressor,
    model_down: RandomForestRegressor,
    metrics_up: dict,
    metrics_down: dict,
    hourly_with_lags: pd.DataFrame,
    days_ahead: int = 7,
) -> pd.DataFrame:
    """
    Genera predicciones horarias para los próximos N días.

    Estrategia: para cada hora futura, los lags se toman del mismo día/hora
    del PASADO (24h antes o 168h antes). No hacemos predicción recursiva
    (que acumularía errores), sino que usamos los valores REALES del pasado
    como features, aprovechando que los patrones son cíclicos.

    Incluye banda de confianza 95% basada en residual_std del test set.
    """
    last_timestamp = hourly_with_lags["timestamp"].max()
    if last_timestamp.tz is None:
        last_timestamp = last_timestamp.tz_localize("UTC").tz_convert("America/Bogota")

    start = last_timestamp.ceil("1h")
    future_ts = pd.date_range(
        start=start,
        periods=days_ahead * 24,
        freq="1h",
        tz="America/Bogota",
    )

    # Lookup de valores históricos para resolver los lags futuros
    hist = hourly_with_lags.set_index("timestamp")

    # Usamos el promedio global como rolling_24h para el futuro (aproximación)
    mean_up = float(hourly_with_lags["rate_up"].mean())
    mean_down = float(hourly_with_lags["rate_down"].mean())

    # Construir DataFrame de features para el futuro
    rows = []
    for ts in future_ts:
        lag_24 = ts - pd.Timedelta(hours=24)

        # Si el lag cae dentro del histórico lo usamos; si no, usamos el promedio
        up_24 = hist.loc[lag_24, "rate_up"] if lag_24 in hist.index else mean_up
        down_24 = hist.loc[lag_24, "rate_down"] if lag_24 in hist.index else mean_down

        rows.append({
            "timestamp": ts,
            "rate_up_lag_24h": up_24,
            "rate_up_rolling_24h": mean_up,
            "rate_down_lag_24h": down_24,
            "rate_down_rolling_24h": mean_down,
        })

    future_df = pd.DataFrame(rows)
    X_temporal = make_features(future_df["timestamp"]).reset_index(drop=True)

    # Features para cada modelo
    X_up = pd.concat([
        X_temporal,
        future_df[["rate_up_lag_24h", "rate_up_rolling_24h"]].reset_index(drop=True),
    ], axis=1)
    X_down = pd.concat([
        X_temporal,
        future_df[["rate_down_lag_24h", "rate_down_rolling_24h"]].reset_index(drop=True),
    ], axis=1)

    pred_up = model_up.predict(X_up)
    pred_down = model_down.predict(X_down)

    # Banda 95%: ±1.96 × residual_std
    band_up = 1.96 * metrics_up["residual_std"]
    band_down = 1.96 * metrics_down["residual_std"]

    future_df["pred_up"] = np.maximum(pred_up, 0)
    future_df["pred_up_low"] = np.maximum(pred_up - band_up, 0)
    future_df["pred_up_high"] = pred_up + band_up

    future_df["pred_down"] = np.maximum(pred_down, 0)
    future_df["pred_down_low"] = np.maximum(pred_down - band_down, 0)
    future_df["pred_down_high"] = pred_down + band_down

    future_df["pred_net"] = future_df["pred_up"] - future_df["pred_down"]

    # Limpiar columnas intermedias
    keep = ["timestamp", "pred_up", "pred_up_low", "pred_up_high",
            "pred_down", "pred_down_low", "pred_down_high", "pred_net"]
    return future_df[keep]


# ---------------------------------------------------------------------------
# PIPELINE COMPLETO: ENTRENA, EVALÚA, GUARDA
# ---------------------------------------------------------------------------

def train_and_save(df_raw: pd.DataFrame, out_dir: Path) -> dict:
    """
    Pipeline completo:
      1. Agregar a nivel hora
      2. Entrenar 2 modelos (aperturas y cierres)
      3. Evaluar métricas en test set
      4. Generar forecast de 7 días
      5. Guardar modelos + forecast + métricas

    Devuelve un dict con las métricas (útil para imprimir en consola).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🧠 Entrenando modelo de predicción...")
    hourly = prepare_hourly_dataset(df_raw)
    print(f"   Dataset horario: {len(hourly)} puntos "
          f"({hourly['timestamp'].min()} → {hourly['timestamp'].max()})")

    # Agregar lag features (valores de hace 24h y 168h)
    hourly_lags = add_lag_features(hourly)
    n_usable = hourly_lags.dropna().shape[0]
    print(f"   Con lag features aplicados: {n_usable} puntos utilizables")

    model_up, metrics_up = train_one_target(hourly_lags, "rate_up")
    model_down, metrics_down = train_one_target(hourly_lags, "rate_down")

    print(f"   Modelo APERTURAS → MAE={metrics_up['mae']:.2f}  "
          f"R²={metrics_up['r2']:.3f}  MAPE={metrics_up['mape_percent']}%")
    print(f"   Modelo CIERRES   → MAE={metrics_down['mae']:.2f}  "
          f"R²={metrics_down['r2']:.3f}  MAPE={metrics_down['mape_percent']}%")

    # Forecast 7 días
    forecast = forecast_next_days(
        model_up, model_down, metrics_up, metrics_down,
        hourly_with_lags=hourly_lags,
        days_ahead=7,
    )

    # Guardar todo
    with open(out_dir / "model_up.pkl", "wb") as f:
        pickle.dump(model_up, f)
    with open(out_dir / "model_down.pkl", "wb") as f:
        pickle.dump(model_down, f)

    # Forecast a parquet (para el frontend)
    forecast_to_save = forecast.copy()
    forecast_to_save["timestamp"] = forecast_to_save["timestamp"].astype(str)
    forecast_to_save.to_parquet(out_dir / "forecast.parquet", index=False)

    # Métricas a JSON
    model_info = {
        "trained_on": {
            "start": str(hourly["timestamp"].min()),
            "end": str(hourly["timestamp"].max()),
            "n_hours": int(len(hourly)),
        },
        "forecast": {
            "start": str(forecast["timestamp"].min()),
            "end": str(forecast["timestamp"].max()),
            "n_hours": int(len(forecast)),
            "days_ahead": 7,
        },
        "metrics_up": metrics_up,
        "metrics_down": metrics_down,
        "model_type": "RandomForestRegressor",
        "model_params": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 2,
        },
    }
    with open(out_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    print(f"   ✔ Modelos guardados en {out_dir}")
    print(f"   ✔ Forecast de 7 días generado ({len(forecast)} horas)")

    return model_info