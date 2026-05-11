"""
Script de ingesta: lee los 201 CSVs crudos de /data/raw, los normaliza
a formato long, elimina duplicados por overlap, calcula la tasa (derivada)
y guarda:
    - data/processed/data.parquet   -> dataset completo en long format
    - data/processed/summary.json   -> KPIs y agregados para el dashboard

Uso:
    cd backend
    source venv/bin/activate
    python ingest.py
"""

import glob
import json
import re
from pathlib import Path

import pandas as pd

# ---- Rutas ----
# El script vive en /backend pero los datos están en /data (un nivel arriba)
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_timestamp_column(col_name: str) -> str:
    """
    Limpia el nombre feo de columna tipo:
        'Sun Feb 01 2026 06:59:40 GMT-0500 (hora estándar de Colombia)'
    deja solo:
        'Sun Feb 01 2026 06:59:40 GMT-0500'
    """
    return re.sub(r"\s*\(.*\)\s*", "", col_name).strip()


def load_one_file(path: Path) -> pd.DataFrame:
    """
    Lee UN csv en formato wide y lo devuelve en formato long:
        timestamp | metric | value
    """
    df = pd.read_csv(path)

    # Las primeras 4 columnas son metadatos, el resto son timestamps
    id_cols = df.columns[:4].tolist()  # Plot name, metric, prefix, suffix
    time_cols = df.columns[4:].tolist()

    # Wide -> long
    long_df = df.melt(
        id_vars=id_cols,
        value_vars=time_cols,
        var_name="timestamp_raw",
        value_name="value",
    )

    # Renombrar columna de métrica para que sea usable
    long_df = long_df.rename(columns={"metric (sf_metric)": "metric"})

    # Limpiar timestamps
    long_df["timestamp"] = long_df["timestamp_raw"].apply(parse_timestamp_column)
    long_df["timestamp"] = pd.to_datetime(
        long_df["timestamp"], format="%a %b %d %Y %H:%M:%S GMT%z", errors="coerce"
    )

    # Nos quedamos con lo esencial
    long_df = long_df[["timestamp", "metric", "value"]].copy()

    # Convertir value a numérico (por si hay celdas vacías o basura)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    # Tirar filas sin timestamp o sin valor
    long_df = long_df.dropna(subset=["timestamp", "value"])

    return long_df


def main() -> None:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No se encontraron CSVs en {RAW_DIR}")

    print(f"📂 Encontrados {len(csv_files)} archivos en {RAW_DIR}")

    # ---- Ingesta ----
    frames = []
    failed = []
    for i, path in enumerate(csv_files, start=1):
        try:
            frames.append(load_one_file(path))
        except Exception as e:
            failed.append((path.name, str(e)))
        if i % 25 == 0 or i == len(csv_files):
            print(f"  procesados {i}/{len(csv_files)}")

    if failed:
        print(f"⚠️  {len(failed)} archivos fallaron:")
        for name, err in failed[:5]:
            print(f"    - {name}: {err}")

    df = pd.concat(frames, ignore_index=True)
    print(f"\n🧩 Filas concatenadas (con duplicados): {len(df):,}")

    # ---- Deduplicar overlaps ----
    # Los archivos consecutivos comparten 30 segundos en la frontera
    df = df.drop_duplicates(subset=["timestamp", "metric"]).sort_values("timestamp")
    df = df.reset_index(drop=True)
    print(f"🧹 Filas únicas tras dedupe: {len(df):,}")

    # ---- Feature derivada: tasa ----
    # value es un acumulado. La "tasa" es cuánto crece entre un punto y el siguiente.
    # delta / delta_t nos da eventos por segundo -> la métrica operacional interesante.
    df["delta"] = df.groupby("metric")["value"].diff()
    df["delta_seconds"] = df.groupby("metric")["timestamp"].diff().dt.total_seconds()
    df["rate_per_sec"] = df["delta"] / df["delta_seconds"]

    ## ---- Invalidar deltas a través de gaps anómalos ----
    # Cuando hay un hueco grande entre dos muestras (ej: gap nocturno de ~6h),
    # el delta calculado cruza ese vacío y produce un "rate" ficticio.
    # Marcamos como NaN cualquier delta cuyo intervalo de tiempo sea más del doble
    # del esperado (10s). NaN se propaga a las columnas derivadas y a los agregados.
    EXPECTED_INTERVAL_S = 10
    abnormal_gap = df["delta_seconds"] > EXPECTED_INTERVAL_S * 2
    n_invalidated = int(abnormal_gap.sum())
    df.loc[abnormal_gap, ["delta", "rate_per_sec"]] = pd.NA
    print(f"Deltas invalidados por gap > 20s: {n_invalidated}")

    # ---- Descomposición del flujo neto ----
    # rate_per_sec es el cambio NETO de tiendas visibles por segundo.
    # No podemos separar aperturas de cierres porque sólo tenemos el saldo.
    # Lo que SÍ podemos hacer es separar el saldo en momentos de crecimiento
    # y momentos de decrecimiento.
    df["net_flow"]      = df["rate_per_sec"]
    df["net_growth"]    = df["rate_per_sec"].where(df["rate_per_sec"] > 0, 0)
    df["net_attrition"] = (-df["rate_per_sec"]).where(df["rate_per_sec"] < 0, 0)

    # ---- Guardar dataset principal ----
    # IMPORTANTE: este save tiene que ir DESPUÉS de la descomposición, porque
    # main.py lee este parquet y necesita las columnas net_flow / net_growth / net_attrition.
    parquet_path = OUT_DIR / "data.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Dataset guardado en {parquet_path}")
    print(f"   Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Métricas: {df['metric'].unique().tolist()}")
    print(f"   Columnas: {df.columns.tolist()}")

    # ---- Resumen compacto para dashboard + chatbot ----
    # Este JSON es lo que el frontend cargará para las KPI cards,
    # y será parte del system prompt del chatbot (por eso pequeño).
    summary = {
        "period": {
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
            "total_points": int(len(df)),
            "duration_hours": round(
                (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600, 2
            ),
        },
        "metrics": {},
    }

    for metric_name, group in df.groupby("metric"):
        flow = group["net_flow"].dropna()
        growth_periods    = flow[flow > 0]
        attrition_periods = flow[flow < 0]
        peak_growth_row    = group.loc[flow.idxmax()] if len(flow) else None
        peak_attrition_row = group.loc[flow.idxmin()] if len(flow) else None

        summary["metrics"][metric_name] = {
        # Estado observado
        "value_first_observation": float(group["value"].iloc[0]),
        "value_last_observation":  float(group["value"].iloc[-1]),
        "value_max":               float(group["value"].max()),
        "value_max_at":            group.loc[group["value"].idxmax(), "timestamp"].isoformat(),
        "value_min":               float(group["value"].min()),
        "value_mean":               round(float(group["value"].mean()), 2),

        # Flujo neto
        "avg_net_flow":           round(float(flow.mean()), 2),
        "avg_growth_when_positive":   round(float(growth_periods.mean())    if len(growth_periods)    else 0, 2),
        "avg_attrition_when_negative": round(float(attrition_periods.abs().mean()) if len(attrition_periods) else 0, 2),

        # Recuento de muestras
        "n_samples_growing":   int(len(growth_periods)),
        "n_samples_shrinking": int(len(attrition_periods)),
        "n_samples_total":     int(len(flow)),

        # Picos extremos del saldo
        "max_net_growth_per_sec":    round(float(flow.max()) if len(flow) else 0, 2),
        "max_net_attrition_per_sec": round(float(-flow.min()) if len(flow) else 0, 2),  # magnitud del mínimo (más negativo)
        "peak_growth_at":    peak_growth_row["timestamp"].isoformat()    if peak_growth_row    is not None else None,
        "peak_attrition_at": peak_attrition_row["timestamp"].isoformat() if peak_attrition_row is not None else None,
    }

    # Pre-agregaciones del dashboard

    df_by_minute = (
        df.set_index("timestamp")
        .groupby("metric")
        .resample("1min")
        .agg(
            value=("value", "last"),
            net_flow=("net_flow", "mean"),  
        )
        .reset_index()
    )
    df_by_hour = (
        df.set_index("timestamp")
        .groupby("metric")
        .resample("1h")
        .agg(
            value=("value", "last"),
            net_flow=("net_flow", "mean"),
        )
        .reset_index()
    )

    df_by_minute.to_parquet(OUT_DIR / "by_minute.parquet", index=False)
    df_by_hour.to_parquet(OUT_DIR / "by_hour.parquet", index=False)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"💾 Agregados por minuto y hora guardados.")
    print(f"💾 Summary JSON guardado en {OUT_DIR / 'summary.json'}")

    # ---- Entrenar modelo de predicción ----
    try:
        from model import train_and_save
        train_and_save(df, OUT_DIR)
    except ImportError:
        print("\n⚠️  model.py no encontrado, saltando entrenamiento. Instala scikit-learn: pip install scikit-learn")
    except Exception as e:
        print(f"\n⚠️  Error entrenando modelo: {e}")

    print("\n✅ Ingesta completa.")


if __name__ == "__main__":
    main()