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
    # Los archivos consecutivos comparten 1 segundo en la frontera
    df = df.drop_duplicates(subset=["timestamp", "metric"]).sort_values("timestamp")
    df = df.reset_index(drop=True)
    print(f"🧹 Filas únicas tras dedupe: {len(df):,}")

    # ---- Feature derivada: tasa ----
    # value es un acumulado. La "tasa" es cuánto crece entre un punto y el siguiente.
    # delta / delta_t nos da eventos por segundo -> la métrica operacional interesante.
    df["delta"] = df.groupby("metric")["value"].diff()
    df["delta_seconds"] = df.groupby("metric")["timestamp"].diff().dt.total_seconds()
    df["rate_per_sec"] = df["delta"] / df["delta_seconds"]

    # ---- Limpiar resets del contador ----
    # El contador acumulado se reinicia cada ~24h (probable reset del sistema
    # de monitoreo). Estos resets producen deltas negativos enormes que ensucian
    # los promedios. Los detectamos por "caída relativa" > 30% del valor previo.
    # OJO: oscilaciones pequeñas (-50, -200) son comportamiento normal del
    # monitoreo sintético (tiendas que salen/entran brevemente). NO las filtramos.
    prev_value = df.groupby("metric")["value"].shift(1)
    is_reset = (df["delta"] < 0) & (df["delta"].abs() > 0.3 * prev_value)
    n_resets = int(is_reset.sum())
    df.loc[is_reset, "rate_per_sec"] = pd.NA
    df.loc[is_reset, "delta"] = pd.NA
    print(f"🔄 Resets del contador detectados (caída >30%): {n_resets}")

    # ---- Guardar dataset principal ----
    parquet_path = OUT_DIR / "data.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Dataset guardado en {parquet_path}")
    print(f"   Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"   Métricas: {df['metric'].unique().tolist()}")

    # ---- Columnas bidireccionales ----
    # Reinterpretación: rate_per_sec = cambio neto de tiendas visibles por segundo.
    # Positivo = aperturas netas (tiendas que aparecieron)
    # Negativo = cierres netos (tiendas que desaparecieron)
    # Separamos en dos columnas para facilitar agregaciones y visualización.
    df["rate_up"] = df["rate_per_sec"].where(df["rate_per_sec"] > 0, 0)
    df["rate_down"] = (-df["rate_per_sec"]).where(df["rate_per_sec"] < 0, 0)  # valor absoluto de los negativos
    # Mantenemos rate_positive por compatibilidad, pero el dashboard ya no la usa.
    df["rate_positive"] = df["rate_per_sec"].where(df["rate_per_sec"] >= 0)

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
        rate_series = group["rate_per_sec"].dropna()
        ups = rate_series[rate_series > 0]
        downs = rate_series[rate_series < 0]  # valores negativos
        peak_up_row = group.loc[group["rate_per_sec"].idxmax()] if rate_series.notna().any() else None
        peak_down_row = group.loc[group["rate_per_sec"].idxmin()] if rate_series.notna().any() else None

        summary["metrics"][metric_name] = {
            "value_start": float(group["value"].iloc[0]),
            "value_end": float(group["value"].iloc[-1]),
            "total_growth": float(group["value"].iloc[-1] - group["value"].iloc[0]),

            # Bidireccional
            "avg_rate_up": round(float(ups.mean()) if len(ups) else 0, 2),
            "avg_rate_down": round(float(downs.abs().mean()) if len(downs) else 0, 2),
            "net_flow": round(float(rate_series.mean()), 2),
            "n_up_points": int(len(ups)),
            "n_down_points": int(len(downs)),

            "max_rate_up": round(float(ups.max()) if len(ups) else 0, 2),
            "max_rate_down": round(float(downs.abs().max()) if len(downs) else 0, 2),
            "peak_up_at": peak_up_row["timestamp"].isoformat() if peak_up_row is not None else None,
            "peak_down_at": peak_down_row["timestamp"].isoformat() if peak_down_row is not None else None,

            # Legacy (chatbot puede usarlos)
            "avg_rate_per_sec": round(float(rate_series.mean()), 2),
            "max_rate_per_sec": round(float(rate_series.max()), 2),
            "peak_rate_at": peak_up_row["timestamp"].isoformat() if peak_up_row is not None else None,
        }

    # Pre-agregaciones que el dashboard necesitará
    # Por minuto (para zoom medio) y por hora (para zoom-out)
    # Usamos max para aperturas y cierres (picos puntuales de cada bucket),
    # consistente con la tabla "Top picos". net_flow = diferencia de los máximos.
    df_by_minute = (
        df.set_index("timestamp")
        .groupby("metric")
        .resample("1min")
        .agg(
            value=("value", "last"),
            rate_up=("rate_up", "max"),
            rate_down=("rate_down", "max"),
            rate_net=("rate_per_sec", "mean"),  # promedio neto del bucket
        )
        .reset_index()
    )
    df_by_hour = (
        df.set_index("timestamp")
        .groupby("metric")
        .resample("1h")
        .agg(
            value=("value", "last"),
            rate_up=("rate_up", "max"),
            rate_down=("rate_down", "max"),
            rate_net=("rate_per_sec", "mean"),
        )
        .reset_index()
    )

    df_by_minute.to_parquet(OUT_DIR / "by_minute.parquet", index=False)
    df_by_hour.to_parquet(OUT_DIR / "by_hour.parquet", index=False)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"💾 Agregados por minuto y hora guardados.")
    print(f"💾 Summary JSON guardado en {OUT_DIR / 'summary.json'}")
    print("\n✅ Ingesta completa.")


if __name__ == "__main__":
    main()