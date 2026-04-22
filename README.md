# 📊 Rappi · Dashboard de Disponibilidad de Tiendas

Aplicación web para explorar y conversar con datos históricos de disponibilidad de tiendas de Rappi. Combina un dashboard de visualización con un asistente conversacional basado en un modelo de lenguaje grande (LLM) con function calling.

**Stack:** FastAPI (Python) · React + Vite · Recharts · Groq (LLM) · Pandas + Parquet

---

## 🏗️ Arquitectura

Tres capas con responsabilidades separadas:

```
  [201 CSVs crudos]
         │
         ▼
 ┌──────────────┐
 │ INGESTA      │  ← se corre una sola vez (ingest.py)
 │ ingest.py    │     transforma wide→long, deduplica,
 └──────────────┘     calcula la derivada, pre-agrega
         │
         ▼
 ┌─────────────────────────────┐
 │ /data/processed/            │
 │   data.parquet              │  ← ~67K filas crudas
 │   by_minute.parquet         │  ← agregado por minuto
 │   by_hour.parquet           │  ← agregado por hora
 │   forecast.parquet          │  ← modelo de predicción
 │   summary.json              │  ← KPIs globales
 └─────────────────────────────┘
         │
         ▼
 ┌──────────────┐     ┌──────────────┐
 │ BACKEND      │     │ CHATBOT      │
 │ main.py      │────▶│ chatbot.py   │──▶ Groq API (LLM)
 │ FastAPI      │     │ function     │
 │              │     │ calling      │
 └──────────────┘     └──────────────┘
         │
         ▼
 ┌──────────────┐
 │ FRONTEND     │
 │ React+Vite   │
 └──────────────┘
```

---

## 📁 Estructura del repositorio

```
rappi-dashboard/
├── backend/
│   ├── venv/                  # entorno virtual Python
│   ├── ingest.py              # script de ingesta (correr una vez)
│   ├── main.py                # API FastAPI
│   ├── chatbot.py             # agente conversacional
│   ├── model.py               # modelo para la predicción de cierres de tiendas
│   ├── .env                   # GROQ_API_KEY 
│   └── .env.example           # plantilla
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # dashboard principal
│   │   ├── App.css
│   │   ├── ChatBot.jsx        # widget flotante del chat
│   │   ├── ChatBot.css
│   │   ├── api.js             # capa única de llamadas al backend
│   │   ├── ForecastChart.jsx
│   │   ├── ForecastChart.css
│   │   └── index.css
│   └── package.json
└── data/
    ├── raw/                   # 201 CSVs de entrada
    └── processed/             # Parquets + summary.json (generados)
```

---

## 🚀 Cómo correr la aplicación

### Pre-requisitos
- Python 3.10+
- Node.js 18+
- API key gratuita de [Groq](https://console.groq.com) (sin tarjeta de crédito)

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install fastapi uvicorn pandas pyarrow pydantic python-dotenv groq
```

Crear archivo `.env` con tu API key de Groq:
```
GROQ_API_KEY=gsk_tu_key_aqui
```

Correr la ingesta (una sola vez, tarda ~15 segundos):
```bash
python ingest.py
```

Arrancar la API:
```bash
uvicorn main:app --reload
```

API disponible en `http://localhost:8000` · Documentación interactiva en `http://localhost:8000/docs`

### 2. Frontend

En otra terminal:
```bash
cd frontend
npm install
npm run dev
```

Dashboard en `http://localhost:5173`

---

## 🧠 Decisiones técnicas clave

### 1. Formato de datos: wide → long

Los CSVs tienen 1 fila con 367 columnas (formato wide, donde cada timestamp es una columna separada). Este formato es imposible de graficar directamente. La ingesta lo pivotea a formato long usando `pd.melt`:

```python
long_df = df.melt(
    id_vars=id_cols,             # metadata
    value_vars=time_cols,        # las 363 columnas de tiempo → filas
    var_name="timestamp_raw",
    value_name="value",
)
```

Resultado: de `(1 × 367)` a `(363 × 3)` por archivo, con columnas `timestamp | metric | value`.

### 2. Feature derivada: tasa de cambio

El `value` crudo es un **contador acumulado** — graficarlo directamente produce una línea monótonamente creciente, poco informativa. Los patrones operacionales relevantes (horas pico, cambios abruptos) solo se revelan en su **derivada**:

```python
df["rate_per_sec"] = df["value"].diff() / df["timestamp"].diff().dt.total_seconds()
```

Analogía: se pasa del **odómetro** (kilómetros totales) al **velocímetro** (kilómetros por hora en este instante).

### 3. Métrica bidireccional

`rate_per_sec` puede ser positivo (aumentos del contador = aperturas) o negativo (disminuciones = cierres). Se separan en dos columnas para facilitar agregaciones:

```python
df["rate_up"]   = df["rate_per_sec"].where(df["rate_per_sec"] > 0, 0)
df["rate_down"] = (-df["rate_per_sec"]).where(df["rate_per_sec"] < 0, 0)
```

El dashboard muestra **ambos signos simultáneamente** con un eje de referencia en 0, revelando el pulso bidireccional del negocio (no solo el crecimiento neto).

### 4. Pre-agregados con `max` para consistencia visual

Los agregados por minuto/hora usan `max` para picos y `mean` para flujo neto:

```python
df_by_hour = df.resample("1h").agg(
    rate_up=("rate_up", "max"),       # pico del bucket
    rate_down=("rate_down", "max"),
    rate_net=("rate_per_sec", "mean") # promedio neto
)
```

Esto garantiza que el gráfico temporal sea **coherente con la tabla "Top 10 picos"**: si la tabla muestra un pico de +19.532 a las 8:40 pm, el gráfico también lo muestra.

### 5. Datos en memoria al arrancar

Al iniciar, el backend carga los Parquets en memoria global:

```python
@asynccontextmanager
async def lifespan(app):
    DATA["raw"] = pd.read_parquet(...)
    DATA["by_minute"] = pd.read_parquet(...)
    DATA["by_hour"] = pd.read_parquet(...)
    yield
```

Cada request HTTP lee de un DataFrame en RAM → respuestas en milisegundos, incluso con 67 mil puntos.

### 6. Downsampling automático

El endpoint `/api/timeseries` limita a 3000 puntos máximo (Recharts se degrada arriba de eso) y elige la granularidad automáticamente según el rango solicitado:

```python
if granularity == "auto":
    df = DATA["by_minute"]
    if len(df) > 3000:
        df = DATA["by_hour"]        # cambia a granularidad gruesa
```

### 7. Detección de anomalías con z-score ajustado por hora

El detector compara cada punto contra el promedio **de la misma hora del día**, no contra el promedio global:

```python
stats = df.groupby("hour")["rate_per_sec"].agg(hourly_mean="mean", hourly_std="std")
df["z_score"] = (df["rate_per_sec"] - df["hourly_mean"]) / df["hourly_std"]
```

**Por qué:** las 8 pm siempre son pico de actividad. Comparar contra un promedio global produciría muchos falsos positivos. Ajustando por hora, se encuentran los momentos que son inusuales *dentro de su propio contexto horario*.

---

## 🤖 El chatbot conversacional

### Arquitectura: function calling sobre pandas

Cuando el usuario hace una pregunta, el LLM no responde "de memoria". En su lugar, **invoca funciones que ejecutan pandas sobre el DataFrame real**:

```
Usuario: "¿En qué franja horaria hay más aperturas?"
    ↓
LLM razona: necesito escanear las 24 horas
    ↓
LLM invoca: rank_by_dimension(dimension="hour", metric="pico_apertura")
    ↓
Backend ejecuta la función sobre el DataFrame
    ↓
Backend devuelve: [{hora: "20:00", valor: 19532}, ...]
    ↓
LLM redacta respuesta en español con los números exactos
```

Esto garantiza que **cada cifra que el chatbot reporta viene literalmente del dataset**, no de la "imaginación" del modelo. Cero alucinaciones numéricas.

### Las 7 funciones disponibles para el LLM

| Función | Clase de pregunta que resuelve |
|---|---|
| `get_overview` | "¿Qué tenemos en los datos?" |
| `get_hour_stats` | "¿Qué pasa a las 8 pm?" |
| `get_day_stats` | "¿Cómo son los domingos?" |
| `find_extreme_events` | "¿Cuándo hubo más aperturas?" |
| `compare_two_periods` | "Compara el lunes con el martes" |
| **`rank_by_dimension`** | "¿Qué hora tiene más X?" (escanea las 24 o los 7) |
| `get_date_range_stats` | "¿Cómo fue del 3 al 5 de febrero?" |

La función más expresiva es `rank_by_dimension`: soporta 7 métricas × 2 dimensiones = 14 combinaciones de ranking distintas con una sola interfaz.

### System prompt dinámico

El contexto que recibe el LLM se construye en cada request con los KPIs actuales del dataset:

```python
def build_system_prompt(summary):
    m = list(summary["metrics"].values())[0]
    return f"""...
    Rango: {p['start']} → {p['end']}
    Contador al final: {int(m['value_end']):,}
    Mayor aumento registrado: +{int(m['max_rate_up']):,}
    ..."""
```

Si los datos se actualizan, el chatbot se adapta automáticamente sin modificar código.

### Robustez ante fallos del modelo

El sistema intenta primero con el modelo principal; si falla por razones específicas del modelo, reintenta automáticamente con un modelo secundario:

```python
models_to_try = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
for current_model in models_to_try:
    try:
        return _run_agent_loop(...)
    except Exception:
        continue
```

---

## 🤖 Uso de Inteligencia Artificial en el desarrollo

La inteligencia artificial se usó en dos dimensiones distintas del proyecto: **(a)** como herramienta de productividad durante el desarrollo, y **(b)** como componente funcional dentro de la aplicación entregada (el chatbot). Ambos usos respondieron a decisiones explícitas, no a adopción automática.

### Herramientas utilizadas

| Herramienta | Para qué | Por qué |
|---|---|---|
| **Claude (Anthropic)** | Diálogo arquitectónico, revisión de decisiones técnicas, depuración | Modelo fuerte en razonamiento, bueno para discutir compensaciones de diseño |
| **Groq (GPT-OSS 120B + Llama 3.3)** | LLM que potencia el chatbot en la aplicación final | Gratuito, function calling nativo, latencia muy baja por hardware LPU |
| **GitHub Copilot / asistencia en editor** | Autocompletado de fragmentos conocidos | Acelera tareas repetitivas sin reemplazar criterio |

### Cómo se usó la IA durante el desarrollo

El enfoque fue **iterativo y dialogado**, no generativo. En cada fase del proyecto seguí un ciclo: definir el problema → discutir enfoques con la IA → elegir una dirección con razones → implementar con la ayuda de IA → validar resultado → ajustar.

### Por qué esta aproximación

El uso de IA como copiloto amplifica la productividad, pero cada decisión de arquitectura, cada elección de biblioteca y cada criterio analítico se discutió y se entendió antes de implementarse. El código entregado no es código generado a ciegas: es código decidido con criterio propio y escrito con asistencia.

### El LLM dentro de la aplicación: Groq con function calling

El chatbot usa Groq como proveedor de LLM. Groq ejecuta modelos open-source (GPT-OSS 120B como primario, Llama 3.3 70B como respaldo) sobre hardware especializado (LPU) que logra latencias de ~1 segundo por respuesta, notablemente más rápido que proveedores basados en GPU.

La arquitectura es **function calling** (tool use): el LLM no accede directamente a los datos, sino que invoca funciones Python que se ejecutan sobre los DataFrames de pandas. Esto entrega tres beneficios:

1. **Precisión numérica garantizada** — cada cifra proviene de un cálculo determinístico sobre los datos, no de una estimación del modelo.
2. **Seguridad** — el modelo nunca ejecuta código arbitrario; solo puede invocar las funciones que se le exponen explícitamente.
3. **Trazabilidad** — cada respuesta registra qué función se invocó y con qué argumentos, visible en la interfaz.


---

## 📋 Endpoints disponibles

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Health check |
| GET | `/api/summary` | KPIs globales del dataset |
| GET | `/api/timeseries` | Serie temporal bidireccional |
| GET | `/api/peaks` | Top picos de aperturas y cierres |
| GET | `/api/hourly-pattern` | Promedios por hora del día |
| GET | `/api/daily-pattern` | Promedios por día de la semana |
| GET | `/api/heatmap` | Matriz día × hora con flujo neto |
| GET | `/api/anomalies` | Detección de anomalías por z-score |
| POST | `/api/chat` | Chatbot conversacional |

Documentación interactiva disponible en `http://localhost:8000/docs`.

---

## 🎨 Componentes del dashboard

1. **4 KPI cards** — resumen numérico en el encabezado
2. **Gráfico temporal bidireccional** — línea de aperturas arriba y cierres abajo, con eje de referencia en 0
3. **Heatmap día × hora** — matriz de 168 celdas con paleta divergente (rojo para aperturas, azul para cierres, blanco para balance cero)
4. **Bar charts apilados** — patrón por hora del día y por día de la semana, mostrando ambos signos
5. **Tablas lado a lado** — top 10 picos de apertura y top 10 picos de cierre
6. **Tabla de anomalías** — con selector de umbral configurable (2σ, 3σ, 4σ, 5σ)
7. **Chatbot flotante** — widget abajo a la derecha, con preguntas sugeridas e indicador de escritura

---

## 📊 Tecnologías y decisiones

| Capa | Tecnología | Razón |
|---|---|---|
| Procesamiento | Pandas, PyArrow | Estándar de facto para datos tabulares |
| Formato intermedio | Parquet | Mucho más compacto que CSV, lectura rápida |
| API | FastAPI | Tipado, documentación automática, async nativo |
| LLM | Groq (GPT-OSS 120B + Llama 3.3 como respaldo) | Gratuito, rápido, function calling nativo |
| Frontend | React + Vite | HMR rápido, ecosistema maduro |
| Gráficos | Recharts | Declarativo, liviano, suficiente para el caso |
| Estilos | CSS con variables | Sin dependencias, control total |
| Validación | Pydantic | Se integra nativamente con FastAPI |
