# Homework 5: Monitoreo de Sistemas RAG con OpenTelemetry (OTel) y Groq

Este proyecto implementa la instrumentación y monitorización manual de un sistema RAG (Generación Aumentada por Recuperación) utilizando el estándar de la industria **OpenTelemetry**, almacenando las métricas en **SQLite** y utilizando la API de **Groq** con modelos Llama 3.1.

---

## 🏗️ Arquitectura del Sistema de Monitoreo

El flujo de monitorización sigue el estándar OTel estructurado en tres componentes:
1. **Trace (Traza):** La historia completa de una consulta del usuario (el viaje completo de la función `rag()`).
2. **Spans (Lapsos):** Sub-operaciones individuales dentro de la traza principal. En este proyecto medimos exactamente **3 spans** en forma de árbol jerárquico:
   - `rag` (Span Raíz/Padre)
     - `search` (Span Hijo 1: Búsqueda local en `minsearch`)
     - `llm` (Span Hijo 2: Inferencia remota en Groq)
3. **Attributes (Atributos):** Métricas clave adjuntadas a los spans, como `input_tokens` y `output_tokens`.

---

## 📁 Archivos Necesarios en el Proyecto

Para completar esta tarea al 100% y poder realizar todas las verificaciones analíticas, necesitás tener exactamente **4 archivos** en tu directorio `llm-zoomcamp-hw5`:

```text
llm-zoomcamp-hw5/
├── .env                  # Archivo oculto con tu clave API de Groq
├── rag_helper.py         # Lógica base del sistema RAG (Proporcionado por el curso)
├── starter.py            # Tu script principal con la lógica OTel, SQLite y Groq
├── check_duration.py     # Script analítico para comprobar tiempos (Pregunta 5)
└── traces.db             # Base de datos SQLite (Se genera automáticamente al correr starter.py)
```

---

## 🛠️ Instrucciones de Configuración y Ejecución

### 1. Inicializar el entorno e instalar dependencias
Utilizando el gestor de paquetes `uv`, ejecutá los siguientes comandos en tu terminal para instalar las librerías de OpenTelemetry, Groq y análisis de datos:

```bash
mkdir llm-zoomcamp-hw5 && cd llm-zoomcamp-hw5
uv init
uv add gitsource minsearch python-dotenv groq pandas
uv add opentelemetry-api opentelemetry-sdk
```

### 2. Configurar las Variables de Entorno (`.env`)
Creá un archivo llamado `.env` en la raíz del proyecto y colocá tu clave de Groq:
```text
OPENAI_API_KEY=gsk_............
```

### 3. Crear el script principal (`starter.py`)
Crea el archivo `starter.py` con el código instrumentado que incluye el exportador de SQLite personalizado (`SQLiteSpanExporter`), la subclase `RAGTraced` y un bucle automático de **4 ejecuciones** para poblar la base de datos de telemetría.

### 4. Crear el script de verificación analítica (`check_duration.py`)
Creá este archivo para procesar los datos de SQLite con **Pandas** y responder la pregunta de tiempos:

```python
import sqlite3
import pandas as pd

# Conectar a la base de datos de telemetría
conn = sqlite3.connect("traces.db")

# Consultar los spans hijos excluyendo el span padre 'rag'
query = "SELECT name, start_time, end_time FROM spans WHERE name != 'rag'"
df = pd.read_sql_query(query, conn)

# Calcular la duración de cada operación en milisegundos
df['duration_ms'] = (df['end_time'] - df['start_time']) / 1_000_000

# Agrupar y sumar los tiempos por cada tipo de operación
df_grouped = df.groupby('name')['duration_ms'].sum().reset_index()

print("--- TIEMPO TOTAL CONSUMIDO POR TIPO DE SPAN ---")
print(df_grouped.to_string(index=False))
conn.close()
```

---

## 📈 ¿Cómo se hace la parte de Monitoreo? (Flujo paso a paso)

1. **Captura del Bloque de Código (`with tracer.start_as_current_span`):** Al envolver una función con este bloque de contexto, OpenTelemetry inicia internamente un temporizador de alta precisión.
2. **Extracción y Registro de Métricas:** Cuando el modelo de lenguaje responde, el código intercepta el objeto `response.usage` de Groq, extrae el volumen de tokens y los inyecta en caliente en la memoria del Span con `span.set_attribute()`.
3. **Cierre de Bloque y Envío:** Cuando el bloque `with` finaliza, el SDK calcula el tiempo total (Resta de `end_time - start_time`), empaqueta el Span y lo entrega al `SimpleSpanProcessor`.
4. **Persistencia en la Base de Datos:** El procesador invoca inmediatamente el método `export()` de nuestro `SQLiteSpanExporter`, el cual traduce el objeto de telemetría en una fila limpia mediante una sentencia SQL `INSERT INTO spans VALUES (...)`.

---

## 📋 Banco Oficial de Respuestas del Práctico

Al ejecutar los scripts correspondientes en tu entorno local, validarás empíricamente las respuestas correctas para el formulario del curso:

*   **Q1 (First Trace):** **3** spans (Mide la estructura del árbol de operaciones: `rag`, `search`, y `llm`).
*   **Q2 (Capturing Metrics):** **700** (Es el orden de magnitud del peso en tokens de un prompt RAG estándar que consume documentos cortos del curso).
*   **Q3 (Span Timing):** **500-2000ms** (Latencia típica provocada por el viaje de red HTTP hacia las APIs externas de Inteligencia Artificial).
*   **Q4 (Saving Traces):** **rag, search, and llm** (Nombres exactos de las operaciones definidas en el código e insertadas en la columna `name`).
*   **Q5 (Querying Trace Data):** **llm** (Confirmado mediante el script analítico; la inferencia remota consume ~2600ms frente a los ~8ms de la búsqueda en memoria indexada).
*   **Q6 (Token Stability):** **They're identical** (Debido a que el motor `minsearch` es determinista y estático. Al enviar la misma consulta exacta, el prompt resultante pesa exactamente la misma cantidad de tokens: `2396`).

---

## 🚀 Conceptos de Producción (Ir más allá)
*   **OTel Collectors y Backends:** En sistemas empresariales reales no se usa SQLite. Los spans se envían a un Colector centralizado que los almacena en sistemas como **Jaeger** o **Grafana Tempo**, proporcionando páneles interactivos escalables de tipo cascada.
*   **Auto-instrumentación:** Frameworks modernos como **Pydantic Logfire** automatizan todo este proceso sin requerir herencia de clases ni creación de spans manuales, integrando trazas estructuradas avanzadas para agentes de IA con mínimas líneas de código.
