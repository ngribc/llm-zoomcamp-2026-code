"""
CÓDIGO PRINCIPAL DEL TRABAJO PRÁCTICO DE MONITOREO (LLM-ZOOMCAMP)
Este script implementa instrumentación manual con OpenTelemetry,
guarda los traces de ejecución en una base de datos local SQLite
y realiza consultas RAG deterministas utilizando la API de Groq.
"""

# =====================================================================
# 📦 PARTE 1: IMPORTACIÓN DE LIBRERÍAS Y CONTROL DEL ENTORNO
# =====================================================================

import os        # Librería nativa para interactuar con variables de entorno del Sistema Operativo
import sys       # Librería nativa para manejar parámetros del sistema y flujos de terminal
import sqlite3   # Motor integrado de base de datos relacional SQLite (sin servidores externos)
from dotenv import load_dotenv  # Componente para leer archivos ocultos de configuración (.env)

# Imprimimos el primer punto de control en la consola
# flush=True obliga a la terminal a mostrar el texto inmediatamente sin almacenamiento previo en buffer
print("[1/5] Cargando variables de entorno...", flush=True)
load_dotenv()  # Busca el archivo '.env' y carga las claves de la API en la memoria de Python

from groq import Groq  # Cliente oficial del SDK de Groq para interactuar con modelos Llama
from gitsource import GithubRepositoryDataReader  # Herramienta del curso para descargar datos de GitHub
from minsearch import Index  # Motor de búsqueda de texto local en memoria desarrollado para el Zoomcamp
from rag_helper import RAGBase  # Clase base provista por los profesores que estructura el flujo RAG

# =====================================================================
# 🛠️ PARTE 2: CONFIGURACIÓN E IMPLEMENTACIÓN DE OPENTELEMETRY
# =====================================================================

print("[2/5] Inicializando OpenTelemetry con SQLite...", flush=True)

# Importamos las herramientas de telemetría del estándar de la industria (OpenTelemetry)
from opentelemetry import trace  # Interfaz global de OTel para manejar trazas y spans
from opentelemetry.sdk.trace import TracerProvider  # El motor o "cerebro" central de telemetría de la app
# Cargamos los componentes para procesar los datos y las clases para definir nuestro exportador
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

class SQLiteSpanExporter(SpanExporter):
    """
    Exportador personalizado de OpenTelemetry (Requisito de la Q4).
    Intercepta cada Span cuando finaliza y guarda sus datos en tablas de SQLite.
    """
    
    def __init__(self, db_path="traces.db"):
        """
        Constructor del exportador. Se ejecuta al inicializar la clase.
        Variable 'db_path': Almacena la ruta del archivo de la base de datos (por defecto 'traces.db').
        """
        # Variable 'self.conn': Almacena el objeto de conexión activa con el archivo SQLite
        self.conn = sqlite3.connect(db_path)
        
        # Ejecutamos una sentencia SQL para asegurarnos de que la tabla 'spans' exista con sus 6 columnas
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,              -- Nombre de la operación rastreada (rag, search, llm)
                start_time INTEGER,     -- Marca de tiempo de inicio de la función en nanosegundos
                end_time INTEGER,       -- Marca de tiempo de finalización en nanosegundos
                input_tokens INTEGER,   -- Cantidad de tokens enviados en el Prompt (Métrica Q2)
                output_tokens INTEGER,  -- Cantidad de tokens generados por la IA
                cost REAL               -- Costo financiero aproximado de la operación en dólares
            )
        """)
        self.conn.commit()  # Confirma los cambios estructurales y los escribe físicamente en el disco

    def export(self, spans):
        """
        Método obligatorio llamado automáticamente por OpenTelemetry al finalizar los bloques de código.
        Variable 'spans': Una lista que contiene los objetos de telemetría finalizados (ReadableSpan).
        """
        for span in spans:  # Iteramos sobre cada span individual recibido por el procesador
            # Variable 'attrs': Convierte el objeto de atributos del span en un diccionario estándar de Python
            attrs = dict(span.attributes or {})
            
            # Insertamos los datos técnicos y métricas recolectadas adentro de la tabla SQL de forma ordenada
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,                # Nombre asignado al bloque de código evaluado
                    span.start_time,          # Tiempo exacto del inicio del cronómetro
                    span.end_time,            # Tiempo exacto del fin del cronómetro
                    attrs.get("input_tokens"),   # Extrae la métrica guardada de tokens de entrada
                    attrs.get("output_tokens"),  # Extrae la métrica guardada de tokens de salida
                    attrs.get("cost"),           # Extrae el cálculo económico
                ),
            )
        self.conn.commit()  # Asegura la escritura física de las filas insertadas en el archivo .db
        return SpanExportResult.SUCCESS  # Le avisa al SDK que la exportación se completó de forma exitosa

    def shutdown(self):
        """Método obligatorio para cerrar recursos de forma limpia al apagarse el script."""
        self.conn.close()  # Cierra la conexión activa del archivo de base de datos

    def force_flush(self):
        """Método obligatorio que obliga a vaciar los búferes y escribir todo directo al disco."""
        return True  # Retorna verdadero para indicar que el vaciado inmediato está activo

# Inicializamos la infraestructura global de OpenTelemetry
provider = TracerProvider()  # Instanciamos el proveedor central

# Vinculamos nuestro exportador de SQLite al procesador sincrónico SimpleSpanProcessor
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("traces.db")))
trace.set_tracer_provider(provider)  # Registramos la configuración del proveedor a nivel global

# Variable 'tracer': El objeto rastreador Scope que usaremos para envolver funciones en Spans
tracer = trace.get_tracer("llm-zoomcamp")

# =====================================================================
# 📂 PARTE 3: CARGA DE DOCUMENTOS E INDEXACIÓN LOCAL DEL RAG
# =====================================================================

# Variable 'COMMIT': Hash específico de Git para fijar la versión exacta de las lecciones del curso
COMMIT = "8c1834d"

print("[3/5] Descargando/Leyendo lecciones desde GitHub...", flush=True)

# Variable 'reader': Objeto configurado para escanear y descargar el repositorio de DataTalksClub
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},  # Solo procesa archivos de documentación markdown
    filename_filter=lambda path: "/lessons/" in path,  # Filtra para quedarse solo con carpetas de lecciones
)

# Variable 'documents': Lista de Python que almacena el contenido parseado de las 72 páginas descargadas
documents = [file.parse() for file in reader.read()]

# Variable 'index': El motor de búsqueda local minsearch configurado con campos clave y de texto libre
index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)  # Construye el índice de búsqueda estático en base a los documentos cargados

# =====================================================================
# 🤖 PARTE 4: SUBCLASIFICACIÓN INSTRUMENTADA CON SPANS (RAGTraced)
# =====================================================================

class RAGTraced(RAGBase):
    """
    Subclase inteligente que hereda las propiedades de RAGBase.
    Sobreescribe los métodos principales para medirlos con cronómetros de telemetría (Spans).
    """
    
    def search(self, query, num_results=5):
        """Método de búsqueda envuelto en un Span."""
        # Se abre un Span hijo llamado 'search' mediante un bloque de contexto 'with'
        with tracer.start_as_current_span("search"):
            # Llama al comportamiento clásico del buscador en memoria y retorna las coincidencias
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        """Método que interactúa con la Inteligencia Artificial y registra métricas de tokens (Q2)."""
        # Se abre un Span hijo llamado 'llm' y guardamos la referencia en la variable 'span'
        with tracer.start_as_current_span("llm") as span:
            # Variable 'response': Almacena el objeto devuelto por la API nativa del chat de Groq
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.instructions},  # Inyecta directivas del sistema
                    {'role': 'user', 'content': prompt}               # Inyecta el prompt formateado
                ]
            )
            
            # Verificamos si Groq nos envió el informe del consumo de tokens en los metadatos de respuesta
            if response.usage:
                usage = response.usage  # Variable 'usage': Atajo para el nodo de uso del API
                
                # Variables de conteo numérico de palabras de entrada y salida
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                # Inyectamos de forma manual estas métricas numéricas como atributos dentro del span actual
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                
            return response  # Devuelve la respuesta estructural del cliente al flujo principal

    def rag(self, query):
        """
        Método principal que orquesta todo el pipeline del RAG (Recuperación y Generación).
        'query' (str): La pregunta original que hace el alumno en texto plano.
        """
        # Abre el Span Raíz o Padre llamado 'rag'. Engloba, mide y cronometra toda la operación completa.
        with tracer.start_as_current_span("rag"):
            
            # Llama a tu buscador local minsearch para traer los documentos más relevantes.
            # Nota técnica: Usamos num_results=1 para no exceder los límites de tokens por minuto (TPM) de Groq.
            search_results = self.search(query, num_results=1)
            
            # Toma tu pregunta junto con el documento encontrado en GitHub y los une usando la plantilla del asistente.
            # Genera una única cadena de texto larga lista para enviar a la Inteligencia Artificial.
            prompt = self.build_prompt(query, search_results)
            
            # Envía el prompt empaquetado a la función llm() para que viaje hacia los servidores de Groq.
            response = self.llm(prompt)
            
            # Accede al nodo de elecciones ('choices'), toma la primera opción, extrae el mensaje 
            # y devuelve estrictamente la cadena de caracteres redactada por el Asistente de IA.
            return response.choices[0].message.content

# Imprime el cuarto punto de control en la consola avisando que arranca la conexión externa.
print("[4/5] Configurando el cliente de Groq...", flush=True)

# Busca en las variables del entorno del sistema el token secreto de autenticación.
# El operador 'or' intenta con ambos nombres posibles para evitar fallos si guardaste la clave al revés.
api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Validación de seguridad: Verifica si la variable api_key quedó vacía tras la búsqueda anterior.
if not api_key:
    # Si la clave está ausente, detiene el script lanzando un error descriptivo para que el usuario lo arregle.
    raise ValueError("¡Error! No se encontró ninguna clave API en tu archivo .env")

# Inicializa el objeto cliente oficial de Groq inyectándole de forma explícita la clave API recuperada.
client = Groq(api_key=api_key)

# Instancia nuestro componente inteligente de RAG pasándole:
# index: El buscador de lecciones del curso fitrado en el paso 3.
# llm_client: La conexión activa a Groq que acabamos de crear arriba.
# model: El identificador del modelo vigente 'llama-3.1-8b-instant' que reemplaza al obsoleto del curso.
rag = RAGTraced(index=index, llm_client=client, model="llama-3.1-8b-instant")

# Bloque de ejecución estándar que asegura que el código de abajo solo corra si ejecutas este archivo de forma directa.
if __name__ == "__main__":
    # Imprime el último punto de control indicando el inicio de las pruebas de usuario.
    print("[5/5] Ejecutando la consulta RAG...", flush=True)
    
    # Define en texto plano la pregunta de la tarea sobre el bucle agéntico.
    query = "How does the agentic loop keep calling the model until it stops?"
    
    # Un bucle 'for' que se repite 4 veces seguidas (Variable 'i' toma valores del 1 al 4).
    # Requisito fundamental para simular tráfico real de datos y poder responder las preguntas analíticas de la entrega.
    for i in range(1, 5):
        # Muestra en qué número de iteración va el proceso de prueba.
        print(f"\n--> Corrida número {i}/4...", flush=True)
        
        # Ejecuta la consulta RAG. Al procesarse, las trazas se insertan automáticamente en la tabla SQLite.
        answer = rag.rag(query)
        
    # Mensajes finales informativos que confirman el fin del laboratorio.
    print("\n--- PROCESO FINALIZADO COMPLETAMENTE ---")
    print("Las 4 ejecuciones fueron guardadas en 'traces.db'")
