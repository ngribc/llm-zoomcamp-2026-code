"""Starter code for the homework with explicit checkpoints."""

import os
import sys
from dotenv import load_dotenv

print("[1/5] Cargando variables de entorno...", flush=True)
load_dotenv()

from groq import Groq 
from gitsource import GithubRepositoryDataReader
from minsearch import Index
from rag_helper import RAGBase

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

print("[2/5] Inicializando OpenTelemetry...", flush=True)
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")

COMMIT = "8c1834d"

print("[3/5] Descargando/Leyendo lecciones desde GitHub (esto puede demorar)...", flush=True)
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

class RAGTraced(RAGBase):
    def search(self, query, num_results=5):
        print(" -> Entrando a search()...", flush=True)
        with tracer.start_as_current_span("search"):
            res = super().search(query, num_results=num_results)
            print(" -> search() completado.", flush=True)
            return res

    def llm(self, prompt):
        print(" -> Entrando a llm(). Llamando a Groq...", flush=True)
        with tracer.start_as_current_span("llm") as span:
            # Corrección de sintaxis nativa de Groq
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.instructions},
                    {'role': 'user', 'content': prompt}
                ]
            )
            print(" -> Groq respondió correctamente.", flush=True)
            
            if response.usage:
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                
                print(f"\n=========================================")
                print(f"--> ¡CANTIDAD DE INPUT TOKENS REAL!: {input_tokens}")
                print("=========================================\n", flush=True)
                
            return response

    def rag(self, query):
        print(" -> Iniciando flujo completo de RAG...", flush=True)
        with tracer.start_as_current_span("rag"):
            # AGREGAMOS num_results=1 PARA REDUCIR EL TAMAÑO DEL PROMPT
            search_results = self.search(query, num_results=1)
            prompt = self.build_prompt(query, search_results)
            response = self.llm(prompt)
            return response.choices[0].message.content


print("[4/5] Configurando el cliente de Groq...", flush=True)
api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("¡Error! No se encontró ninguna clave API en tu archivo .env")

client = Groq(api_key=api_key)
# Forzamos un modelo estable de Groq (Llama 3 8B)
# Corrección exacta con puntos:
rag = RAGTraced(index=index, llm_client=client, model="llama-3.1-8b-instant")

if __name__ == "__main__":
    print("[5/5] Ejecutando la consulta RAG...", flush=True)
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print("\n--- RESPUESTA FINAL DEL MODELO ---")
    print(answer, flush=True)


# =====================================================================
# CONFIGURACIÓN DE OPENTELEMETRY CON EXPORTADOR SQLITE (REQUISITO Q4)
# =====================================================================
import sqlite3
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

class SQLiteSpanExporter(SpanExporter):
    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True

# Inicializamos el proveedor de OpenTelemetry usando nuestro nuevo exportador
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("traces.db")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")
# =====================================================================
