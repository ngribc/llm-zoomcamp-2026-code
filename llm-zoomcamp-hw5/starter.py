"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""

import os
import sys
import sqlite3
from dotenv import load_dotenv

print("[1/5] Cargando variables de entorno...", flush=True)
load_dotenv()

from groq import Groq 
from gitsource import GithubRepositoryDataReader
from minsearch import Index
from rag_helper import RAGBase

# --- CONFIGURACIÓN ÚNICA DE OPENTELEMETRY (SQLITE EXPORTER) ---
print("[2/5] Inicializando OpenTelemetry con SQLite...", flush=True)
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

# Configuramos el proveedor una única vez de manera correcta
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("traces.db")))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("llm-zoomcamp")
# -------------------------------------------------------------

COMMIT = "8c1834d"

print("[3/5] Descargando/Leyendo lecciones desde GitHub...", flush=True)
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
        with tracer.start_as_current_span("search"):
            return super().search(query, num_results=num_results)

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.instructions},
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            if response.usage:
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                span.set_attribute("input_tokens", input_tokens)
                span.set_attribute("output_tokens", output_tokens)
                
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag"):
            # Mantenemos num_results=1 para respetar la cuota gratuita de Groq
            search_results = self.search(query, num_results=1)
            prompt = self.build_prompt(query, search_results)
            response = self.llm(prompt)
            return response.choices[0].message.content

print("[4/5] Configurando el cliente de Groq...", flush=True)
api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("¡Error! No se encontró ninguna clave API en tu archivo .env")

client = Groq(api_key=api_key)
rag = RAGTraced(index=index, llm_client=client, model="llama-3.1-8b-instant")

if __name__ == "__main__":
    print("[5/5] Ejecutando la consulta RAG...", flush=True)
    query = "How does the agentic loop keep calling the model until it stops?"
    
    # Lo ejecutamos 4 veces para poblar la base de datos (Requisito Q6)
    for i in range(1, 5):
        print(f"\n--> Corrida número {i}/4...")
        answer = rag.rag(query)
        
    print("\n--- PROCESO FINALIZADO COMPLETAMENTE ---")
    print("Las 4 ejecuciones fueron guardadas en 'traces.db'")
