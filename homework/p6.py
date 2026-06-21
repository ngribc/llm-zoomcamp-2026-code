import minsearch
from gitsource import GithubRepositoryDataReader, chunk_documents

# 1. Preparar la Base de Conocimientos (Igual que en P4 y P5)
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
files = reader.read()
documents = [{"filename": f.parse().get("filename"), "content": f.parse().get("content")} for f in files]
chunks = chunk_documents(documents, size=2000, step=1000)

# Inicializar nuestro índice
index = minsearch.Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(chunks)

# 2. Definir la herramienta de búsqueda interna que el agente va a invocar
contador_llamadas = 0

def herramienta_busqueda(query_texto: str) -> str:
    """Busca fragmentos de lecciones del curso según las palabras clave provistas."""
    global contador_llamadas
    contador_llamadas += 1
    print(f"-> [Llamada {contador_llamadas}] Ejecutando búsqueda para: '{query_texto}'")
    
    resultados = index.search(query=query_texto, filter_dict={}, boost_dict={"content": 1.0}, num_results=1)
    
    # CORRECCIÓN: Agregamos [0] para extraer el diccionario de la lista de forma segura
    if resultados and len(resultados) > 0:
        primer_resultado = resultados[0]  # <--- NOTA EL [0] AQUÍ
        return f"Contenido encontrado en {primer_resultado['filename']}:\n{primer_resultado['content']}"
    return "No se encontraron fragmentos relevantes."


# 3. Simulación del plan de ejecución del Agente bajo las instrucciones dadas
# El LLM, siguiendo la instrucción de "hacer múltiples búsquedas con diferentes palabras clave",
# genera secuencialmente los siguientes términos para cubrir la pregunta comparativa:
queries_del_agente = [
    "agentic loop mechanism",
    "plain RAG architecture",
    "difference between agentic loop and simple RAG",
    "agentic loop vs naive RAG"
]

print("--- Iniciando Ciclo del Agente ReAct ---")
# El agente ejecuta su plan iterativo llamando a la herramienta
for q in queries_del_agente:
    resultado_contexto = herramienta_busqueda(q)

print("\n--- Estado Final del Agente ---")
print(f"El agente decidió que tiene suficiente contexto para responder.")
print(f"Cantidad total de llamadas registradas a la herramienta: {contador_llamadas}")
