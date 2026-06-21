import minsearch
from gitsource import GithubRepositoryDataReader

# 1. Extraer los documentos originales
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
documents = [{"filename": f.parse().get("filename"), "content": f.parse().get("content")} for f in files]

# 2. Indexar en minsearch
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
index.fit(documents)

# 3. Buscar usando la consulta oficial en inglés
query = "How does the agentic loop keep calling the model until it stops?"
search_results = index.search(query=query, filter_dict={}, boost_dict={"content": 1.0}, num_results=1)

# CORRECCIÓN DEFINITIVA: Extraemos el primer resultado de la lista usando [0]
if search_results and len(search_results) > 0:
    primer_documento = search_results[0]  # <--- Acceso seguro al diccionario

    # 4. Construir el prompt de la tarea
    context = f"Filename: {primer_documento['filename']}\nContent: {primer_documento['content']}\n\n"
    prompt = f"""
You are a course assistant. Answer the QUESTION based on the CONTEXT from the lesson pages.
Use only the facts from the CONTEXT. If the context doesn't contain the answer, output "I don't know".

QUESTION: {query}

CONTEXT:
{context}
""".strip()

    # 5. Estimación nativa de tokens (1 token ≈ 4 caracteres en inglés)
    tokens_estimados = len(prompt) // 4

    print(f"\n--- Conteo de Tokens (P3) ---")
    print(f"Tokens aproximados en el prompt de entrada: {tokens_estimados}")
else:
    print("No se encontraron documentos en la búsqueda.")
