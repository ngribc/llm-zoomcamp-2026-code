import minsearch
from gitsource import GithubRepositoryDataReader, chunk_documents

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

# 2. Aplicar fragmentación
chunks = chunk_documents(documents, size=2000, step=1000)

# 3. Indexar los chunks en minsearch
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
index.fit(chunks)

# 4. Consulta oficial en inglés
query = "How does the agentic loop keep calling the model until it stops?"
search_results = index.search(query=query, filter_dict={}, boost_dict={"content": 1.0}, num_results=1)

# Tomamos el primer fragmento de la lista de resultados
primer_chunk = search_results[0]  # <--- Corrección aquí

# 5. Construcción del Prompt RAG
context = f"Filename: {primer_chunk['filename']}\nContent: {primer_chunk['content']}\n\n"
prompt = f"""
You are a course assistant. Answer the QUESTION based on the CONTEXT from the lesson pages.
Use only the facts from the CONTEXT. If the context doesn't contain the answer, output "I don't know".

QUESTION: {query}

CONTEXT:
{context}
""".strip()

# En promedio, 1 token equivale a 4 caracteres de texto en inglés
tokens_estimados = len(prompt) // 4

print(f"\n--- Estimación de Reducción ---")
print(f"Tokens estimados en P3 (Sin fragmentación): ~7000 tokens")
print(f"Tokens estimados en P5 (Con fragmentación): {tokens_estimados} tokens")
print(f"Reducción calculada: {7000 / tokens_estimados:.1f}× menos")
