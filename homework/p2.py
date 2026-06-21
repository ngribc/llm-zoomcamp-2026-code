import minsearch
from gitsource import GithubRepositoryDataReader

# 1. Extraer los documentos asegurando la estructura plana
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

print("Descargando archivos desde el repositorio...")
files = reader.read()

documents = []
for file in files:
    doc = file.parse()
    documents.append({
        "filename": doc.get("filename", ""),
        "content": doc.get("content", "")
    })

print(f"Documentos cargados exitosamente: {len(documents)}")

# 2. Inicializar e indexar en minsearch
index = minsearch.Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)
index.fit(documents)

# 3. Consulta oficial en INGLÉS para forzar el match con el contenido del dataset
query_ingles = "How does the agentic loop keep calling the model until it stops?"

results = index.search(
    query=query_ingles,
    filter_dict={},
    boost_dict={"content": 1.0},
    num_results=1
)

# 4. Mostrar el primer resultado de la lista de forma segura usando [0]
if results and len(results) > 0:
    primer_resultado = results[0]  # Acceso correcto al diccionario dentro de la lista
    print("\n--- ¡Búsqueda Exitosa! ---")
    print(f"El primer resultado es: {primer_resultado['filename']}")
else:
    print("\nEl índice no pudo emparejar la consulta con ningún documento.")
