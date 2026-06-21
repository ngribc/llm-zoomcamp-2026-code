from gitsource import GithubRepositoryDataReader, chunk_documents

# 1. Extraer los documentos originales del repositorio de DataTalksClub
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []
for file in files:
    doc = file.parse()
    documents.append({
        "filename": doc.get("filename", ""),
        "content": doc.get("content", "")
    })

# 2. Aplicar la función chunk_documents con los parámetros exactos
chunks = chunk_documents(documents, size=2000, step=1000)

# 3. Imprimir el total de fragmentos (chunks) generados
print(f"Total de chunks generados: {len(chunks)}")
