from gitsource import GithubRepositoryDataReader
# 1. Configurar el lector para el repositorio y commit específico
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
# 2. Descargar los archivos del repositorio
files = reader.read()

# 3. Parsear el contenido de los archivos y guardarlos en una lista
documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

# 4. Imprimir la cantidad total de páginas encontradas
print(f"Total de páginas de clase encontradas: {len(documents)}")