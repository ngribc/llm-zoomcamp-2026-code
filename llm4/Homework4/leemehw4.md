# 🚀 LLM Zoomcamp 2026 – Homework 4

## Evaluación de Sistemas de Recuperación de Información mediante Búsqueda por Texto, Búsqueda Vectorial y Búsqueda Híbrida (RRF)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Curso](https://img.shields.io/badge/DataTalksClub-LLM%20Zoomcamp-green)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini-orange)
![RAG](https://img.shields.io/badge/RAG-Retrieval%20Evaluation-red)

---

# 📖 Índice

* Introducción
* Objetivos del proyecto
* Estructura del repositorio
* Tecnologías utilizadas
* Instalación
* Configuración del entorno
* Arquitectura del proyecto
* Construcción de la Base de Conocimiento
* Chunking de documentos
* Generación automática de preguntas
* Construcción del Ground Truth
* Búsqueda por palabras clave
* Búsqueda vectorial
* Reciprocal Rank Fusion (RRF)
* Búsqueda híbrida
* Métricas de evaluación
* Resultados
* Conclusiones
* Mejoras futuras

---

# 📚 Introducción

Este repositorio contiene mi solución del **Homework 4** del curso **LLM Zoomcamp 2026**, organizado por **DataTalksClub**.

El objetivo principal de esta práctica es estudiar uno de los componentes más importantes de un sistema moderno de **Retrieval-Augmented Generation (RAG)**:

> **La recuperación de información (Information Retrieval).**

Cuando utilizamos herramientas como ChatGPT con documentos propios, asistentes empresariales o chatbots sobre bases de conocimiento, el modelo de lenguaje normalmente **no responde únicamente con la información aprendida durante su entrenamiento**.

Antes de generar una respuesta ocurre un proceso adicional:

1. El sistema recibe una pregunta.
2. Busca los documentos más relevantes dentro de una base de conocimiento.
3. Recupera los fragmentos más importantes.
4. Finalmente el LLM utiliza esos fragmentos como contexto para generar la respuesta.

Este mecanismo recibe el nombre de **Retrieval-Augmented Generation (RAG)**.

Por esta razón, **la calidad de la recuperación de información es tan importante como la calidad del propio modelo de lenguaje**.

Si el sistema recupera documentos incorrectos, incluso el mejor LLM generará respuestas incorrectas.

---

# 🎯 Objetivos del proyecto

En este Homework se desarrolla un pequeño framework para evaluar distintos métodos de recuperación de información utilizando la documentación oficial del curso LLM Zoomcamp como base de conocimiento.

Los objetivos principales son:

* Descargar automáticamente la documentación del curso desde GitHub.
* Construir una base de conocimiento estructurada.
* Dividir documentos largos en fragmentos (chunks).
* Generar preguntas automáticamente utilizando un LLM.
* Crear un conjunto de evaluación (Ground Truth).
* Implementar distintos métodos de búsqueda.
* Comparar los resultados obtenidos.
* Evaluar objetivamente la calidad de cada método mediante métricas estándar de Information Retrieval.

Al finalizar el proyecto podremos comparar tres estrategias diferentes:

* 🔎 Búsqueda por palabras clave (Keyword Search)
* 🧠 Búsqueda semántica mediante embeddings (Vector Search)
* ⚡ Búsqueda híbrida utilizando Reciprocal Rank Fusion (Hybrid Search)

---

# 📁 Estructura del proyecto

```text
.
├── homework4.ipynb
├── README.md
├── README_ES.md
├── requirements.txt
├── .env
└── data/
```

El notebook concentra todo el flujo de trabajo del proyecto:

* descarga de la documentación,
* preprocesamiento,
* generación de preguntas,
* implementación de los buscadores,
* evaluación,
* comparación de resultados.

---

# 🛠 Tecnologías utilizadas

| Categoría              | Herramienta                     |
| ---------------------- | ------------------------------- |
| Lenguaje               | Python 3.12                     |
| Notebook               | Jupyter Notebook                |
| LLM                    | Groq                            |
| Modelo                 | Llama 3.3 70B Versatile         |
| API alternativa        | Google Gemini                   |
| Base de conocimiento   | Documentación LLM Zoomcamp      |
| Fuente de datos        | GitHub                          |
| Recuperación léxica    | Keyword Search                  |
| Recuperación semántica | Dense Embeddings                |
| Búsqueda híbrida       | Reciprocal Rank Fusion          |
| Evaluación             | Hit Rate y Mean Reciprocal Rank |

---

# ⚙️ Instalación

Clonar el repositorio

```bash
git clone https://github.com/usuario/repositorio.git
```

Ingresar al proyecto

```bash
cd repositorio
```

Crear un entorno virtual

```bash
python -m venv .venv
```

Activar el entorno

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Instalar dependencias

```bash
pip install -r requirements.txt
```

o utilizando **uv**

```bash
uv pip install -r requirements.txt
```

---

# 🔑 Variables de entorno

Para utilizar las APIs de Groq o Gemini es necesario crear un archivo `.env`.

Ejemplo:

```text
GROQ_API_KEY=tu_api_key
GOOGLE_API_KEY=tu_api_key
```

Posteriormente el notebook carga automáticamente estas credenciales mediante:

```python
from dotenv import load_dotenv

load_dotenv()
```

De esta manera las claves permanecen protegidas y no quedan escritas directamente en el código.

---

# 🏗 Arquitectura general del proyecto

El flujo completo implementado durante el Homework puede resumirse mediante el siguiente diagrama.

```text
              Repositorio GitHub
                     │
                     ▼
      Documentación oficial (.md)
                     │
                     ▼
      GithubRepositoryDataReader
                     │
                     ▼
         Documentos procesados
                     │
                     ▼
        División en fragmentos
               (Chunking)
                     │
                     ▼
          Base de Conocimiento
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
 Keyword Search         Vector Search
          │                     │
          └──────────┬──────────┘
                     ▼
      Reciprocal Rank Fusion
                     │
                     ▼
           Búsqueda Híbrida
                     │
                     ▼
              Evaluación
             Hit Rate / MRR
```

Esta arquitectura representa el flujo típico utilizado actualmente por muchos sistemas de **Retrieval-Augmented Generation** en producción.

---

# 📚 Construcción de la Base de Conocimiento

A diferencia de otros proyectos de Machine Learning, en este Homework no se utiliza un archivo CSV como dataset principal.

En su lugar, la información se obtiene directamente desde el repositorio oficial de **LLM Zoomcamp**.

Para ello se utiliza la clase `GithubRepositoryDataReader`.

```python
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
```

Este procedimiento descarga automáticamente todas las lecciones escritas en formato Markdown.

Cada documento contiene información como:

* nombre del archivo,
* ruta dentro del repositorio,
* contenido completo,
* metadatos asociados.

Una vez descargados, estos documentos pasan a formar parte de la **Base de Conocimiento**, que será utilizada por los distintos métodos de recuperación implementados en el proyecto.

---

# 📖 ¿Por qué dividir los documentos?

Las lecciones del curso pueden ser demasiado extensas para ser indexadas como un único documento.

Por ese motivo se aplica una técnica denominada **Chunking**, que consiste en dividir cada documento en pequeños fragmentos con solapamiento.

En lugar de indexar un documento completo, el sistema indexa muchos fragmentos independientes.

Esta estrategia ofrece varias ventajas:

* mejora la precisión de la búsqueda,
* reduce el costo computacional,
* facilita la generación de embeddings,
* conserva mejor el contexto,
* incrementa la calidad de recuperación.

En este proyecto cada fragmento tiene:

* **2000 caracteres de tamaño**
* **1000 caracteres de solapamiento**

Gracias a este solapamiento, la información que aparece cerca de los límites entre fragmentos no se pierde durante la búsqueda.

---

# 🤖 Generación automática de preguntas mediante un LLM

Uno de los aspectos más interesantes de este Homework es que el conjunto de evaluación **no fue creado manualmente**.

En lugar de escribir cientos de preguntas una por una, el notebook utiliza un **Large Language Model (LLM)** para generarlas automáticamente.

La idea consiste en pedirle al modelo que **actúe como un estudiante del curso** y formule preguntas que puedan responderse utilizando únicamente el contenido de una determinada lección.

El prompt utilizado es similar al siguiente:

```python
data_gen_instructions = """
You emulate a student who is taking our LLM course.

You are given one lesson page from the course.

Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer.
- Don't copy the lesson wording.
- Ask naturally.
"""
```

Estas instrucciones buscan obtener preguntas que se parezcan a las que realizaría una persona real mientras estudia.

Por ejemplo, en lugar de preguntar:

> "¿Qué dice el apartado 3.2?"

el modelo puede generar preguntas como:

* ¿Cómo puedo unirme al curso?
* ¿Qué es Agentic RAG?
* ¿Cuál es la diferencia entre Retrieval y Generation?

Este tipo de preguntas resulta mucho más útil para evaluar un sistema RAG.

---

# 🚀 ¿Por qué utilizar un LLM para generar preguntas?

Crear un conjunto de evaluación manual presenta varios inconvenientes.

Si el curso contiene decenas de lecciones, escribir cientos de preguntas requiere mucho tiempo.

Además:

* diferentes personas escribirían preguntas distintas;
* es difícil mantener un estilo consistente;
* ampliar el dataset supone un trabajo considerable.

El uso de un LLM permite generar automáticamente preguntas realistas con un coste muy bajo.

Las ventajas son:

* escalabilidad;
* rapidez;
* consistencia;
* diversidad de formulaciones;
* facilidad para crear nuevos conjuntos de evaluación.

Actualmente este enfoque es habitual en proyectos de evaluación de sistemas RAG.

---

# ⚡ Utilización de Groq y Llama 3.3 70B

Para la generación de preguntas el notebook utiliza la API de **Groq**.

El modelo seleccionado es:

> **Llama-3.3-70B-Versatile**

El flujo general es el siguiente:

```text
Lección
    │
    ▼
Prompt
    │
    ▼
Groq API
    │
    ▼
JSON
    │
    ▼
Preguntas generadas
```

La llamada principal tiene una estructura similar a:

```python
client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    ...
)
```

El parámetro

```python
temperature = 0
```

permite obtener respuestas deterministas.

Esto es importante porque hace que el experimento sea reproducible.

Si ejecutamos nuevamente el notebook bajo las mismas condiciones, obtendremos prácticamente las mismas preguntas.

---

# 📦 Respuestas estructuradas mediante Pydantic

Una dificultad frecuente al trabajar con modelos de lenguaje consiste en procesar correctamente sus respuestas.

Si el modelo responde con texto libre, posteriormente es necesario analizar la salida manualmente.

Para evitar este problema se utiliza **Pydantic**.

```python
class Questions(BaseModel):
    questions: list[str]
```

Gracias a este esquema el modelo devuelve un objeto JSON con un formato conocido.

Ejemplo:

```json
{
    "questions": [
        "...",
        "...",
        "...",
        "...",
        "..."
    ]
}
```

Las ventajas son numerosas:

* validación automática;
* menor probabilidad de errores;
* código más limpio;
* integración sencilla con el resto del pipeline.

---

# 🔢 Análisis del consumo de tokens

El notebook también registra la cantidad de tokens consumidos durante las llamadas al modelo.

```python
total_input_tokens += response.usage.prompt_tokens
```

Posteriormente calcula el promedio de tokens utilizados por documento.

Este análisis resulta importante porque:

* permite estimar el coste económico;
* ayuda a optimizar prompts;
* reduce tiempos de respuesta;
* facilita comparar distintos modelos.

En proyectos reales este tipo de métricas suele monitorizarse continuamente.

---

# 📄 Selección de las lecciones

Para simplificar el experimento, las preguntas no se generan sobre toda la documentación del curso.

El notebook selecciona únicamente las primeras lecciones relacionadas con Agentic RAG.

```python
target_lessons = {
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md"
}
```

Estas páginas contienen los conceptos fundamentales necesarios para responder las preguntas del Homework.

---

# 📊 Construcción del Ground Truth

Una vez generadas las preguntas, es necesario conocer cuál es el documento correcto para responder cada una de ellas.

Este conjunto recibe el nombre de **Ground Truth**.

Cada registro contiene al menos dos elementos:

| Campo    | Descripción        |
| -------- | ------------------ |
| question | Pregunta generada  |
| filename | Documento correcto |

Ejemplo:

| Pregunta                     | Documento esperado |
| ---------------------------- | ------------------ |
| ¿Cómo puedo unirme al curso? | 01-intro.md        |

El Ground Truth constituye la referencia contra la cual posteriormente se evaluarán todos los buscadores.

---

# 🔍 Búsqueda por palabras clave (Keyword Search)

La búsqueda por palabras clave representa el enfoque clásico utilizado durante muchos años por motores de búsqueda.

Su funcionamiento es sencillo:

```text
Pregunta
      │
      ▼
Tokenización
      │
      ▼
Coincidencia de palabras
      │
      ▼
Ranking
      │
      ▼
Documentos recuperados
```

El sistema intenta localizar documentos que contengan las mismas palabras presentes en la consulta.

### Ventajas

* muy rápida;
* bajo coste computacional;
* sencilla de interpretar;
* excelente línea base.

### Desventajas

* no comprende significado;
* depende mucho de la redacción de la pregunta;
* no maneja bien sinónimos.

---

# 🧠 Búsqueda vectorial (Vector Search)

La búsqueda vectorial utiliza una filosofía completamente distinta.

En lugar de comparar palabras, convierte tanto las preguntas como los documentos en vectores numéricos denominados **embeddings**.

```text
Pregunta
      │
      ▼
Embedding
      │
      ▼
Espacio vectorial
      │
      ▼
Búsqueda por similitud
      │
      ▼
Documentos similares
```

Gracias a ello puede recuperar documentos aunque no compartan exactamente las mismas palabras.

Por ejemplo:

Pregunta

> ¿Cómo puedo inscribirme?

puede recuperar correctamente un documento cuyo título sea

> Joining the Course

aunque nunca aparezca la palabra "inscribirme".

### Ventajas

* comprende relaciones semánticas;
* tolera paráfrasis;
* recupera documentos conceptualmente similares.

### Desventajas

* requiere mayor capacidad computacional;
* depende del modelo de embeddings;
* resulta más difícil de interpretar.

---

# ⚖️ Comparación entre ambos métodos

| Característica          | Keyword Search | Vector Search |
| ----------------------- | -------------- | ------------- |
| Coincidencia exacta     | ✅              | ❌             |
| Comprensión semántica   | ❌              | ✅             |
| Velocidad               | Muy alta       | Alta          |
| Coste                   | Bajo           | Mayor         |
| Robustez ante sinónimos | Baja           | Alta          |
| Interpretabilidad       | Alta           | Media         |

Ambos enfoques presentan fortalezas y debilidades.

Por esta razón surge una tercera alternativa.

---

# 🔀 Búsqueda Híbrida

La búsqueda híbrida intenta aprovechar lo mejor de ambos mundos.

En lugar de utilizar un único buscador, ejecuta simultáneamente:

* Keyword Search
* Vector Search

Posteriormente fusiona ambos rankings.

```text
                    Consulta
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
 Keyword Search              Vector Search
         │                           │
         ▼                           ▼
 Ranking A                   Ranking B
         └─────────────┬─────────────┘
                       ▼
        Reciprocal Rank Fusion (RRF)
                       ▼
              Ranking Final
```

Esta estrategia suele producir resultados más robustos que cualquiera de los métodos por separado.

---

# 🔢 Reciprocal Rank Fusion (RRF)

El algoritmo encargado de combinar ambos rankings se denomina **Reciprocal Rank Fusion (RRF)**.

La idea es extremadamente simple.

Cada documento recibe una puntuación calculada mediante:

[
\text{Score}=\frac{1}{k+rank}
]

donde:

* **rank** representa la posición del documento.
* **k** controla cuánto peso reciben las primeras posiciones.

La puntuación final corresponde a la suma de las contribuciones de todos los rankings.

[
RRF(d)=\sum_{i=1}^{n}\frac{1}{k+r_i(d)}
]

Gracias a este mecanismo:

* un documento muy bien posicionado por ambos buscadores obtiene una puntuación elevada;
* un documento recuperado únicamente por uno de ellos recibe menor puntuación.

RRF es actualmente una de las técnicas de fusión de rankings más utilizadas en sistemas modernos de búsqueda y en aplicaciones basadas en **Retrieval-Augmented Generation (RAG)** debido a su sencillez, robustez y excelente desempeño práctico.


---

# 📈 Evaluación del sistema de recuperación

Una vez implementados los distintos métodos de búsqueda, el siguiente paso consiste en medir objetivamente su rendimiento.

En Recuperación de Información (Information Retrieval) no basta con obtener resultados; es necesario determinar **qué tan buenos son**.

Para ello, este Homework emplea dos de las métricas más utilizadas en la literatura científica y en sistemas de búsqueda modernos:

* **Hit Rate (Recall@k)**
* **Mean Reciprocal Rank (MRR)**

Estas métricas permiten comparar diferentes algoritmos bajo las mismas condiciones y establecer cuál recupera la información más relevante.

---

# 🎯 Hit Rate

El **Hit Rate** responde a una pregunta muy sencilla:

> **¿El documento correcto aparece entre los primeros resultados recuperados?**

Para calcularlo, el sistema sigue el siguiente procedimiento:

1. Se toma una pregunta del Ground Truth.
2. Se ejecuta el método de búsqueda correspondiente.
3. Se recuperan los primeros documentos (Top-k).
4. Se verifica si el documento esperado se encuentra entre ellos.

Si aparece, se considera un **acierto (Hit)**.

En caso contrario, se registra un fallo.

Matemáticamente:

[
HitRate=\frac{\text{Número de aciertos}}{\text{Número total de consultas}}
]

Ejemplo:

| Consulta | Documento encontrado |
| -------- | -------------------- |
| Q1       | ✅                    |
| Q2       | ✅                    |
| Q3       | ❌                    |
| Q4       | ✅                    |
| Q5       | ✅                    |

Resultado:

[
HitRate=\frac{4}{5}=0.80
]

Un valor cercano a **1** indica que el sistema recupera correctamente los documentos esperados en la mayoría de las consultas.

---

# 🥇 Mean Reciprocal Rank (MRR)

Aunque el Hit Rate indica si el documento correcto aparece, **no considera en qué posición fue encontrado**.

No es lo mismo encontrar la respuesta en el primer lugar que en el décimo.

Para tener en cuenta este aspecto se utiliza el **Mean Reciprocal Rank (MRR)**.

Para cada consulta se calcula:

[
RR=\frac{1}{Posición}
]

Por ejemplo:

| Posición | Reciprocal Rank |
| -------: | --------------: |
|        1 |            1.00 |
|        2 |            0.50 |
|        3 |            0.33 |
|        4 |            0.25 |
|        5 |            0.20 |

Finalmente se obtiene el promedio para todas las consultas:

[
MRR=\frac1N\sum_{i=1}^{N}\frac1{rank_i}
]

Cuanto más próximo sea el MRR a **1**, mejor será la calidad del ranking producido por el sistema.

---

# 🔬 Función de evaluación

El notebook implementa una función que evalúa automáticamente cualquier método de búsqueda.

El flujo general es el siguiente:

```text
Ground Truth
      │
      ▼
Pregunta
      │
      ▼
Motor de búsqueda
      │
      ▼
Top-k documentos
      │
      ▼
Comparación con el documento esperado
      │
      ▼
Cálculo de métricas
```

Gracias a esta estructura es posible evaluar varios buscadores utilizando exactamente el mismo conjunto de preguntas.

Esto garantiza una comparación justa entre todos los métodos implementados.

---

# 📊 Resultados obtenidos

Durante el experimento se compararon distintos enfoques de recuperación.

## 🔎 Keyword Search

La búsqueda basada en palabras clave obtuvo el mejor rendimiento entre los buscadores individuales.

Resultado aproximado:

| Métrica  |    Valor |
| -------- | -------: |
| Hit Rate | **0.76** |

Este comportamiento se explica porque gran parte de las preguntas contienen términos técnicos presentes literalmente en la documentación.

---

## 🧠 Vector Search

La búsqueda semántica obtuvo un rendimiento inferior en este experimento.

Resultado aproximado:

| Métrica  |    Valor |
| -------- | -------: |
| Hit Rate | **0.45** |

Aunque este enfoque comprende el significado de las consultas, su rendimiento depende en gran medida del modelo de embeddings utilizado y de la calidad de la indexación.

---

## 🔀 Hybrid Search

La búsqueda híbrida combina ambos enfoques utilizando Reciprocal Rank Fusion.

Su objetivo consiste en aprovechar simultáneamente:

* la precisión léxica de Keyword Search;
* la comprensión semántica de Vector Search.

En muchos sistemas reales esta estrategia suele producir resultados más robustos que cualquiera de los métodos utilizados individualmente.

---

# ⚙️ Optimización del parámetro *k*

Uno de los objetivos del Homework consiste en estudiar el comportamiento del parámetro **k** utilizado por Reciprocal Rank Fusion.

Se evaluaron distintos valores:

|   k |   MRR |
| --: | ----: |
|   1 | 0.771 |
|  50 | 0.693 |
| 100 | 0.621 |
| 200 | 0.548 |

El mejor resultado se obtuvo con:

# ⭐ **k = 1**

Esto significa que, para este conjunto de datos, resulta beneficioso otorgar un mayor peso a los documentos mejor posicionados por los buscadores individuales.

---

# 📉 Discusión de resultados

Los experimentos permiten extraer varias conclusiones interesantes.

## Keyword Search

### Ventajas

* Muy rápida.
* Bajo coste computacional.
* Fácil de interpretar.
* Excelente línea base.

### Desventajas

* No comprende el significado de las palabras.
* Muy sensible a la redacción de la consulta.
* Dificultad para manejar sinónimos.

---

## Vector Search

### Ventajas

* Comprende similitud semántica.
* Tolera paráfrasis.
* Recupera documentos relacionados conceptualmente.

### Desventajas

* Mayor consumo computacional.
* Dependencia del modelo de embeddings.
* Más difícil de interpretar.

---

## Hybrid Search

### Ventajas

* Combina recuperación léxica y semántica.
* Produce rankings más robustos.
* Reduce errores individuales.
* Se considera una de las mejores estrategias actuales para sistemas RAG.

### Desventajas

* Mayor complejidad de implementación.
* Necesita mantener dos índices distintos.

A pesar de ello, la búsqueda híbrida representa actualmente una de las estrategias más utilizadas en aplicaciones de inteligencia artificial basadas en recuperación de información.

---

# 💡 Aprendizajes obtenidos

Este Homework demuestra que el rendimiento de un sistema RAG **no depende únicamente del modelo de lenguaje**.

Otros componentes tienen una influencia decisiva:

* calidad de la documentación;
* proceso de chunking;
* tamaño de los fragmentos;
* solapamiento entre chunks;
* generación de preguntas;
* calidad del Ground Truth;
* estrategia de recuperación;
* algoritmo de fusión;
* métricas de evaluación.

Una pequeña modificación en cualquiera de estos elementos puede alterar significativamente la calidad de las respuestas obtenidas por el sistema.

---

# 🚀 Posibles mejoras

Aunque el sistema implementado cumple correctamente con los objetivos del Homework, existen múltiples posibilidades de ampliación.

## Embeddings de mayor calidad

Se podrían evaluar modelos como:

* OpenAI text-embedding-3-large
* BGE
* E5
* Instructor XL

---

## Rerankers

Tras recuperar documentos podría añadirse una etapa adicional de reordenamiento utilizando modelos especializados.

Algunas alternativas son:

* BAAI BGE Reranker
* Cohere Rerank
* Jina AI Reranker

---

## Bases de datos vectoriales

En un entorno de producción sería recomendable utilizar motores especializados como:

* Qdrant
* Milvus
* Weaviate
* Pinecone
* Elasticsearch
* pgvector

---

## Filtrado mediante metadatos

También podrían incorporarse filtros por:

* módulo;
* tema;
* nivel de dificultad;
* capítulo;
* tipo de contenido.

---

## Expansión automática de consultas

Otra mejora interesante consiste en utilizar un LLM para reformular automáticamente las preguntas antes de realizar la búsqueda.

Por ejemplo:

```text
Consulta original
        │
        ▼
LLM
(Query Expansion)
        │
        ▼
Múltiples consultas
        │
        ▼
Recuperación
        │
        ▼
Fusión de resultados
```

Esta estrategia suele incrementar considerablemente el Recall del sistema.

---

# 📚 Referencias

* DataTalksClub – LLM Zoomcamp
* Lewis et al. – Retrieval-Augmented Generation (2020)
* Cormack et al. – Reciprocal Rank Fusion (2009)
* Documentación oficial de Groq
* Documentación oficial de Google Gemini
* Pydantic
* Python
* GitHub

---

# ✅ Conclusiones

Este Homework permitió construir un flujo completo para evaluar sistemas de recuperación de información utilizados en aplicaciones modernas de **Retrieval-Augmented Generation (RAG)**.

A partir de la documentación oficial del curso se desarrolló un pipeline que incluyó:

* descarga automática del contenido desde GitHub;
* construcción de una base de conocimiento;
* división de documentos mediante Chunking;
* generación automática de preguntas utilizando un Large Language Model;
* creación del Ground Truth;
* implementación de búsqueda por palabras clave;
* implementación de búsqueda vectorial;
* combinación mediante Reciprocal Rank Fusion;
* evaluación utilizando Hit Rate y Mean Reciprocal Rank;
* análisis del impacto del parámetro **k**.

Los experimentos muestran claramente que **la calidad de un sistema RAG depende tanto del mecanismo de recuperación como del propio modelo de lenguaje**.

Incluso el LLM más avanzado producirá respuestas deficientes si recibe documentos irrelevantes como contexto.

Por ello, la evaluación rigurosa de los sistemas de recuperación constituye una etapa fundamental antes de desplegar cualquier aplicación basada en inteligencia artificial generativa.

---

# 👨‍💻 Autor

Trabajo desarrollado como parte del **LLM Zoomcamp 2026** de **DataTalksClub**.

Este proyecto tiene fines educativos y busca comprender los principios fundamentales de la evaluación de sistemas de recuperación utilizados en aplicaciones modernas de Inteligencia Artificial y Retrieval-Augmented Generation.



