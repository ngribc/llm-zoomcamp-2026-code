# 🚀 LLM Zoomcamp 2026 – Homework 4
## Retrieval Evaluation using Keyword Search, Vector Search and Hybrid Search (RRF)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Course](https://img.shields.io/badge/DataTalksClub-LLM%20Zoomcamp-green)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini-orange)
![Retrieval](https://img.shields.io/badge/RAG-Retrieval%20Evaluation-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# 📚 Table of Contents

- [Introduction](#introduction)
- [Project Goals](#project-goals)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Project Architecture](#project-architecture)
- [Knowledge Base](#knowledge-base)
- [Document Chunking](#document-chunking)
- [Question Generation using LLMs](#question-generation-using-llms)
- [Ground Truth Dataset](#ground-truth-dataset)
- [Keyword Search](#keyword-search)
- [Vector Search](#vector-search)
- [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
- [Hybrid Search](#hybrid-search)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Conclusions](#conclusions)
- [Future Improvements](#future-improvements)
- [References](#references)

---

# 📖 Introduction

This repository contains my solution for **Homework 4** of the **LLM Zoomcamp 2026** course organized by **DataTalksClub**.

The objective of this homework is to understand one of the most important components of modern Retrieval-Augmented Generation (RAG) systems:

> **Information Retrieval**

Instead of asking an LLM to answer questions directly, RAG systems first retrieve the most relevant documents from a knowledge base and then use those documents as context for generation.

For this reason, retrieval quality directly impacts the quality of the final answer.

This homework focuses on evaluating different retrieval strategies over the official LLM Zoomcamp course documentation.

---

# 🎯 Project Goals

The notebook implements a complete retrieval evaluation pipeline.

The main goals are:

- Download the official course documentation directly from GitHub.
- Build a searchable knowledge base.
- Split large documents into overlapping chunks.
- Generate realistic questions using an LLM.
- Build a Ground Truth dataset.
- Compare different retrieval approaches.
- Evaluate retrieval quality using Information Retrieval metrics.
- Analyze the strengths and weaknesses of each retrieval strategy.

By the end of this homework we can objectively compare:

- Traditional Keyword Search
- Dense Vector Search
- Hybrid Search using Reciprocal Rank Fusion (RRF)

---

# 📁 Repository Structure

```

.
├── homework4.ipynb
├── README.md
├── requirements.txt
├── .env
└── data/
└── ground_truth.csv

```

The notebook contains every step of the retrieval pipeline, from downloading the documentation to evaluating retrieval performance.

---

# 🛠 Technologies Used

| Category | Technology |
|------------|----------------------------|
| Language | Python 3.12 |
| Notebook | Jupyter |
| LLM | Groq |
| Model | Llama 3.3 70B Versatile |
| Alternative API | Google Gemini |
| Dataset | LLM Zoomcamp Lessons |
| Source | GitHub |
| Retrieval | Keyword Search |
| Semantic Retrieval | Dense Embeddings |
| Hybrid Retrieval | Reciprocal Rank Fusion |
| Evaluation | Hit Rate, MRR |

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

or using **uv**

```bash
uv pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

Example:

```text
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

The notebook loads the API keys automatically using:

```python
from dotenv import load_dotenv
load_dotenv()
```

This allows the notebook to communicate with Groq and Google Gemini without exposing credentials inside the code.

---

# 🏗 Project Architecture

The complete workflow implemented in this homework is illustrated below.

```text
                GitHub Repository
                       │
                       │
                       ▼
          Markdown Documentation (.md)
                       │
                       ▼
            GithubRepositoryDataReader
                       │
                       ▼
                 Parsed Documents
                       │
                       ▼
               Document Chunking
                       │
                       ▼
               Knowledge Base
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
 Keyword Search             Vector Search
          │                         │
          └────────────┬────────────┘
                       ▼
          Reciprocal Rank Fusion
                       │
                       ▼
               Hybrid Search
                       │
                       ▼
               Evaluation Dataset
                       │
                       ▼
            Hit Rate & MRR Metrics
```

This architecture follows the same retrieval pipeline used by many production Retrieval-Augmented Generation systems.

---

# 📚 Knowledge Base

Unlike many machine learning projects that rely on CSV datasets, this homework builds its knowledge base directly from the official **LLM Zoomcamp** GitHub repository.

The notebook downloads every lesson written in Markdown using the `GithubRepositoryDataReader` utility.

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

This process automatically retrieves all lesson files from the repository, parses their contents, and converts them into Python dictionaries that can later be indexed and searched.

Each document contains metadata such as:

- filename
- lesson path
- title
- Markdown content

After this step, the project has a structured knowledge base ready for preprocessing.

---

# ✂️ Why Chunking?

Large Language Models and retrieval systems do not work efficiently with extremely long documents.

Instead, each lesson is divided into multiple overlapping segments called **chunks**.

Rather than indexing an entire lesson as a single document, the notebook indexes many smaller pieces of text.

This offers several advantages:

- Better retrieval precision.
- Lower embedding cost.
- Faster search.
- Improved semantic matching.
- Better context preservation.

The notebook performs chunking using:

```python
chunks = chunk_documents(
    documents,
    size=2000,
    step=1000
)
```

where:

- Chunk size = **2000 characters**
- Overlap = **1000 characters**

The overlap ensures that important information located near chunk boundaries is not lost during retrieval.


---

# 🤖 Question Generation using Large Language Models

A retrieval system cannot be evaluated without a dataset of questions and expected answers.

Instead of manually writing hundreds of questions, this homework automatically generates realistic questions using a Large Language Model (LLM).

The objective is to simulate how real students would interact with the course documentation.

The notebook defines the following prompt:

```python
data_gen_instructions = """
You emulate a student who is taking our LLM course.

You are given one lesson page from the course.

Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete.
- Don't copy the lesson wording.
- Ask naturally.
"""
```

Rather than asking generic questions, the model is instructed to:

- behave like a student
- read the lesson
- understand its contents
- generate realistic questions
- avoid copying the original text
- produce structured JSON

This produces a much more realistic evaluation dataset than manually created examples.

---

# 🚀 Why Use an LLM to Generate Questions?

Creating evaluation datasets manually is expensive.

For a repository containing dozens of lessons, writing hundreds of realistic questions would require many hours.

Using an LLM allows us to automatically create a synthetic benchmark.

Benefits include:

- scalable
- reproducible
- inexpensive
- realistic
- diverse wording
- easier experimentation

This approach has become common when evaluating Retrieval-Augmented Generation systems.

---

# ⚡ Groq + Llama 3.3 70B

The notebook uses the Groq API to generate questions.

Model used:

> **Llama-3.3-70B-Versatile**

The workflow is:

```
Lesson
      │
      ▼
Prompt
      │
      ▼
Groq API
      │
      ▼
JSON Questions
      │
      ▼
Ground Truth Dataset
```

The implementation is straightforward.

```python
client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role":"system","content":instructions},
        {"role":"user","content":user_prompt}
    ],
    response_format={
        "type":"json_object"
    },
    temperature=0
)
```

Setting **temperature = 0** guarantees deterministic outputs, making experiments reproducible.

---

# 📦 Structured Output with Pydantic

One challenge when working with LLMs is ensuring consistent output formatting.

Instead of parsing arbitrary text, the notebook uses **Pydantic** schemas.

```python
class Questions(BaseModel):
    questions: list[str]
```

This guarantees every response contains:

```json
{
  "questions":[
      "...",
      "...",
      "...",
      "...",
      "..."
  ]
}
```

Advantages:

- no manual parsing
- automatic validation
- fewer runtime errors
- cleaner downstream processing

---

# 🔢 Token Usage Analysis

The notebook also measures how many prompt tokens are consumed.

```python
total_input_tokens += response.usage.prompt_tokens
```

Average token consumption is then calculated.

Monitoring token usage is important because:

- API costs depend on tokens.
- Larger prompts increase latency.
- Prompt optimization reduces expenses.
- Efficient prompts improve scalability.

---

# 🔍 Selecting the Evaluation Lessons

Rather than generating questions from the entire repository, the homework focuses on three introductory lessons.

```python
target_lessons = {
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md"
}
```

These lessons introduce:

- Agentic RAG
- Environment setup
- Retrieval-Augmented Generation fundamentals

The notebook filters the repository accordingly before sending pages to the LLM.

---

# 📊 Building the Ground Truth Dataset

A retrieval system requires knowing which document correctly answers each question.

This mapping is called the **Ground Truth**.

Each record contains:

| Field | Description |
|--------|-------------|
| question | User question |
| filename | Expected lesson |
| lesson | Source document |

Example:

| Question | Expected Document |
|-----------|------------------|
| How can I join Zoomcamp? | 01-intro.md |

Ground Truth is essential because evaluation metrics compare retrieved documents against these expected answers.

---

# 🔎 Keyword Search

Keyword Search is the simplest retrieval strategy.

Instead of understanding semantic meaning, it searches for words appearing inside documents.

```
Question

↓

Tokenization

↓

Keyword Matching

↓

Ranking

↓

Top Documents
```

Advantages:

- extremely fast
- deterministic
- inexpensive
- interpretable

Limitations:

- cannot understand synonyms
- sensitive to wording
- struggles with semantic similarity

Nevertheless, keyword search remains a strong baseline for many retrieval systems.

---

# 🧠 Dense Vector Search

Unlike keyword search, vector search converts documents into embeddings.

An embedding is a numerical representation of meaning.

```
Question

↓

Embedding Model

↓

Vector

↓

Similarity Search

↓

Nearest Documents
```

Instead of matching words, vector search matches semantic meaning.

For example:

Question

> "How do I enroll?"

can retrieve

> "Joining the course"

even if the exact words never appear.

Advantages:

- semantic understanding
- robust to paraphrasing
- language flexibility

Limitations:

- computationally expensive
- requires embedding models
- embedding quality strongly affects retrieval performance

---

# ⚖️ Keyword Search vs Vector Search

| Feature | Keyword | Vector |
|----------|----------|---------|
| Exact matching | ✅ | ❌ |
| Semantic understanding | ❌ | ✅ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Cost | Very Low | Higher |
| Robust to paraphrases | ❌ | ✅ |
| Explainability | High | Medium |

Neither approach is perfect.

This motivates combining them.

---

# 🔀 Hybrid Search

Hybrid Search combines multiple retrieval systems into a single ranking.

Instead of choosing between keyword search and vector search, both are executed independently.

```
                 Query
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
Keyword Search           Vector Search
      │                         │
      ▼                         ▼
 Ranked Results          Ranked Results
      └────────────┬────────────┘
                   ▼
          Reciprocal Rank Fusion
                   ▼
            Final Ranking
```

The intuition is simple:

If both systems independently rank a document highly, that document is likely to be relevant.

Hybrid retrieval usually outperforms either individual retriever.

---

# 🔢 Reciprocal Rank Fusion (RRF)

The notebook combines rankings using **Reciprocal Rank Fusion**.

Implementation:

```python
score += 1 / (k + rank)
```

where:

- **rank** = document position
- **k** = smoothing parameter

The final score becomes

\[
RRF(d)=\sum_{i=1}^{n}\frac{1}{k+r_i(d)}
\]

Instead of relying on raw similarity scores, RRF only considers document rankings.

Benefits:

- simple
- robust
- model independent
- state-of-the-art baseline for hybrid retrieval

This explains why RRF is widely adopted in production search engines and Retrieval-Augmented Generation systems.


---

# 📈 Evaluation Metrics

Building a retrieval system is only half of the task.

The other half consists of measuring **how well it retrieves the correct documents**.

This homework evaluates retrieval quality using two standard Information Retrieval metrics:

- Hit Rate (Recall@k)
- Mean Reciprocal Rank (MRR)

These metrics allow us to compare different retrieval strategies objectively.

---

# 🎯 Hit Rate

Hit Rate answers a simple question:

> **Was the correct document retrieved among the top-k results?**

For each query:

- Retrieve the Top-k documents.
- Check whether the expected document appears.
- Count it as a hit if found.

Mathematically:

\[
HitRate=\frac{\text{Number of Hits}}{\text{Total Queries}}
\]

Example:

| Query | Correct document found? |
|---------|------------------------|
| Q1 | ✅ |
| Q2 | ✅ |
| Q3 | ❌ |
| Q4 | ✅ |
| Q5 | ✅ |

Hit Rate

\[
=\frac{4}{5}=0.80
\]

Higher values indicate better retrieval coverage.

---

# 🥇 Mean Reciprocal Rank (MRR)

While Hit Rate only measures whether the document appears, MRR also considers **its position**.

Finding the correct answer in position **1** is much better than finding it in position **10**.

For each query:

\[
RR=\frac{1}{Rank}
\]

The Mean Reciprocal Rank is simply the average Reciprocal Rank across all queries.

\[
MRR=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{rank_i}
\]

Example:

| Rank | Reciprocal Rank |
|------|----------------|
|1|1.00|
|2|0.50|
|3|0.33|
|4|0.25|
|5|0.20|

Higher MRR means relevant documents appear earlier in the ranking.

---

# 🧪 Evaluation Function

The notebook implements an evaluation function that iterates over every query contained in the Ground Truth dataset.

For every question:

1. Execute the retrieval function.
2. Retrieve the Top-5 documents.
3. Compare them with the expected document.
4. Compute the corresponding metric.

Simplified workflow:

```text
Ground Truth
      │
      ▼
Question
      │
      ▼
Search Function
      │
      ▼
Retrieved Documents
      │
      ▼
Compare with Expected Answer
      │
      ▼
Compute Metrics
```

This evaluation framework makes it possible to compare different retrieval systems under identical conditions.

---

# 📊 Experimental Results

The notebook compares multiple retrieval approaches.

## Keyword Search

Keyword search achieved the highest Hit Rate among the individual retrieval systems evaluated.

Approximate result:

| Metric | Value |
|---------|------:|
| Hit Rate | **0.76** |

Keyword matching works particularly well because the documentation contains many technical terms that users tend to repeat in their questions.

---

## Vector Search

Dense retrieval produced a lower Hit Rate.

Approximate result:

| Metric | Value |
|---------|------:|
| Hit Rate | **0.45** |

Although semantic search can retrieve conceptually related documents, its performance depends heavily on the embedding model and indexing strategy.

In this homework, dense retrieval underperformed keyword matching.

---

# 🔀 Hybrid Search Results

Hybrid Search combines both retrieval methods using Reciprocal Rank Fusion.

The intuition is simple:

- Keyword Search captures exact terminology.
- Vector Search captures semantic similarity.
- RRF merges the strengths of both.

This generally produces rankings that are more robust than either individual retriever.

---

# ⚙ Hyperparameter Optimization

The notebook evaluates several values of the RRF parameter **k**.

The tested values are:

| k |
|---|
|1|
|50|
|100|
|200|

The corresponding MRR values are approximately:

| k | MRR |
|---:|----:|
|1|0.771|
|50|0.693|
|100|0.621|
|200|0.548|

The best performance is achieved with:

# ⭐ k = 1

This indicates that giving more importance to highly ranked documents improves retrieval quality for this dataset.

---

# 📉 Discussion

The experiments reveal several interesting observations.

### Keyword Search

Pros

- Extremely fast
- Very stable
- Excellent baseline
- Easy to interpret

Cons

- Sensitive to wording
- Cannot recognize synonyms

---

### Dense Vector Search

Pros

- Understands semantic similarity
- Robust to paraphrases
- Language independent

Cons

- More computationally expensive
- Depends on embedding quality
- Harder to interpret

---

### Hybrid Search

Pros

- Combines lexical and semantic retrieval
- More robust
- Better ranking quality
- State-of-the-art baseline

Cons

- Slightly more computationally expensive
- Requires maintaining two retrieval systems

Despite the added complexity, Hybrid Search usually offers the best trade-off between precision and recall.

---

# 💡 Lessons Learned

This homework demonstrates that retrieval quality depends on much more than selecting an embedding model.

Important components include:

- Document preprocessing
- Chunk size
- Chunk overlap
- Query formulation
- Ranking strategy
- Fusion algorithm
- Evaluation methodology

Small implementation details can significantly affect retrieval performance.

---

# 🚀 Possible Improvements

Several improvements could further enhance the retrieval system.

## Better Embeddings

- OpenAI text-embedding-3-large
- BGE
- E5
- Instructor XL

---

## Better Reranking

Instead of relying only on RRF, Cross-Encoder rerankers could be added after retrieval.

Examples:

- BAAI BGE Reranker
- Cohere Rerank
- Jina AI Reranker

---

## Vector Databases

Current experiments use lightweight indexing.

Production systems could use:

- Qdrant
- Pinecone
- Weaviate
- Milvus
- Elasticsearch
- pgvector

---

## Metadata Filtering

Additional metadata could improve retrieval.

Examples:

- lesson number
- module
- chapter
- topic
- difficulty

---

## Query Expansion

Instead of searching with the original query only, an LLM could generate alternative formulations.

Example:

```
Original Question
        │
        ▼
LLM Query Expansion
        │
        ▼
Multiple Searches
        │
        ▼
Fusion
```

This often improves recall.

---

# 📚 References

- DataTalksClub — LLM Zoomcamp
- Retrieval-Augmented Generation (Lewis et al., 2020)
- Reciprocal Rank Fusion (Cormack et al., 2009)
- Groq API Documentation
- Google Gemini API Documentation
- Python
- Pydantic
- GitHub

---

# ✅ Conclusion

This homework provided a practical introduction to retrieval evaluation for Retrieval-Augmented Generation (RAG) systems.

Starting from the official LLM Zoomcamp documentation, the project built a complete retrieval pipeline that included:

- downloading course material from GitHub,
- preprocessing and chunking documents,
- generating realistic evaluation questions with a Large Language Model,
- constructing a Ground Truth dataset,
- implementing keyword, vector, and hybrid retrieval,
- evaluating retrieval quality using Hit Rate and Mean Reciprocal Rank (MRR),
- optimizing the Reciprocal Rank Fusion parameter.

The experiments demonstrate that no single retrieval strategy is universally optimal.

Keyword Search offers excellent lexical matching, Vector Search captures semantic meaning, and Hybrid Search combines the strengths of both approaches to produce more robust rankings.

Beyond the specific implementation, this homework highlights an important lesson:

> **The quality of a Retrieval-Augmented Generation system depends as much on retrieval as on the language model itself.**

Careful preprocessing, indexing, ranking, and evaluation are fundamental components of modern AI systems and should always be measured systematically before deploying a production RAG application.

---

## 👨‍💻 Author

Homework completed as part of the **DataTalksClub LLM Zoomcamp 2026**.

This project was developed for educational purposes to explore retrieval evaluation techniques used in modern Retrieval-Augmented Generation systems.

