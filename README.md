# RAG Based Chatbot

A PDF-grounded RAG chatbot built with LlamaParse and Llama index for ingestion and retrival, Qdrant DB and FastEmbed for vector storage, LangGraph for Orchestration, and Chainlit for the UI. The chatbot is designed to answers questions strictly from a provided PDF document with mandatory source citations.

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| UI | Chainlit |
| Vector Store | Qdrant Cloud |
| Embedding | FastEmbed — `BAAI/bge-base-en-v1.5` (768 dims) |
| Reranker | `BAAI/bge-reranker-base` via sentence-transformers |
| PDF Parsing | LlamaParse v2 (`llama-cloud>=2.1`) |
| Chat Model | `gpt-4o-mini` |

---

## Setup & Running

### Prerequisites
- Docker
- API keys for: OpenAI, Qdrant Cloud, LlamaCloud

### Steps

1. Clone the repository:
```bash
   git clone https://github.com/HaninIsmail01/RAG-Chatbot.git
   cd RAG-Chatbot
```

2. Copy the environment template and fill in your API keys:
```bash
   cp .env.example .env
   # Edit .env with your actual keys
```

3. Build the Docker image:
```bash
   docker build -t chatbot:1.0 .
```

4. Run the container:
```bash
   docker run -p 8000:8000 --env-file .env chatbot:1.0
```

5. Open your browser at: http://localhost:8000

The vector index is pre-populated in Qdrant Cloud — no ingestion
step is required. The app is ready to answer questions immediately.

---

## Port

The application runs on port **8000**. Use `-p 8000:8000` for correct
port mapping as shown above.

---

## Architecture

User query
↓
Chainlit UI (src/interface/app.py)
↓
LangGraph StateGraph (src/RAG/agent_graph.py)
├── Node 1: retrieve  — Qdrant vector search + BGE reranker
├── Node 2: generate  — gpt-4o-mini with grounded system prompt
└── Node 3: cite      — append source page/section citations
↓
Grounded response with mandatory citations

---

## Chunking Strategy

The PDF is processed in two stages:

1. **MarkdownNodeParser** — LlamaParse outputs structured markdown,
   so splitting on header boundaries (##, ###) maps each chunk to
   a real document section. Tables and lists stay intact. Each node
   inherits the section header as metadata for rich citations.

2. **TokenTextSplitter** — handles oversized sections using the BGE
   tokenizer directly, ensuring chunk sizes are accurate for the
   embedding model (512 tokens, 64 overlap).

---

## Ingestion (for reference — not required to run)

The ingestion script is provided for viewing purposes:

```bash
poetry run python -m scripts.ingestion.ingest
```

The Qdrant index is already populated. Do not re-run unless you
want to repopulate the collection.

---

## Environment Variables

See `.env.example` for the full list with descriptions.