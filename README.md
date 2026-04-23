# RAG Based Chatbot

A PDF-grounded RAG chatbot built with LlamaParse and LlamaIndex for ingestion, Qdrant and FastEmbed for vector storage, LangGraph for orchestration, and Chainlit for the UI. The chatbot answers questions strictly from a provided PDF document with mandatory source citations on every response.

---

## Features

- **Document-grounded answers** — the chatbot answers exclusively from the content of the provided PDF. It will never fabricate information or draw on general knowledge outside the document.
- **Mandatory source citations** — every response includes the source page number and section from the PDF, allowing answers to be traced directly back to the original document.
    
     ![Chatbot Grounded Answer Sample](assets\grounded_answer.png)

- **Query classification** — user inputs are classified before retrieval into one of three categories: topic-relevant, chitchat, or irrelevant. This avoids unnecessary vector search, handles off-topic inputs gracefully, and improves the overall user experience.
      
      ![Responses to different query types](assets\sample-query-classes.png)

- **Retrieval with reranking** — chunks are first retrieved by vector similarity from Qdrant, then reranked using a cross-encoder (`BAAI/bge-reranker-base`) to surface the most relevant content before passing it to the LLM.
- **Multi-turn conversation** — the chatbot maintains conversation history within a session, allowing follow-up questions and context-aware responses across multiple turns.

      ![Multiturn support](assets\conv-history.png)

- **Streaming responses** — answers are streamed token by token in the UI for a responsive, real-time feel.
- **Structured ingestion pipeline** — the PDF is parsed using LlamaParse v2's agentic OCR, chunked with a two-stage MarkdownNodeParser + TokenTextSplitter strategy, embedded locally with FastEmbed, and stored in Qdrant Cloud with rich metadata (page number, section) for accurate citations.
- **Containerized deployment** — the full application runs in a single Docker container with model weights pre-cached at build time for instant startup.

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

5. Open your browser at:

```
http://localhost:8000
```

The vector index is pre-populated in Qdrant Cloud — no ingestion step is required. The app is ready to answer questions immediately after the container starts.

---

## Port

The application runs on port **8000**. Use `-p 8000:8000` for correct port mapping as shown above.

---

## Architecture

```
User query
    ↓
Chainlit UI (src/interface/app.py)
    ↓
LangGraph StateGraph (src/RAG/agent_graph.py)
    ├── Node 0: classify  — classifies input as topic-relevant, chitchat,
    │                        or irrelevant to avoid unnecessary retrieval
    │                        and handle varied user inputs appropriately
    ├── Node 1: retrieve  — Qdrant vector search + BGE reranker
    ├── Node 2: generate  — gpt-4o-mini with grounded system prompt
    └── Node 3: cite      — appends source page/section citations
    ↓
Grounded response with mandatory citations
```

---

## Ingestion (for reference — not required to run)

The ingestion script is provided for review purposes only. The Qdrant index is already populated — do not re-run unless you want to repopulate the collection from scratch.

Ingest upload confirmation: 
    ![Ingestion upload](assets\ingestion-log.png)

```bash
poetry run python -m scripts/ingestion/ingestion.py
```

---

## Chunking Strategy

The PDF is processed in two stages:

**Stage 1 — MarkdownNodeParser:** LlamaParse outputs structured markdown, so splitting on header boundaries (`##`, `###`) maps each chunk to a real document section. Tables and lists stay intact within their node, and each node inherits the section header as metadata — enabling rich citations like *Page 4, Section 2.1, "Section title"* rather than just a page number. 

**Stage 2 — TokenTextSplitter:** `MarkdownNodeParser` can produce oversized nodes for long sections. `TokenTextSplitter` acts as a safety net, splitting using the BGE tokenizer directly to ensure token counts are accurate for the embedding model (512 tokens, 64 token overlap). Using the same tokenizer as the embedding model avoids the mismatch that would occur with a generic tokenizer like tiktoken.

---

##  RAG Retrieval Strategy

Retrieval is designed as a two-stage pipeline to balance recall and precision — vector similarity alone is fast but imprecise, while a
cross-encoder reranker is accurate but too slow to run over the entire index. Combining both gives the best of each.

**Stage 1 — Dense Vector Retrieval:**
User queries are embedded using the same `BAAI/bge-base-en-v1.5` model used during ingestion, ensuring query and document vectors live in the
same embedding space. The query vector is used to retrieve the top-10 most similar chunks from Qdrant using cosine similarity. This stage
prioritises recall — it casts a wide net to ensure the relevant chunks are in the candidate set before the reranker narrows it down.

**Stage 2 — Cross-Encoder Reranking:**
The top-10 retrieved chunks are passed to `BAAI/bge-reranker-base`, a cross-encoder model that scores each query-chunk pair jointly rather
than independently. Unlike bi-encoders (used in Stage 1), a cross-encoder sees the query and chunk together in a single forward
pass, producing a more accurate relevance score. The top-5 chunks by reranker score are passed to the LLM as context. 

**Query Classification (pre-retrieval):**
Before retrieval is triggered, the user's input is classified by the
LangGraph `classify` node into one of three categories:

- `relevant` — proceeds through the full retrieval → generate → cite pipeline
- `chitchat` — handled with a friendly response, no retrieval triggered
- `irrelevant` — politely declined without consuming retrieval resources

This avoids unnecessary vector searches for non-document queries and allows the chatbot to handle casual interaction gracefully without
breaking character.

**Why this combination:**

| Decision | Rationale |
|---|---|
| top-K = 10 for retrieval | Wide enough to ensure relevant chunks aren't missed due to embedding approximation |
| top-N = 5 after reranking | Keeps the LLM context window focused — more chunks increase noise and cost |
| Cosine similarity | Standard for normalised embeddings — measures directional alignment |
| Same model for query + doc embedding | Guarantees vectors occupy the same space — a mismatch would silently destroy retrieval quality |
| Cross-encoder over bi-encoder for reranking | Cross-encoders score query-chunk pairs jointly, giving significantly more accurate relevance scores at the cost of speed — acceptable since reranking only runs on 10 candidates, not the full index |

## Environment Variables

See `.env.example` for the full list of required variables with placeholder values and descriptions.