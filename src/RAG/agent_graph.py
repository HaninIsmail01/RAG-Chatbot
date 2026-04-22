from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# =========================
# Imports
# =========================
from typing import TypedDict, Annotated, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config.config import (
    OPENAI_API_KEY,
    CHAT_MODEL_NAME,
    CHAT_TEMPERATURE,
    SYSTEM_PROMPT,
)

from src.RAG.retriever import QdrantRerankedRetriever, build_retriever
from src.RAG.chat_history import SessionMemory
from config.logging import get_logger

logger = get_logger(__name__)


# =========================
# STATE (FIXED)
# =========================
class RAGState(TypedDict, total=False):
    """
    total=False is CRITICAL:
    prevents LangGraph from dropping keys between async nodes.
    """

    question: str
    history: Annotated[List[BaseMessage], add_messages]

    context: str
    docs: List[Document]

    answer: str
    response: str


# =========================
# HELPERS
# =========================
def _format_context(docs: list[Document]) -> str:
    chunks = []

    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page_number", "unknown")
        section = doc.metadata.get("section", "unknown")
        source = doc.metadata.get("source", "")

        header = f"[Chunk {i} | Source: {source.strip('/')} | Page: {page}"
        if section:
            header += f" | Section: {section.strip('/')}"
        header += "]"

        chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


def _format_citations(docs: list[Document]) -> str:
    seen = set()
    citations = []

    for doc in docs:
        page = doc.metadata.get("page_number", "unknown")
        section = doc.metadata.get("section", "unknown")
        source = doc.metadata.get("source", "")

        key = f"{source}|{page}|{section}"
        if key in seen:
            continue
        seen.add(key)

        citation = f"📄 **{source}** — Page {page}"
        if section:
            citation += f", Section: _{section}_"
        citations.append(citation)

    return "\n".join(citations)


# =========================
# NODE 1 — RETRIEVE
# =========================
def make_retrieve_node(retriever: QdrantRerankedRetriever):
    async def retrieve(state: RAGState) -> dict:

        logger.info(f"[retrieve] Query: {state['question']}")

        docs = await retriever.ainvoke(state["question"])

        context = _format_context(docs)

        logger.info(f"[retrieve] docs: {len(docs)}")
        logger.info(f"[retrieve] context preview: {context[:200]}")

        # 🔴 IMPORTANT: explicitly return all fields
        return {
            "docs": docs,
            "context": context,
        }

    return retrieve


# =========================
# NODE 2 — GENERATE
# =========================
def make_generate_node(llm: ChatOpenAI):

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),

        MessagesPlaceholder(variable_name="history"),

        ("human",
         "Use ONLY the context below to answer.\n\n"
         "CONTEXT:\n{context}\n\n"
         "QUESTION:\n{question}")
    ])

    chain = prompt | llm

    async def generate(state: RAGState) -> dict:

        logger.info("[generate] running LLM")

        logger.info(f"[generate] context exists: {bool(state.get('context'))}")

        logger.info(f"[generate] context preview:\n{state.get('context', '')[:300]}")

        response = await chain.ainvoke({
            "context": state.get("context", ""),
            "question": state["question"],
            "history": state["history"],
        })

        return {"answer": response.content}

    return generate

# =========================
# NODE 3 — CITATION
# =========================
def cite(state: RAGState) -> dict:

    logger.info("[cite] adding citations")

    citations = _format_citations(state.get("docs", []))

    response = f"{state['answer']}\n\n---\n**Sources:**\n{citations}"

    return {
        "response": response
    }


# =========================
# GRAPH WRAPPER
# =========================
class RAGChain:

    def __init__(self, retriever: QdrantRerankedRetriever):
        self.logger = get_logger(self.__class__.__name__)
        self._graph = self._build_graph(retriever)

    def _build_graph(self, retriever):

        llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            temperature=CHAT_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )

        graph = StateGraph(RAGState)

        graph.add_node("retrieve", make_retrieve_node(retriever))
        graph.add_node("generate", make_generate_node(llm))
        graph.add_node("cite", cite)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "cite")
        graph.add_edge("cite", END)

        return graph.compile()

    async def ainvoke(self, query: str, memory: SessionMemory):

        self.logger.info(f"Query: {query}")

        initial_state: RAGState = {
            "question": query,
            "history": memory.get_messages(),
            "context": "",
            "docs": [],
            "answer": "",
            "response": "",
        }

        final_state = await self._graph.ainvoke(initial_state)

        response = final_state["response"]

        memory.add_user_message(query)
        memory.add_ai_message(response)

        return response


# =========================
# ENTRY POINT
# =========================
def build_chain() -> RAGChain:
    retriever = build_retriever()
    return RAGChain(retriever)