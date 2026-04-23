from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# =========================
# Imports
# =========================
from typing import TypedDict, Annotated, List, AsyncIterator
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
# STATE
# =========================
class RAGState(TypedDict, total=False):
    question: str
    history: Annotated[List[BaseMessage], add_messages]

    context: str
    docs: List[Document]

    answer: str
    response: str

    intent: str  # relevant | chitchat | irrelevant


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
# NODE 0 — CLASSIFIER
# =========================
def make_classify_node(llm: ChatOpenAI):

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a classifier for an iPhone assistant.\n"
         "Classify the query into ONE word:\n"
         "- relevant (iPhone related)\n"
         "- chitchat (greetings/small talk)\n"
         "- irrelevant (not about iPhone)\n\n"
         "Return ONLY one word."),
        ("human", "{question}")
    ])

    chain = prompt | llm

    async def classify(state: RAGState) -> dict:
        result = await chain.ainvoke({"question": state["question"]})
        intent = result.content.strip().lower()

        if intent not in {"relevant", "chitchat", "irrelevant"}:
            intent = "relevant"

        logger.info(f"[classify] intent: {intent}")

        return {"intent": intent}

    return classify


# =========================
# NODE — DIRECT RESPONSE
# =========================
def make_direct_answer_node(llm: ChatOpenAI):

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly iPhone assistant.\n"
         "- If greeting → greet back briefly\n"
         "- If chitchat → respond briefly and steer to iPhone help\n"
         "- If irrelevant → politely refuse and say you only help with iPhones"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | llm

    async def direct_answer(state: RAGState) -> dict:
        logger.info("[direct_answer] skipping retrieval")

        response = await chain.ainvoke({
            "question": state["question"],
            "history": state["history"],
        })

        return {"response": response.content}

    return direct_answer


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

        return {"docs": docs, "context": context}

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

        response = await chain.ainvoke({
            "context": state.get("context", ""),
            "question": state["question"],
            "history": state["history"],
        })

        return {"answer": response.content}

    return generate


# =========================
# NODE 3 — CITE
# =========================
def cite(state: RAGState) -> dict:
    logger.info("[cite] adding citations")

    citations = _format_citations(state.get("docs", []))
    response = f"{state['answer']}\n\n---\n**Sources:**\n{citations}"

    return {"response": response}


# =========================
# ROUTER
# =========================
def route(state: RAGState) -> str:
    if state.get("intent") == "relevant":
        return "retrieve"
    return "direct_answer"


# =========================
# GRAPH WRAPPER
# =========================
class RAGChain:

    def __init__(self, retriever: QdrantRerankedRetriever):
        self.logger = get_logger(self.__class__.__name__)
        self._retriever = retriever

        self._llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            temperature=CHAT_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )

        self._graph = self._build_graph(retriever)

    def _build_graph(self, retriever):

        graph = StateGraph(RAGState)

        graph.add_node("classify", make_classify_node(self._llm))
        graph.add_node("retrieve", make_retrieve_node(retriever))
        graph.add_node("generate", make_generate_node(self._llm))
        graph.add_node("cite", cite)
        graph.add_node("direct_answer", make_direct_answer_node(self._llm))

        graph.set_entry_point("classify")

        graph.add_conditional_edges(
            "classify",
            route,
            {
                "retrieve": "retrieve",
                "direct_answer": "direct_answer",
            },
        )

        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "cite")
        graph.add_edge("cite", END)

        graph.add_edge("direct_answer", END)

        return graph.compile()

    # =========================
    # STANDARD INVOKE
    # =========================
    async def ainvoke(self, query: str, memory: SessionMemory) -> str:

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
    # STREAMING
    # =========================
    async def astream(self, query: str, memory: SessionMemory) -> AsyncIterator[str]:

        # classify first
        classifier = make_classify_node(self._llm)
        intent_result = await classifier({
            "question": query,
            "history": memory.get_messages()
        })

        intent = intent_result["intent"]
        logger.info(f"[astream] intent: {intent}")

        # ── DIRECT PATH ──
        if intent != "relevant":
            direct_node = make_direct_answer_node(self._llm)

            result = await direct_node({
                "question": query,
                "history": memory.get_messages()
            })

            text = result["response"]

            for char in text:
                yield char

            memory.add_user_message(query)
            memory.add_ai_message(text)
            return

        # ── RAG PATH ──
        docs = await self._retriever.ainvoke(query)
        context = _format_context(docs)
        citations = _format_citations(docs)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human",
             "Use ONLY the context below to answer.\n\n"
             "CONTEXT:\n{context}\n\n"
             "QUESTION:\n{question}")
        ])

        chain = prompt | self._llm

        full_answer = ""

        async for chunk in chain.astream({
            "context": context,
            "question": query,
            "history": memory.get_messages(),
        }):
            token = chunk.content
            if token:
                full_answer += token
                yield token

        citation_block = f"\n\n---\n**Sources:**\n{citations}"
        for char in citation_block:
            yield char

        full_response = full_answer + citation_block

        memory.add_user_message(query)
        memory.add_ai_message(full_response)


# =========================
# ENTRY POINT
# =========================
def build_chain() -> RAGChain:
    retriever = build_retriever()
    return RAGChain(retriever)