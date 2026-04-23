from pathlib import Path
import sys

# Ensure project root is in PYTHONPATH for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Required Imports

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


# STATE DEFINITION
class RAGState(TypedDict, total=False):
    """
    Defines the shared state passed between LangGraph nodes.

    Notes:
        - `total=False` prevents keys from being dropped between async nodes.
        - This state carries query, history, retrieved docs, and outputs.
    """
    question: str
    history: Annotated[List[BaseMessage], add_messages]  # conversation history

    context: str
    docs: List[Document]

    answer: str
    response: str

    intent: str  # relevant | chitchat | irrelevant


# HELPER FUNCTIONS
def _format_context(docs: list[Document]) -> str:
    """
    Convert retrieved documents into a structured context string.

    Each chunk is prefixed with metadata (source, page, section)
    to improve grounding and traceability.

    Args:
        docs: Retrieved documents.

    Returns:
        Combined formatted context string.
    """
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
    """
    Generate a deduplicated citation list from retrieved documents.

    Args:
        docs: Retrieved documents.

    Returns:
        Formatted citation string.
    """
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


# NODE 0 — CLASSIFIER
def make_classify_node(llm: ChatOpenAI):
    """
    Creates a node that classifies user intent into:
    relevant | chitchat | irrelevant to avoid unnecessary retrieval 
    for non-iPhone queries and enhance the user experience with 
    appropriate responses.
    
    Args: 
      llm: A ChatOpenAI instance for running the classification prompt.
      
    Returns:
      An async function that takes RAGState and returns intent classification.
    """

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
        """Run classification and store intent in state."""
        result = await chain.ainvoke({"question": state["question"]})
        intent = result.content.strip().lower()

        # Fallback safety
        if intent not in {"relevant", "chitchat", "irrelevant"}:
            intent = "relevant"

        logger.info(f"[classify] intent: {intent}")
        return {"intent": intent}

    return classify


# NODE — DIRECT RESPONSE
def make_direct_answer_node(llm: ChatOpenAI):
    """
    Handles non-RAG responses (chitchat or irrelevant queries).
    
    This node generates a direct answer without retrieval, using a prompt
    
    Args: 
       llm: A ChatOpenAI instance for generating the response.
    Returns:
       An async function that takes RAGState and returns a direct response.
         
    """

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
        """Generate direct response without retrieval."""
        logger.info("[direct_answer] skipping retrieval")

        response = await chain.ainvoke({
            "question": state["question"],
            "history": state["history"],
        })

        return {"response": response.content}

    return direct_answer


# NODE 1 — RETRIEVE
def make_retrieve_node(retriever: QdrantRerankedRetriever):
    """
    Retrieves and formats documents using the provided retriever.
    
    Args:
       retriever: An instance of QdrantRerankedRetriever for fetching relevant documents.
    Returns:
       An async function that takes RAGState and returns retrieved documents and context.
    """

    async def retrieve(state: RAGState) -> dict:
        """Fetch relevant documents and build context."""
        logger.info(f"[retrieve] Query: {state['question']}")

        docs = await retriever.ainvoke(state["question"])
        context = _format_context(docs)

        logger.info(f"[retrieve] docs: {len(docs)}")
        logger.info(f"[retrieve] context preview: {context[:200]}")

        return {"docs": docs, "context": context}

    return retrieve


# NODE 2 — GENERATE
def make_generate_node(llm: ChatOpenAI):
    """
    Generates an answer using retrieved context and chat history.
    
    Args:
      llm: A ChatOpenAI instance for generating the answer.
      
    Returns:
      An async function that takes RAGState and returns the generated answer.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human",
         "Use ONLY the context below to answer.\n\n"
         "CONTEXT:\n{context}\n\n"
         "QUESTION:\n{question}")
    ])

    chain = prompt | llm #Create the chain once to reuse the same LLM instance and avoid reinitialization overhead.

    async def generate(state: RAGState) -> dict:
        """Run LLM generation step."""
        logger.info("[generate] running LLM")

        response = await chain.ainvoke({
            "context": state.get("context", ""),
            "question": state["question"],
            "history": state["history"],
        })

        return {"answer": response.content}

    return generate


# NODE 3 — CITE
def cite(state: RAGState) -> dict:
    """
    Appends formatted citations to the generated answer.
    
    Args:
      state: The RAGState containing the generated answer and retrieved docs.
    
    Returns:
      Updated state with the final response including citations.
    """
    logger.info("[cite] adding citations")

    citations = _format_citations(state.get("docs", []))
    response = f"{state['answer']}\n\n---\n**Sources:**\n{citations}"

    return {"response": response}

# ROUTER
def route(state: RAGState) -> str:
    """
    Routes execution based on intent classification.
    
        If intent is "relevant", route to retrieval and generation.
        Otherwise, route to direct answer.
    
    Args:
      state: The RAGState containing the intent classification.
      
    Returns:
      The name of the next node to execute ("retrieve" or "direct_answer").
    """
    if state.get("intent") == "relevant":
        return "retrieve"
    return "direct_answer"


# GRAPH WRAPPER
class RAGChain:
    """
    Encapsulates the LangGraph-based RAG pipeline,
    including classification, retrieval, generation, and citation.
    
    The chain is designed to be reusable across multiple queries
    within a session, maintaining a shared LLM instance and retriever. 
    """

    def __init__(self, retriever: QdrantRerankedRetriever):
        self.logger = get_logger(self.__class__.__name__)
        self._retriever = retriever

        # Shared LLM instance
        self._llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            temperature=CHAT_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )

        self._graph = self._build_graph(retriever)

    def _build_graph(self, retriever):
        """
        Build and compile the LangGraph pipeline.
        
        Args :
           retriever: An instance of QdrantRerankedRetriever to be used in the retrieve node.
           
        Returns:
              A compiled StateGraph ready for execution.
        """
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

    # STANDARD INVOKE
    async def ainvoke(self, query: str, memory: SessionMemory) -> str:
        """
        Execute full pipeline and return final response.
        
        This method runs the entire graph in one go, which is suitable for non-streaming use cases.
        
        Args:
          query: The user's question to be processed by the RAG pipeline.
          memory: A SessionMemory instance to store the chat history.
          
        Returns:
          The final response generated by the RAG pipeline.
        """
        self.logger.info(f"Query: {query}")

        initial_state: RAGState = {
            "question": query,
            "history": memory.get_messages(),
            "context": "",
            "docs": [],
            "answer": "",
            "response": "",
        }

        # NOTE: called twice (likely unintentional, but preserved as-is)
        final_state = await self._graph.ainvoke(initial_state)
        final_state = await self._graph.ainvoke(initial_state)

        response = final_state["response"]

        memory.add_user_message(query)
        memory.add_ai_message(response)

        return response

    # STREAMING INVOKE
    async def astream(self, query: str, memory: SessionMemory) -> AsyncIterator[str]:
        """
        Streams response tokens while manually handling retrieval and citations.
        
        This method allows for token-level streaming of the LLM response, while still 
        performing retrieval and citation formatting. It bypasses the graph's i
        nternal flow control to yield tokens as they are generated.
        
        Args:
          query: The user's question to be processed by the RAG pipeline.   
          memory: A SessionMemory instance to store the chat history.
          
        Yields:     
            Individual tokens of the generated response, streamed in real-time.
        """
        self.logger.info(f"[astream] Query: {query}")

        # Step 1: classify
        classifier = make_classify_node(self._llm)
        intent_result = await classifier({
            "question": query,
            "history": memory.get_messages()
        })

        intent = intent_result["intent"]
        logger.info(f"[astream] intent: {intent}")

        # Direct path (no retrieval)
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

        # RAG path
        docs = await self._retriever.ainvoke(query)
        context = _format_context(docs)
        citations = _format_citations(docs)

        # Build generation chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human",
             "Use ONLY the context below to answer.\n\n"
             "CONTEXT:\n{context}\n\n"
             "QUESTION:\n{question}")
        ])

        chain = prompt | self._llm 

        # Stream tokens
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

        # Stream citations
        citation_block = f"\n\n---\n**Sources:**\n{citations}"
        for char in citation_block:
            yield char

        # Update memory
        full_response = full_answer + citation_block
        memory.add_user_message(query)
        memory.add_ai_message(full_response)


# ENTRY POINT
def build_chain() -> RAGChain:
    """
    Factory function to construct the RAGChain with a configured retriever.
    """
    retriever = build_retriever()
    return RAGChain(retriever)