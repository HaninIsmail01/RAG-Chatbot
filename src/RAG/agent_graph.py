from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#required imports

from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config.config import (
    OPENAI_API_KEY,
    CHAT_MODEL_NAME,
    CHAT_TEMPERATURE,
)

from prompts import system_prompt
from src.RAG.retriever import QdrantRerankedRetriever, build_retriever
from src.RAG.chat_history import SessionMemory
from config.logging import get_logger

logger = get_logger(__name__)


# State

class RAGState(TypedDict):
    """
    The state object that flows through every node in the graph.
    Each node in the graph reads from and writes to this state object. 
    The state is initialized with the user query and conversation history,
    then flows through retrieval, generation, and citations. Each node has access to the full state, 
    but by convention:
    
    - The retrieve node writes 'context' and 'docs'
    - The generate node writes 'answer'
    - The cite node writes 'response'  
    
    This separation of concerns keeps the graph modular and each node focused on a single responsibility, 
    while the shared state allows them to communicate effectively. 
    The final 'response' field is what gets sent back to the user, and it includes both the LLM's answer 
    and the formatted citations based on the retrieved documents.

    Fields:
        question    — the current user query
        history     — full conversation history (add_messages merges turns)
        context     — formatted string of retrieved chunks
        docs        — raw retrieved Document objects (used for citations)
        answer      — the LLM's raw answer before citations are appended
        response    — final response with citations, written to the user
    """
    question: str
    # Conversation history as a list of messages, annotated for automatic 
    # merging into turns by add_messages
    history: Annotated[list[BaseMessage], add_messages] 
    context: str
    docs: list[Document]
    answer: str
    response: str

# Helpers

def _format_context(docs: list[Document]) -> str:
    """
    Format retrieved docs into a numbered context block for the LLM.
    Each chunk is labelled with its source metadata so the model
    can reference specific chunks in its answer.
    The context is formatted as a single string with clear separators between chunks.
    
    Asumes each doc has 'page_number', 'section', and 'source' metadata fields
    
    Args:
    docs (list[Document]): A list of retrieved Document objects, each containing page_content and metadata.
    
    Returns:  
    str: A formatted string that concatenates all retrieved chunks, each prefixed with its source metadata
        for clear reference by the LLM during answer generation.
    """
    chunks = []
    for i, doc in enumerate(docs, 1):
        # Extract metadata with fallbacks
        page = doc.metadata.get("page_number", "unknown")
        section = doc.metadata.get("section", " unidentified")
        source = doc.metadata.get("source", "")

        # Format header with available metadata
        header = f"[Chunk {i} | Source: {source} | Page: {page}"
        if section and section != "unknown":
            header += f" | Section: {section}"
        header += "]"
        # Combine header with content, ensuring clear separation between chunks
        chunks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


def _format_citations(docs: list[Document]) -> str:
    """
    Build a deduplicated citation block from retrieved chunks.
    Appended to every response — mandatory per the assessment.
    
    Each source is cited only once, even if multiple chunks 
    come from the same page/section. The citation format is designed 
    to be human-readable and clearly reference the source of the information.
    
    Assumes each doc has 'page_number', 'section', and 'source' metadata fields.
    
    Args:
      docs (list[Document]): A list of retrieved Document objects, each containing page_content and metadata.
    
    Returns:
      str: A formatted string containing the deduplicated citation block.
      
    """
    seen = set() # to track unique source-page-section combinations
    citations = []

    for doc in docs:
        # Extract metadata with fallbacks
        page = doc.metadata.get("page_number", "unknown")
        section = doc.metadata.get("section", "unidentified")
        source = doc.metadata.get("source", "")

        key = f"{source}|{page}|{section}" #unique key for deduplication
        if key in seen:
            continue
        seen.add(key)

        # Format citation with available metadata
        citation = f"📄 **{source}** — Page {page}" 
        if section and section != "unknown":
            citation += f", Section: _{section}_"
        citations.append(citation)

    return "\n".join(citations)

# Graph nodes

def make_retrieve_node(retriever: QdrantRerankedRetriever):
    """
    Node 1 — Retrieve.

    Fetches the top-K chunks from Qdrant, reranks them,
    and writes the formatted context + raw docs into state.
    Keeping docs in state decouples retrieval from citation
    formatting — each node has a single responsibility.
    
        The retriever is built outside the graph and passed in as a dependency,
        making it independently testable and easily replaceable in the future.
        The formatted context is what gets injected into the LLM prompt, while
        the raw docs are used later to build the citation block.
    
    Args:
      retriever (QdrantRerankedRetriever): An instance of the retriever class 
      that handles fetching and reranking chunks from Qdrant.
      
    Returns:
         A function that takes the current RAGState, performs retrieval and reranking, 
         and returns a dictionary with the retrieved docs and formatted context 
         to be merged into the state.
          
    """
    async def retrieve(state: RAGState) -> dict: 
        # The input state contains the user question and conversation 
        # history, which can be used for retrieval.
        logger.info(f"[retrieve] Query: '{state['question']}'")
        # The retriever returns a list of Document objects, 
        # which are then formatted into a context string for the LLM.
        docs = await retriever.ainvoke(state["question"])
        context = _format_context(docs)
        logger.info(f"[retrieve] Got {len(docs)} chunks")
        return {"docs": docs, "context": context}

    return retrieve


def make_generate_node(llm: ChatOpenAI):
    """
    Node 2 — Generate.

    Injects conversation history and retrieved context into
    the prompt, then calls the LLM. Streaming is enabled on
    the LLM so Chainlit can stream tokens in the UI layer.

    The system prompt enforces strict grounding — the LLM is
    instructed to answer only from the provided context and
    to explicitly say so if the answer isn't there.
    
    Args:
      llm (ChatOpenAI): An instance of the ChatOpenAI class, 
      configured with the desired model and parameters.
      
    Returns:
        A function that takes the current RAGState, constructs a prompt 
        with the conversation history and retrieved context, calls the LLM 
        to generate an answer, and returns a dictionary with the generated 
        answer to be merged into the state.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    # The prompt and LLM are composed into a single chain for simplicity,
    # but they could be separate nodes if we wanted to inspect the prompt output or reuse it
    chain = prompt | llm 

    async def generate(state: RAGState) -> dict:
        logger.info("[generate] Calling LLM...")
        response = await chain.ainvoke({ # unpack state for prompt formatting     
            "context": state["context"], # context from retrieve
            "question": state["question"], # original user question
            "history": state["history"], # full conversation history for better context
        })
        logger.info("[generate] LLM response received")
        return {"answer": response.content}

    return generate


def cite(state: RAGState) -> dict:
    """
    Node 3 — Cite.

    Appends the citation block to the LLM's answer.
    Separated into its own node so it's independently
    testable and easy to modify without touching generation.
    
    Args:
      state (RAGState): The current state containing the LLM's answer 
      and the retrieved docs.
    
    Returns:
        dict: A dictionary containing the final response with citations 
        to be merged into the state.
    """
    logger.info("[cite] Appending citations")
    citations = _format_citations(state["docs"])
    response = f"{state['answer']}\n\n---\n**Sources:**\n{citations}"
    return {"response": response}

# Graph

class RAGChain:
    """
    LangGraph-based RAG pipeline.

    Graph structure:
        retrieve → generate → cite → END

        retrieve : vector search + rerank → writes context + docs to state
        generate : LLM call with history + context → writes answer to state
        cite     : appends citations to answer → writes final response

    The StateGraph makes each stage explicit and independently
    inspectable — a significant advantage over a flat LangChain
    chain for debugging, evaluation, and future extension.

    Usage:
        chain = RAGChain(retriever)
        response = await chain.ainvoke("What is X?", memory)
    
    Attrs:
        _graph (StateGraph): The compiled graph object that defines the RAG pipeline.
    """

    def __init__(self, retriever: QdrantRerankedRetriever):
        self.logger = get_logger(self.__class__.__name__)
        # Build the graph at initialization so it's ready to go when ainvoke is called
        self._graph = self._build_graph(retriever) 

    # Build the graph with nodes and edges
    def _build_graph(self, retriever: QdrantRerankedRetriever): 
        llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            temperature=CHAT_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            streaming=True, # enable streaming for chainlit UI
        )

        # the graph is typed with our RAGState for clarity and type safety
        graph = StateGraph(RAGState) 

        # Register the nodes 
        graph.add_node("retrieve", make_retrieve_node(retriever))
        graph.add_node("generate", make_generate_node(llm))
        graph.add_node("cite", cite)

        # Define the edges
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "cite")
        graph.add_edge("cite", END)

        return graph.compile() # compile the graph for execution

    async def ainvoke(self, query: str, memory: SessionMemory) -> str:
        """
        Run the full graph for one user turn and return the
        final response string with citations appended.
        
        The state is initialized with the user query and conversation history,
        then flows through retrieval, generation, and citation nodes. After the graph
        completes, the final response is extracted from the state and returned.
        The conversation history in memory is also updated with the new user query and 
        assistantresponse after the graph execution, ensuring that the context is 
        maintained for future turns.
        
        Args:
            query (str): The user's question or input that initiates the RAG process.
            memory (SessionMemory): The memory object containing the conversation history.
        
        Returns:
            str: The final response string with citations appended.
            
        """
        self.logger.info(f"Invoking RAG graph for: '{query}'")

        initial_state: RAGState = { 
            # initialize state with query and history; context, docs, 
            # answer, and response will be filled by the graph nodes
            "question": query,
            "history": memory.get_messages(),
            "context": "",
            "docs": [],
            "answer": "",
            "response": "",
        }

        final_state = await self._graph.ainvoke(initial_state)
        response = final_state["response"]

        # Update memory after graph completes
        memory.add_user_message(query)
        memory.add_ai_message(response)

        return response


def build_chain() -> RAGChain:
    """Entry point — builds retriever and RAG graph. Called at app startup."""
    retriever = build_retriever()
    return RAGChain(retriever=retriever)

