from pathlib import Path
import sys

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



# STATE
class RAGState(TypedDict, total=False):
    """
    RAGState defines the shape of the state object that flows through the LangGraph.
    
    total=False is CRITICAL:
    prevents LangGraph from dropping keys between async nodes.
    """
    question: str
    history: Annotated[List[BaseMessage], add_messages] # Conversation history for the LLM, annotated to be processed by add_messages
    context: str
    docs: List[Document]
    answer: str
    response: str


# HELPER FUNCTIONS
def _format_context(docs: list[Document]) -> str:
    """
    Formats a list of documents for use as context in the LLM prompt.
    Each document's metadata (source, page number, section) is included as a header above its content.

    Args:
        docs (list[Document]): List of retrieved documents, each with page_content and metadata
    Returns:
        str: Formatted string combining all documents, ready to be included in the LLM prompt as context
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
    Takes a of retrieved documents and prepares them for citation.
    

    Args:
        docs (list[Document]):  List of retrieved documents, each with page_content and metadata 

    Returns:
        str: Formatted string of unique citations derived from the documents' metadata, ready to be appended to the LLM response
    """
    seen = set()
    citations = []

    for doc in docs:
        page = doc.metadata.get("page_number", "unknown")
        section = doc.metadata.get("section", "unknown")
        source = doc.metadata.get("source", "")

        key = f"{source}|{page}|{section.strip('/')}"
        if key in seen:
            continue
        seen.add(key)

        citation = f"📄 **{source}** — Page {page}"
        if section:
            citation += f", Section: _{section.strip('/')}_"
        citations.append(citation)

    return "\n".join(citations)


# NODE 1 — RETRIEVE
def make_retrieve_node(retriever: QdrantRerankedRetriever):
    """
    Defines the retrieve node for the agent graph, which uses the provided 
    retriever to fetch relevant documents based on the query in the state.
      
    Args:
        retriever (QdrantRerankedRetriever): An instance of the retriever to 
        use for fetching documents
    Returns:
        retrieve(async function): An async function that takes the current RAGState, performs retrieval, 
        and returns an updated state with retrieved documents and formatted context.
    """
    async def retrieve(state: RAGState) -> dict:
        
        logger.info(f"[retrieve] Query: {state['question']}")
        # Retrieve relevant documents using the retriever's async method
        docs = await retriever.ainvoke(state["question"]) 
        context = _format_context(docs)

        logger.info(f"[retrieve] docs: {len(docs)}")
        logger.info(f"[retrieve] context preview: {context[:200]}")

        return {
            "docs": docs,
            "context": context,
        }

    return retrieve

# NODE 2 — GENERATE
def make_generate_node(llm: ChatOpenAI):
    """
    Defines the generate node for the agent graph, which takes 
    in the formatted retrieved context and conversation history from the state,
    constructs a prompt, and generates an answer using the chat LLM.

    Args:
        llm (ChatOpenAI): An instance of the ChatOpenAI LLM to use for 
        generating responses based on the prompt.

    Returns:
        generate(async function): An async function that takes the current RAGState, 
        constructs a prompt with the question, context, and history, and returns an 
        updated state with the generated answer.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        # Injects the retrieved context and user question into the user prompt, 
        # separately to mitigate system prompt injections
        ("human",
         "Use ONLY the context below to answer.\n\n"
         "CONTEXT:\n{context}\n\n"
         "QUESTION:\n{question}") 
    ])

    chain = prompt | llm # Create a prompt chain that feeds into the LLM

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


# NODE 3 — CITATION
def cite(state: RAGState) -> dict:
    """
    Final node that takes the generated answer and the retrieved documents from the state,
    formats the citations, and combines them into a final response string.

    Args:
        state (RAGState): The current state containing the generated answer and 
        the list of retrieved documents with their metadata.

    Returns:
        dict: A dictionary with the final response string that includes the answer and formatted citations,
        ready to be sent back to the user.
    """
    logger.info("[cite] adding citations")

    citations = _format_citations(state.get("docs", []))
    response = f"{state['answer']}\n\n---\n**Sources:**\n{citations}"

    return {"response": response}


# GRAPH WRAPPER
class RAGChain:
    def __init__(self, retriever: QdrantRerankedRetriever):
        self.logger = get_logger(self.__class__.__name__)
        self._retriever = retriever
        self._graph = self._build_graph(retriever)
        self._llm = ChatOpenAI(
            model=CHAT_MODEL_NAME,
            temperature=CHAT_TEMPERATURE,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )

    def _build_graph(self, retriever):
        """
        Constructs the LangGraph with the defined nodes and edges for the RAG agent.

        Args:
            retriever: QdrantRerankedRetriever instance to be used in the retrieve node of the graph

        Returns:
            Compiled LangGraph ready for invocation, with nodes for retrieval, generation, and citation, 
            connected in the appropriate order.
        """
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

 
    # FULL INVOCATION
    async def ainvoke(self, query: str, memory: SessionMemory) -> str:
        """
        Runs the full LangGraph pipeline and return the complete
        response string.
        
        Args:
            query (str): The user's question to be processed by the RAG agent.
            memory (SessionMemory): The session memory object that holds the conversation history.
        Returns:
            str: The final response generated by the RAG agent, including the answer and citations.
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
        
        # Run the graph with the initial state and await the final state after processing through all nodes
        final_state = await self._graph.ainvoke(initial_state) 
        response = final_state["response"]

        memory.add_user_message(query)
        memory.add_ai_message(response)

        return response

    # STREAMING INVOKE
    async def astream(self, query: str, memory: SessionMemory) -> AsyncIterator[str]:
        """
        Stream response tokens one by one to the UI.

        Because LangGraph does not natively expose token-level streaming
        from individual nodes, the retrieve and cite are run manually outside
        the graph and stream only the LLM generation step — which is
        where the latency actually lives.

        Flow:
            1. retrieve  — fetch + rerank docs via the same retriever
                           the graph uses, keeping behaviour consistent
            2. astream   — stream LLM tokens directly from the prompt chain
            3. cite      — append citation block character by character
                           so the full response streams smoothly
            4. memory    — update conversation history after streaming ends

        Yields:
            str — individual tokens or characters
        """
        self.logger.info(f"[astream] Query: {query}")

        # Step 1: retrieve (same logic as graph node) 
        docs = await self._retriever.ainvoke(query)
        context = _format_context(docs)
        citations = _format_citations(docs)
        self.logger.info(f"[astream] Retrieved {len(docs)} docs")

        # Step 2: build prompt chain 
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human",
             "Use ONLY the context below to answer.\n\n"
             "CONTEXT:\n{context}\n\n"
             "QUESTION:\n{question}")
        ])

        stream_chain = prompt | self._llm

        # Step 3: stream LLM tokens
        full_answer = ""
        async for chunk in stream_chain.astream({
            "context": context,
            "question": query,
            "history": memory.get_messages(),
        }):
            token = chunk.content
            if token:
                full_answer += token
                yield token

        # Step 4: stream citation block
        citation_block = f"\n\n---\n**Sources:**\n{citations}"
        for char in citation_block:
            yield char

        # Step 5: update memory with the complete response after streaming finishes
        full_response = full_answer + citation_block
        memory.add_user_message(query)
        memory.add_ai_message(full_response)

        self.logger.info("[astream] Streaming complete")


# ENTRY POINT
def build_chain() -> RAGChain:
    retriever = build_retriever()
    return RAGChain(retriever)