from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#required imports
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from pydantic import Field

from config.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    LLAMA_INDEX_EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
    OPENAI_API_KEY,
)
from config.logging import get_logger

logger = get_logger(__name__)


class QdrantRerankedRetriever(BaseRetriever):
    """
    LangChain-compatible retriever that:
      1. Fetches top-K chunks from Qdrant via cosine similarity
      2. Reranks them with SentenceTransformerRerank (cross-encoder)
      3. Returns top-N most relevant chunks as LangChain Documents

    Using LangChain's BaseRetriever makes this retriever a drop-in
    component for any LangChain chain without any special wiring.
    
        Attributes:
            vector_store: QdrantVectorStore
            reranker: SentenceTransformerRerank
            top_k: int
            top_n: int  
        
        Returns:
            list[Document]: Reranked relevant documents for the query.
    """
    # Pydantic fields with exclude=True to prevent validation issues with complex objects
    # These are set manually in the build_retriever() function and not expected to be passed in via constructor 
    
    vector_store: QdrantVectorStore = Field(exclude=True) #exclude from pydantic validation 
    reranker: SentenceTransformerRerank = Field(exclude=True) #exclude from pydantic validation
    top_k: int = RETRIEVAL_TOP_K # number of initial top similar chunks to retrieve from Qdrant before reranking
    top_n: int = RERANKER_TOP_N  # number of top chunks to return after reranking

    class Config:
        # allow non-pydantic types like QdrantVectorStore and SentenceTransformerRerank
        arbitrary_types_allowed = True 

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        
        """
        Retrieves relevant documents for a given query by first performing a 
        vector similarity search and then reranking them.

        Args:
            query (str):  The  user input query string for which relevant documents are to be retrieved.
            run_manager (CallbackManagerForRetrieverRun): A callback manager for handling retriever 
            run events (not used in this implementation but required by the BaseRetriever interface).

        Returns:
            list[Document]: A list of top N reranked documents as LangChain Document objects representing the
            most relevant chunks after reranking.
        """

        # Step 1: vector similarity retrieval
        logger.debug(f"Retrieving top-{self.top_k} chunks for: '{query}'")
        raw_docs = self.vector_store.similarity_search(
            query, k=self.top_k
        )

        #Step 2: convert to LlamaIndex nodes for reranking using SentenceTransformerRerank
        nodes = [
            NodeWithScore(
                node=TextNode(
                    text=doc.page_content,
                    metadata=doc.metadata,
                ),
                score=1.0,
            )
            for doc in raw_docs
        ]

        # Step 3: rerank
        reranked = self.reranker.postprocess_nodes(
            nodes,
            #The SentenceTransformerRerank expects a QueryBundle, 
            # which can contain additional info such as metadata if needed. Here we just pass the query string.
            query_bundle=QueryBundle(query_str=query), 
        )
        logger.debug(f"Reranked to top-{len(reranked)} chunks")

        # Step 4: convert back the reranked nodes to LangChain Documents 
        return [
            Document(
                page_content=node.get_content(),
                metadata=node.metadata,
            )
            for node in reranked
        ]

def build_retriever() -> QdrantRerankedRetriever:
    """
    Initialise and return the retriever — called once at app startup.
    This function sets up the Qdrant client, the embedding model, and the reranker,
    and then combines them into a QdrantRerankedRetriever instance.

    Returns:
        QdrantRerankedRetriever: An instance of the retriever ready to be used for querying.
        
    Note:
        The retriever uses a two-stage retrieval process:
        1. Vector similarity search to retrieve top-K chunks from Qdrant
        2. Rerank with FlagEmbeddingReranker (cross-encoder) to top-N most relevant chunks
    """
    logger.info("Building retriever...")

    client = QdrantClient(
        url=QDRANT_URL.strip().rstrip("/"),
        api_key=QDRANT_API_KEY.strip(),
    )

    # LangChain-native Qdrant integration
    # Uses FastEmbed-compatible dimensions via OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
    )

    # LangChain-native reranker
    # Uses a cross-encoder model to re-score retrieved chunks
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME,
        top_n=RERANKER_TOP_N,
    )

    logger.info("Retriever ready")
    return QdrantRerankedRetriever(
        vector_store=vector_store,
        reranker=reranker,
    )
    
if __name__ == "__main__":
    retriever = build_retriever()