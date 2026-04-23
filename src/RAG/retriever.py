from pathlib import Path
import sys

# Ensure project root is in PYTHONPATH for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Required imports
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore 
from llama_index.core.postprocessor import SentenceTransformerRerank

from qdrant_client import QdrantClient
from pydantic import Field

from config.config import (
    QDRANT_HOST,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    LLAMA_INDEX_EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
)
from config.logging import get_logger

logger = get_logger(__name__)

# RETRIEVER
class QdrantRerankedRetriever(BaseRetriever):
    """
    Custom retriever that bridges LlamaIndex and LangChain.

    Pipeline:
        1. Retrieve top-k nodes using LlamaIndex vector search (Qdrant backend)
        2. Rerank retrieved nodes using a SentenceTransformer reranker
        3. Convert results into LangChain Document objects

    This ensures compatibility with:
        - LlamaIndex ingestion pipeline
        - LangChain-based RAG pipeline
    """

    # LlamaIndex vector index (excluded from serialization)
    index: VectorStoreIndex = Field(exclude=True)

    # Reranker model (excluded from serialization)
    reranker: SentenceTransformerRerank = Field(exclude=True)

    # Initial retrieval size
    top_k: int = RETRIEVAL_TOP_K

    # Final number of documents after reranking
    top_n: int = RERANKER_TOP_N 

    class Config:
        """
        Pydantic configuration:
            - Allows non-serializable objects (index, reranker)
            - Prevents serialization issues in LangChain pipelines
        """
        arbitrary_types_allowed = True 

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        Retrieve and rerank documents for a given query.

        Args:
            query: User query string.
            run_manager: LangChain callback manager (unused but required).

        Returns:
            List of LangChain Document objects after reranking.
        """

        logger.debug(f"[retrieve] Query: {query}")

        # Step 1: Retrieve top-k nodes using LlamaIndex
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        raw_nodes = retriever.retrieve(query)

        logger.info(f"[retrieve] Retrieved {len(raw_nodes)} nodes")

        # Debug logging (content + metadata preview)
        for i, node in enumerate(raw_nodes):
            logger.info(f"[DEBUG] node {i} text: {repr(node.get_content()[:200])}")
            logger.info(f"[DEBUG] node {i} metadata: {node.metadata}")

        # Step 2: Rerank retrieved nodes
        reranked = self.reranker.postprocess_nodes(
            raw_nodes,
            query_str=query,
        )

        logger.info(f"[retrieve] Reranked to {len(reranked)} nodes")

        # Step 3: Convert LlamaIndex nodes → LangChain Documents
        return [
            Document(
                page_content=node.get_content(),
                metadata=node.metadata,
            )
            for node in reranked
        ]


# BUILDER
def build_retriever() -> QdrantRerankedRetriever:
    """
    Construct a Qdrant-backed retriever aligned with ingestion settings.

    Steps:
        1. Initialize Qdrant client
        2. Configure embedding model (must match ingestion)
        3. Attach Qdrant vector store (LlamaIndex version)
        4. Build VectorStoreIndex
        5. Initialize reranker
        6. Return wrapped retriever

    Returns:
        QdrantRerankedRetriever instance ready for querying.
    """

    logger.info("Building retriever...")

    # 1. Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_HOST.strip().rstrip("/"),
        api_key=QDRANT_API_KEY.strip(),
    )

    # 2. Set embedding model (critical: must match ingestion phase)
    Settings.embed_model = FastEmbedEmbedding(
        model_name=LLAMA_INDEX_EMBED_MODEL_NAME,
        cache_dir=".fastembed_cache",
    )

    # 3. Create LlamaIndex Qdrant vector store (NOT LangChain version)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    # 4. Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # 5. Build index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    # 6. Initialize reranker
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME,
        top_n=RERANKER_TOP_N,
    )

    logger.info("Retriever ready")

    return QdrantRerankedRetriever(
        index=index,
        reranker=reranker,
    )


# TEST ENTRY POINT
if __name__ == "__main__":
    """
    Simple manual test:
        - Builds retriever
        - Runs a sample query
        - Prints first 200 characters of each result
    """
    retriever = build_retriever()
    docs = retriever.invoke("What are those alerts on my screen?")

    for d in docs:
        print(d.page_content[:200])