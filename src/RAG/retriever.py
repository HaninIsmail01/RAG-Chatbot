from pathlib import Path
import sys

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


# Retriever
class QdrantRerankedRetriever(BaseRetriever):
    """
    LlamaIndex-based retriever wrapped as a LangChain BaseRetriever to match the vector build used at ingestion time
    with the structure expected by LangChain. 
    Retrieval is done in two steps: 
        1. First retrieve top-k with LlamaIndex 
        2. Then rerank with SentenceTransformerRerank.
    """
    
     # the LlamaIndex VectorStoreIndex, NOT the LangChain version
    index: VectorStoreIndex = Field(exclude=True)
    # the SentenceTransformerRerank postprocessor
    reranker: SentenceTransformerRerank = Field(exclude=True)

     # Number of initial documents to retrieve with LlamaIndex before reranking
    top_k: int = RETRIEVAL_TOP_K
    # Number of documents to return after reranking (final number of documents fed to LLM)
    top_n: int = RERANKER_TOP_N 

    class Config:
        """
        Configuration for this pydantic object.
            - exclude the LlamaIndex index and reranker from the model dict 
            since they are not serializable and not needed for the retriever to function 
            (they are passed in at init and used directly in the retrieval method).
        """
        # allow arbitrary types since the index and reranker are complex objects.
        arbitrary_types_allowed = True 

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """
        

        Args:
            query (str): _description_
            run_manager (CallbackManagerForRetrieverRun): _description_

        Returns:
            list[Document]: _description_
        """

        logger.debug(f"[retrieve] Query: {query}")

        # Step 1: Retrieve using LlamaIndex (CORRECT)
        retriever = self.index.as_retriever(similarity_top_k=self.top_k)
        raw_nodes = retriever.retrieve(query)

        logger.info(f"[retrieve] Retrieved {len(raw_nodes)} nodes")

        # Debug content
        for i, node in enumerate(raw_nodes):
            logger.info(f"[DEBUG] node {i} text: {repr(node.get_content()[:200])}")
            logger.info(f"[DEBUG] node {i} metadata: {node.metadata}")

        # Step 2: Rerank
        reranked = self.reranker.postprocess_nodes(
            raw_nodes,
            query_str=query,
        )

        logger.info(f"[retrieve] Reranked to {len(reranked)} nodes")

        # Step 3: Convert to LangChain Documents
        return [
            Document(
                page_content=node.get_content(),
                metadata=node.metadata,
            )
            for node in reranked
        ]


# =========================
# Builder
# =========================
def build_retriever() -> QdrantRerankedRetriever:
    """
    Build retriever using LlamaIndex + Qdrant (compatible with your ingestion).
    """

    logger.info("Building retriever...")

    # 1. Qdrant client
    client = QdrantClient(
        url=QDRANT_HOST.strip().rstrip("/"),
        api_key=QDRANT_API_KEY.strip(),
    )

    # 2. Embedding model (must match ingestion)
    Settings.embed_model = FastEmbedEmbedding(
        model_name=LLAMA_INDEX_EMBED_MODEL_NAME,
        cache_dir=".fastembed_cache",
    )

    # 3. LlamaIndex Qdrant vector store (IMPORTANT: NOT LangChain version)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
    )

    # 4. Storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # 5. Build index
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )

    # 6. Reranker
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL_NAME,
        top_n=RERANKER_TOP_N,
    )

    logger.info("Retriever ready")

    return QdrantRerankedRetriever(
        index=index,
        reranker=reranker,
    )


# =========================
# Test
# =========================
if __name__ == "__main__":
    retriever = build_retriever()
    docs = retriever.invoke("What are those alerts on my screen?")
    for d in docs:
        print(d.page_content[:200])