import sys
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Resolves the project root regardless of OS or working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#All needed imports
from dataclasses import dataclass
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from qdrant_client import QdrantClient

from config.config import (
    QDRANT_PORT,
    QDRANT_HOST,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    LLAMA_INDEX_EMBED_MODEL_NAME,
    RERANKER_MODEL_NAME,
    RETRIEVAL_TOP_K,
    RERANKER_TOP_N,
)
from config.logging import get_logger


@dataclass
class RetrievalResult:
    """Represents a single retrieved and reranked chunk."""
    rank: int
    page_number: str
    section: str
    source: str
    score: float
    content_preview: str


class RetrievalTester:
    """
    Validates the ingestion pipeline by running test queries against
    the populated Qdrant collection and printing reranked results.

    Steps:
        1. Connect to Qdrant and load the vector index
        2. Retrieve top-K chunks via vector similarity
        3. Rerank with FlagEmbeddingReranker (cross-encoder) to top-N
        4. Print ranked results with metadata for inspection

    Usage:
        tester = RetrievalTester()
        tester.run(queries=["...", "...", "..."])
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._index = None
        self._reranker = None
        self._setup()
        self.logger.info("Retrieval tester initialised")

    def _setup(self) -> None:
        self.logger.info("Initialising retrieval tester...")
        embed_model = self._build_embed_model()

        # ── Set globally via Settings — required in llama-index-core >= 0.10
        Settings.embed_model = embed_model

        client = self._connect_qdrant()
        self._index = self._load_index(client)
        self._reranker = self._build_reranker()
        self.logger.info("Retrieval tester ready")

    def _build_embed_model(self) -> FastEmbedEmbedding:
        self.logger.info(f"Loading embed model: {LLAMA_INDEX_EMBED_MODEL_NAME}")
        embed_model = FastEmbedEmbedding(
            model_name=LLAMA_INDEX_EMBED_MODEL_NAME,
            cache_dir=".fastembed_cache",
        )
        return embed_model

    def _connect_qdrant(self) -> QdrantClient:
        self.logger.info("Connecting to Qdrant Cloud...")
        return QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

    def _load_index(self, client: QdrantClient) -> VectorStoreIndex:
        # No longer accepts embed_model as a parameter —
        # uses Settings.embed_model set above
        self.logger.info(
            f"Loading index from collection: '{QDRANT_COLLECTION_NAME}'"
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

    def _build_reranker(self) -> SentenceTransformerRerank:
        """
        SentenceTransformerRerank uses a cross-encoder model to re-score
        retrieved chunks based on direct query-chunk relevance — more
        accurate than vector similarity alone, which only measures
        approximate directional proximity in embedding space.
        
        
        """
        self.logger.info(f"Loading reranker: {RERANKER_MODEL_NAME}")
        return SentenceTransformerRerank(
            model=RERANKER_MODEL_NAME,
            top_n=RERANKER_TOP_N,
        )

    def run(self, queries: list[str]) -> None:
        """
        Run all test queries and print ranked results.
        
        Args:
            queries (list[str]): A list of test queries to run.
        Returns:
            None
        """
        self.logger.info(
            f"Running {len(queries)} test queries "
            f"(top_k={RETRIEVAL_TOP_K} → rerank to top_n={RERANKER_TOP_N})"
        )
        for i, query in enumerate(queries, 1):
            print(f"\n{'=' * 60}")
            print(f"Query {i}/{len(queries)}: {query}")
            print("=" * 60)
            results = self.retrieve(query)
            self._print_results(results)

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve and rerank chunks for a single query.
        Returns a list of RetrievalResult dataclasses.
        """
        retriever = self._index.as_retriever(similarity_top_k=RETRIEVAL_TOP_K)
        raw_nodes: list[NodeWithScore] = retriever.retrieve(query)

        reranked: list[NodeWithScore] = self._reranker.postprocess_nodes(
            raw_nodes,
            query_str=query,
        )
        self.logger.debug(
            f"Reranked down to {len(reranked)} chunks"
        )

        return [
            RetrievalResult(
                rank=rank,
                page_number=node.metadata.get("page_number", "unknown"),
                section=node.metadata.get("section", "unknown"),
                source=node.metadata.get("source", "unknown"),
                score=round(node.score, 4) if node.score else 0.0,
                content_preview=node.get_content()[:400].replace("\n", " "),
            )
            for rank, node in enumerate(reranked, 1)
        ]

    def _print_results(self, results: list[RetrievalResult]) -> None:
        if not results:
            print("⚠️  No results returned — check your collection.")
            return

        for r in results:
            print(
                f"\n  Rank    : #{r.rank}"
                f"\n  Score   : {r.score}"
                f"\n  Source  : {r.source}"
                f"\n  Page    : {r.page_number}"
                f"\n  Section : {r.section}"
                f"\n  Preview : {r.content_preview}..."
                f"\n  {'─' * 56}"
            )


TEST_QUERIES = [
    "What are those alerts on my screen?"
]

if __name__ == "__main__":
    tester = RetrievalTester()
    tester.run(queries=TEST_QUERIES)