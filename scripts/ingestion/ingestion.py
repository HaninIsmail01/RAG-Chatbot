# Required imports for the ingestion

import sys
import os

# add the parent directory to the system path to allow imports from config and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

# dataclass is used to create a configuration class for the ingestion process, 
# which will hold all the necessary configuration parameters in a structured way.
from dataclasses import dataclass, field 

"""
Llama Parse was chosen to parse the document for its capaability to extract the text and 
read the images (important because the provided guide contains multiple guidance images) from the pdf
in a well structured manner with accuracy and with respect to its structure. 
"""
from llama_parse import LlamaParse

"""
Llama index was chosen to orchestrate the ingestion process and populate the vector store 
for its alignment with llama parse, its ease of use and mainly for its capability to split 
the document in markdown structure, which preserves the structure of the document with 
respect to its various length sections and figures, and importantly, its ability to extract the 
section names that will be used to cite the answers along with the page numbers.
"""
from llama_index.core import VectorStoreIndex, StorageContext

"""
Markdown splitter was chosen to perserve the document structure and
the sentence splitter is also needed to handle very long sections that 
the markdown splitter will keep as one whole chunk.
"""
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter 
from llama_index.core.schema import BaseNode

"""
Qdrant was chosen for its ease of use, its support for llama index, and its familiarity
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

"""
Fastembed was chosen for its compatibility with qdrant vector DB and 
its suitability for the task, being straightforward.
"""
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config.logging import get_logger # Custom logging configuration for the ingestion process


from config.config import ( # Importing all the necessary configuration parameters 
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    LLAMA_INDEX_EMBED_MODEL_NAME,
    FASTEMBED_CACHE_DIR,
    LLAMAPARSE_API_KEY,
    LLAMAPARSE_RESULT_TYPE,
    LLAMAPARSE_PARSE_LANGUAGE,
    EMBED_DIMENSIONS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
    DOCUMENTS_DIR,
)

logger = get_logger(__name__) # load the logger for the ingestion module



@dataclass
class IngestionConfig:
    """
    Configuration dataclass for the ingestion process.
    This class holds all the necessary configuration parameters for the ingestion process, 
    including Qdrant and Llama Index configurations, as well as data paths.
    """
    qdrant_host: str = QDRANT_HOST
    qdrant_port: int = QDRANT_PORT
    qdrant_api_key: str = QDRANT_API_KEY
    qdrant_collection_name: str = QDRANT_COLLECTION_NAME
    llamaparse_api_key: str = LLAMAPARSE_API_KEY
    llamaparse_result_type: str = LLAMAPARSE_RESULT_TYPE
    llamaparse_language: str = LLAMAPARSE_PARSE_LANGUAGE
    pdf_path: str = DATA_DIR
    documents_dir: str = DOCUMENTS_DIR
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    embed_model_name: str = LLAMA_INDEX_EMBED_MODEL_NAME,
    fastembed_cache_dir: str = FASTEMBED_CACHE_DIR
    emned_dimensions: int = EMBED_DIMENSIONS
    
    
@dataclass
class IngestionResult: 
    """
    Result dataclass for the ingestion process.
    Returned by IngestionPipeline.run() 
    summary of what was processed.
    """
    
    total_documents: int
    total_nodes: int
    collection_name: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    success: bool
    error: str | None = None

class IngestionPipeline:
    """
    End-to-end PDF ingestion pipeline.
 
    Stages:
        1. Parse  — LlamaParse converts PDF to structured markdown
        2. Chunk  — Two-stage splitting: MarkdownNodeParser → SentenceSplitter
        3. Embed  — FastEmbed generates local embeddings (no API cost)
        4. Upload — Nodes + metadata pushed to Qdrant Cloud
 
    Usage:
        config = IngestionConfig()
        pipeline = IngestionPipeline(config)
        result = pipeline.run()
    """
    # Class constructor
    def __init__(self, config: IngestionConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self._validate_config()
 
    # Full pipeline execution
    def run(self) -> IngestionResult:
        """
        Execute the full ingestion pipeline and return a result summary.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting ingestion pipeline")
        self.logger.info("=" * 60)
 
        try:
            documents = self._parse_pdf()
            nodes = self._build_nodes(documents)
            embed_model = self._build_embed_model()
            client = self._connect_qdrant()
            self._setup_collection(client)
            self._upload(nodes, embed_model, client)
 
            result = IngestionResult(
                total_documents=len(documents),
                total_nodes=len(nodes),
                collection_name=self.config.collection_name,
                embed_model=self.config.embed_model_name,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                success=True,
            )
            self._log_summary(result)
            return result
 
        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}", exc_info=True)
            return IngestionResult(
                total_documents=0,
                total_nodes=0,
                collection_name=self.config.collection_name,
                embed_model=self.config.embed_model_name,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                success=False,
                error=str(e),
            )
    
    # Phase 1: Parsing the PDF in markdown using LlamaParse
    def _parse_pdf(self) -> list:
        
        """
        Parse the PDF using LlamaParse with result_type='markdown' 
        to preserve the document structure.
        
        result_type='markdown' preserves document structure:
        headers become #/##, tables stay tabular, lists remain intact.
        
        This structural signal improves how MarkdownNodeParser identifies
        section boundaries in the next stage.
        
        Returns:
            list: A list of the markdown nodes from theparsed document.
        """
        self.logger.info(f"[Stage 1] Parsing PDF: {self.config.pdf_path}")
 
        parser = LlamaParse(
            api_key=self.config.llama_cloud_api_key,
            result_type=self.config.llama_parse_result_type,
            #automatically switches between parsing modes to only  
            #used advanced parsing with images, for cost and effciency
            auto_mode=True, 
            auto_mode_trigger_on_image_in_page=True,
            verbose=True,
            language=self.config.llama_parse_language,
            
        )
 
        documents = parser.load_data(self.config.pdf_path)
 
        self.logger.info(
            f"[Stage 1] LlamaParse returned {len(documents)} document pages"
        )
        return documents
    
    # Phase 2: Building chunks with a two-stage splitting strategy
    def _build_nodes(self, documents: list) -> list[BaseNode]:
        """
        Two-stage chunking strategy:
 
        Stage 1 — MarkdownNodeParser:
            Splits on markdown headers (##, ###) as natural section boundaries.
            Because LlamaParse outputs structured markdown, each header maps
            directly to a real section in the document. This enriches
            each node with a 'header_path' metadata key , will be used 
            as the 'section' field for citations.
 
        Stage 2 — SentenceSplitter:
            MarkdownNodeParser can produce oversized nodes for long sections.
            SentenceSplitter acts as a safety net — it respects sentence
            boundaries (no mid-sentence cuts) while ensuring no node exceeds
            the embedding model's context window.
 
            chunk_size=512: balances retrieval precision vs. context richness.
            chunk_overlap=64: prevents answers spanning chunk boundaries
            from being missed entirely.
 
        Args:
            documents (list): A list of the markdown nodes from the parsed document.
 
        Returns:
            list[BaseNode]: A list of the final chunked nodes.
        """
        self.logger.info("[Stage 1] Splitting on markdown structure...")
        markdown_parser = MarkdownNodeParser()
        markdown_nodes = markdown_parser.get_nodes_from_documents(
            documents, show_progress=True
        )
        self.logger.info(
            f"[Stage 1] Produced {len(markdown_nodes)} structural nodes"
        )
 
        self.logger.info(
            f"[Stage 2] Splitting oversized nodes — "
            f"chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}"
        )
        sentence_splitter = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        final_nodes = sentence_splitter.get_nodes_from_documents(
            markdown_nodes, show_progress=True
        )
 
        final_nodes = self._enrich_metadata(final_nodes)
        self.logger.info(
            f"[Stage 2] Final chunk count: {len(final_nodes)}"
        )
        return final_nodes
    
    def _enrich_metadata(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """
        Normalize and enrich metadata on every node.
 
        Stored fields:
            source       — PDF filename, for tracibility
            page_number  — normalised from LlamaParse's 'page_label'
            section      — section header path from MarkdownNodeParser,
                           used for rich citations in the chatbot response
        Args:
            nodes (list[BaseNode]): A list of nodes to enrich with consistent metadata 
            fields.
 
        Returns:
            list[BaseNode]: A list of enriched nodes with consistent metadata fields.
        """
        self.logger.info("Enriching node metadata...")
        source_name = os.path.basename(self.config.pdf_path)
 
        for node in nodes:
            node.metadata["source"] = source_name
            node.metadata["page_number"] = node.metadata.get(
                "page_label", "unknown"
            )
            node.metadata["section"] = node.metadata.get(
                "header_path", "unknown"
            )
 
        self.logger.info(
            f"Metadata enriched on {len(nodes)} nodes "
            f"(source, page_number, section)"
        )
        return nodes
    
    # Phase 3: Building the embeddings
    
    def _build_embed_model(self) -> FastEmbedEmbedding:
        """
        Initialise the FastEmbed embedding model.
 
        BAAI/bge-base-en-v1.5 is chosen because:
        - Top-ranked on the MTEB retrieval benchmark for its size class
        - Runs fully locally via FastEmbed — no API cost or rate limits
        - 768 dimensions give strong semantic resolution for retrieval
        - Well-supported by LlamaIndex and Qdrant native integrations
 
        Returns:
            FastEmbedEmbedding: An initialised FastEmbed model.
        """
        self.logger.info(
            f"[Stage 3] Initialising embedding model: {self.config.embed_model_name}"
        )
        return FastEmbedEmbedding(
            model_name=self.config.embed_model_name,
            cache_dir=self.config.fastembed_cache_dir,
        )
    
    # Phase 4: Connecting to Qdrant and uploading the nodes
    def _connect_qdrant(self) -> QdrantClient:
        """
        Establish connection to Qdrant Cloud.
 
        Returns:
            QdrantClient: An initialised Qdrant client.
        """
        self.logger.info("[Stage 4] Connecting to Qdrant Cloud...")
        client = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
        )
        self.logger.info("Connected to Qdrant Cloud successfully")
        return client
 
    def _setup_collection(self, client: QdrantClient) -> None:
        """
        Create the Qdrant collection if it does not already exist.
 
        Distance.COSINE is used because FastEmbed produces normalised vectors —
        cosine similarity measures directional alignment, which is what
        semantic search requires.
        
        If the collection already exists, a warning is logged and the existing
        collection is used. 
        
        Args:
            client (QdrantClient): An initialised Qdrant client.    
        Returns:
            None
        """
        existing = [c.name for c in client.get_collections().collections]
 
        if self.config.collection_name in existing:
            self.logger.warning(
                f"Collection '{self.config.collection_name}' already exists. "
                "Delete it from the Qdrant dashboard for a clean re-ingest."
            )
            return
 
        self.logger.info(
            f"Creating collection: '{self.config.collection_name}' "
            f"(dims={self.config.embed_dimensions}, metric=cosine)"
        )
        client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.embed_dimensions,
                distance=Distance.COSINE,
            ),
        )
        self.logger.info("Collection created successfully")
 
    def _upload( self, nodes: list[BaseNode], 
                embed_model: FastEmbedEmbedding, 
                client: QdrantClient,) -> None:
        """
        Embed all nodes and upload them to Qdrant with metadata.
        
        Args:
            nodes (list[BaseNode]): A list of nodes to be embedded and uploaded.    
            embed_model (FastEmbedEmbedding): An initialised FastEmbed model.
            client (QdrantClient): An initialised Qdrant client.
        Returns:
            None
        """
        self.logger.info(
            f"Embedding and uploading {len(nodes)} chunks to Qdrant..."
        )
 
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.config.collection_name,
        )
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
 
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )
 
        self.logger.info("Upload complete")
 
    # Helpers Functions
  
    def _validate_config(self) -> None:
        """
        Fail fast if any required config value is missing.
 
        Returns:
            None
        """
        required = {
            "pdf_path": self.config.pdf_path,
            "llama_cloud_api_key": self.config.llama_cloud_api_key,
            "qdrant_url": self.config.qdrant_url,
            "qdrant_api_key": self.config.qdrant_api_key,
            "collection_name": self.config.collection_name,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required configuration values: {missing}. "
            )
 
        if not os.path.exists(self.config.pdf_path):
            raise FileNotFoundError(
                f"PDF not found at path: '{self.config.pdf_path}'. "
                "Ensure the file is placed in the data/ directory."
            )
 
        self.logger.info("Configuration validated successfully")
 
    def _log_summary(self, result: IngestionResult) -> None:
        """
        Log a summary of the ingestion results in a clear format.
        Args:
            result (IngestionResult): The result object containing ingestion metrics.
        Returns:
            None
        """
        self.logger.info("=" * 60)
        self.logger.info("✅ Ingestion complete")
        self.logger.info(f"   Pages parsed   : {result.total_documents}")
        self.logger.info(f"   Chunks uploaded: {result.total_nodes}")
        self.logger.info(f"   Collection     : {result.collection_name}")
        self.logger.info(f"   Embed model    : {result.embed_model}")
        self.logger.info(
            f"   Chunk config   : size={result.chunk_size}, "
            f"overlap={result.chunk_overlap}"
        )
        self.logger.info("=" * 60)


        if __name__ == "__main__":
            config = IngestionConfig()
            pipeline = IngestionPipeline(config)
            result = pipeline.run()
        
            if not result.success:
                self.logger.error(f"Pipeline failed: {result.error}")
                sys.exit(1)
            else:
                self.logger.info("Pipeline completed successfully")
                sys.exit(0)