import os
from dotenv import load_dotenv
from pathlib import Path
from dotenv import load_dotenv

# Always resolve from this file's location upward to the project root
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_env_path)
    
#Ingestion Related Configurations

# Qdrant Configurations
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "Guide_Collection"

#llama-index embedding Configurations
LLAMA_INDEX_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
FASTEMBED_CACHE_DIR = os.path.join("data", "fastembed_cache")
EMBED_DIMENSIONS = 768 

#llama-cloud Configurations
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_PARSE_TIER = "agentic"        # agentic | cost_effective | fast
LLAMA_PARSE_LANGUAGE = "en"

#Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64 

#Retrieval with Reranking Configurations
RERANKER_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"
RETRIEVAL_TOP_K = 10
RERANKER_TOP_N = 5

#Data configurations
DATA_DIR = "data\\iPhone User Guide 2.pdf"
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")