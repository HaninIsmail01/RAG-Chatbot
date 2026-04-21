
import os
from dotenv import load_dotenv

load_dotenv()
    
#Ingestion Related Configurations

# Qdrant Configurations
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "Guide_Collection"

#llama-index Configurations
LLAMA_INDEX_EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
FASTEMBED_CACHE_DIR = os.path.join("data", "fastembed_cache")
EMBED_DIMENSIONS = 768 

#llamaparse Configurations
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
LLAMAPARSE_MODEL = "gpt-3.5-turbo"
LLAMAPARSE_RESULT_TYPE ="markdown" # Options: "markdown", "json", "text"
LLAMAPARSE_PARSE_LANGUAGE = "en"

#Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64 

#Data configurations
DATA_DIR = "data/"
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")