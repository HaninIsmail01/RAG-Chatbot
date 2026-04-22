import os
from dotenv import load_dotenv
from pathlib import Path
from dotenv import load_dotenv

# Always resolve from this file's location upward to the project root
_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_env_path)
    
#Ingestion Related Configurations

import nltk

NLTK_DATA = os.getenv("NLTK_DATA", ".nltk_data")
os.makedirs(NLTK_DATA, exist_ok=True)
nltk.data.path.insert(0, os.path.abspath(NLTK_DATA))

# Download required corpora to the local directory if not already present
for corpus in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{corpus}")
    except LookupError:
        nltk.download(corpus, download_dir=os.path.abspath(NLTK_DATA))

# Qdrant Configurations
QDRANT_HOST = "https://1eb088d7-6481-4a4a-9d8f-c1bfcaf51566.eu-central-1-0.aws.cloud.qdrant.io"
QDRANT_PORT = 6333
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
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
RETRIEVAL_TOP_K = 10
RERANKER_TOP_N = 5

#Data configurations
DATA_DIR = "data\\iPhone User Guide 2.pdf"
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")

# Generation Configurations
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0"))

#Prompts

SYSTEM_PROMPT = """
You are a document assistant. Your only job is to answer \
questions based strictly on the context provided below.

RULES YOU MUST FOLLOW WITHOUT EXCEPTION:
1. Answer ONLY from the provided context. Do not use any external or general knowledge.
2. If the answer cannot be found in the context, respond with exactly:
   "I could not find the answer to your question in the provided document."
3. Never fabricate, infer, or guess information that is not explicitly stated.
4. Always be concise and precise. Do not pad your answer.
5. You may reference prior conversation turns to understand the question, \
but your answer must still come from the context only.
"""

CITATION_INSTRUCTION = """
Always end your answer before the sources block. \
Do not reproduce the chunk headers in your answer.
"""