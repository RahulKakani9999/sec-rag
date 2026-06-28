"""Central configuration. Change settings here, not scattered through the code."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHROMA_DIR = ROOT / "chroma_db"

# Companies to index (ticker symbols). Start small, expand later.
COMPANIES = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
FILINGS_PER_COMPANY = 2  # most recent N 10-Ks

# Chunking
CHUNK_SIZE = 800          # tokens per chunk
CHUNK_OVERLAP = 100       # overlap between chunks

# Embedding model
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Retrieval
TOP_K_RETRIEVE = 20       # candidates pulled before reranking
TOP_K_RERANK = 5          # passed to the LLM after reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Generation
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_JUDGE_MODEL = "llama-3.1-8b-instant"  # cheaper model for RAGAS judge calls
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")
