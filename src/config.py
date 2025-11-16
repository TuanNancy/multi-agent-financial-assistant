"""
Cấu hình chung cho Multi-Agent Financial Assistant
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Đường dẫn dự án
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SESSIONS_DIR = PROJECT_ROOT / "sessions"

# ============================================================================
# LLM API Configuration
# ============================================================================
LLM_API_URL = os.getenv("LLM_API_URL", "")  # Ví dụ: https://api.openai.com/v1, http://localhost:11434
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # API key cho LLM service
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")  # Tên model, ví dụ: gpt-4, claude-3-sonnet, llama2

# LLM Parameters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

# ============================================================================
# Embedding Configuration
# ============================================================================
# Sentence Transformers Model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")  # cpu or cuda
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# FAISS Index Configuration
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "flat")  # flat, ivf, hnsw
FAISS_N_PROBES = int(os.getenv("FAISS_N_PROBES", "10"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# ============================================================================
# FinBERT Model Configuration
# ============================================================================
FINBERT_MODEL_PATH = PROJECT_ROOT / "models" / "finbert-trained" / "final"
FINBERT_MODEL_NAME = os.getenv("FINBERT_MODEL_NAME", "yiyanghkust/finbert-tone")
FINBERT_DEVICE = os.getenv("FINBERT_DEVICE", "cpu")  # cpu or cuda
FINBERT_MAX_LENGTH = int(os.getenv("FINBERT_MAX_LENGTH", "512"))
FINBERT_BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", "16"))

# ============================================================================
# Data Configuration
# ============================================================================
NEWS_INDEX_FILE = DATA_DIR / "news_index.json"
NEWS_RAW_DIR = DATA_DIR / "news_raw"
NEWS_INDEX_TYPE = os.getenv("NEWS_INDEX_TYPE", "faiss")  # faiss, json

# ============================================================================
# Session Configuration
# ============================================================================
SESSIONS_FILE = SESSIONS_DIR / "sessions.json"
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # seconds
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))

# ============================================================================
# Application Configuration
# ============================================================================
APP_NAME = "Multi-Agent Financial Assistant"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Streamlit Configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")

# ============================================================================
# Agent Configuration
# ============================================================================
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))  # seconds
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "30"))  # seconds

# ============================================================================
# RAG Configuration
# ============================================================================
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))

# ============================================================================
# Validation
# ============================================================================
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check LLM configuration (optional - chỉ cảnh báo nếu cần)
    if not LLM_API_URL or not LLM_MODEL_NAME:
        if DEBUG:
            print("Warning: LLM_API_URL and LLM_MODEL_NAME should be configured for LLM functionality")
    
    # Check model paths
    if not FINBERT_MODEL_PATH.exists():
        errors.append(f"FinBERT model path does not exist: {FINBERT_MODEL_PATH}")
    
    # Check data directories
    if not DATA_DIR.exists():
        errors.append(f"Data directory does not exist: {DATA_DIR}")
    
    if not SESSIONS_DIR.exists():
        errors.append(f"Sessions directory does not exist: {SESSIONS_DIR}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# Auto-validate on import (can be disabled if needed)
if os.getenv("VALIDATE_CONFIG", "True").lower() == "true":
    try:
        validate_config()
    except ValueError as e:
        if DEBUG:
            print(f"Warning: {e}")

