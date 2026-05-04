import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

class Config:
    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
    MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")
    MINIMAX_BASE_URL = "https://api.minimax.chat/v1"
    MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "")
    MINIMAX_TEMPERATURE = 0.7

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    VECTOR_DIMENSION = 1024
    VECTOR_INDEX = "doc_vectors"
    DOC_METADATA_PREFIX = "doc:metadata:"
    SESSION_PREFIX = "session:"
    SESSION_TTL = 86400

    MINIMAX_EMBEDDING_MODEL = "embo-01"
    EMBEDDING_DIMENSION = 1536
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    DOCS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
