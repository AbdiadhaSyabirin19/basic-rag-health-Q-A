import os
from dotenv import load_dotenv
 
load_dotenv()
 
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "artikel_kesehatan")
 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")
LLM_EVAL_MODEL = os.getenv("LLM_EVAL_MODEL", "qwen3:4b")  
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "7"))   # jumlah chunk yang diambil per pertanyaan (7 chunk agar prosedur langkah medis tidak terputus)
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.55"))
# Thinking mode Qwen3: default mati agar jawaban cepat & konsisten,
# akan dinyalakan otomatis untuk pertanyaan multi-hop (lihat rag_chain.py)
THINKING_MODE_DEFAULT = False

