# config.py
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

class RAGConfig(BaseSettings):
    # Data
    children_path: str = "chunked/children.json"
    parents_path: str = "chunked/parents.json"
    index_dir: str = "faiss_index"
    
    # Model
    embedding_model: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 32
    
    # Retrieval
    top_k: int = 3
    similarity_threshold: float = 0.35
    
    # BM25
    use_hybrid: bool = True
    bm25_index_path: str = "bm25_index.pkl"
    rrf_k: int = 60
    hybrid_candidate_k: int = 20
    
    # 🆕 LLM (OpenRouter)
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    # openrouter_model: str = "nvidia/nemotron-3.5-lightning:free" # Change to any OpenRouter model ID
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct"
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Dedup (shared sections repeated across documents)
    dedup_results: bool = True
    dedup_threshold: float = 0.9   # similarity ratio to consider two chunks "the same section"
    dedup_fetch_k: int = 15        # candidates fetched before dedup, so you still end up with k diverse results

    # ---- Admin-controlled generation settings (users cannot change) ----
    openrouter_model: str = "meta-llama/llama-3.1-8b-instruct"
    use_chat_memory: bool = True
    memory_messages: int = 3      # how many past messages the LLM sees per question

settings = RAGConfig()