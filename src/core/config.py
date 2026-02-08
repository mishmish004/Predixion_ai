from pydantic_settings import BaseSettings, SettingsConfigDict
'''
In an ideal scenario i would have fetched each field from the .env file
'''

class ApplicationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "gsk_placeholder"
    llm_model_name: str = "llama-3.1-8b-instant"
    qdrant_collection_name: str = "ground_truth_precedents"
    redis_queue_name: str = "audit_queue"
    redis_results_prefix: str = "audit_result"
    fastembed_model_name: str = "BAAI/bge-small-en-v1.5"
    fastembed_vector_dimension: int = 384
    qdrant_search_limit: int = 3
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    worker_poll_interval_seconds: float = 1.0


settings = ApplicationSettings()