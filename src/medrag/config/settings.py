"""Centralized project settings.

Values are loaded from environment variables first and fall back to simple
defaults so existing local runs continue to work without extra setup.
"""

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_path(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return BASE_DIR / path


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # Neo4j
    neo4j_uri: str = _env_str("NEO4J_URI", "http://localhost:7474")
    neo4j_user: str = _env_str("NEO4J_USER", "neo4j")
    neo4j_password: str = _env_str("NEO4J_PASSWORD", "all-in-rag")
    neo4j_database: str = _env_str("NEO4J_DATABASE", "neo4j")

    # Milvus / Zilliz Cloud
    milvus_host: str = _env_str("MILVUS_HOST", "localhost")
    milvus_port: int = _env_int("MILVUS_PORT", 19530)
    milvus_uri: str = _env_str("MILVUS_URI", "")
    milvus_token: str = _env_str("MILVUS_TOKEN", "")
    milvus_collection: str = _env_str("MILVUS_COLLECTION", "medical_qa")

    # Embedding
    embedding_model_name: str = _env_str(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5",
    )

    # Data paths
    toyhom_dataset_path: Path = _env_path(
        "TOYHOM_DATASET_PATH",
        "Chinese-medical-dialogue-data/Data_数据",
    )
    user_upload_case_path: Path = _env_path(
        "USER_UPLOAD_CASE_PATH",
        "user_uploads/cases",
    )

    # Retrieval
    retrieval_top_k: int = _env_int("RETRIEVAL_TOP_K", 10)
    rerank_top_k: int = _env_int("RERANK_TOP_K", 5)

    # LLM / DeepSeek
    llm_type: str = _env_str("LLM_TYPE", "deepseek")
    llm_provider: str = _env_str("LLM_PROVIDER", "deepseek")  # deepseek | zhipuai | ollama
    deepseek_api_key: str = _env_str("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_default_model: str = _env_str("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")
    deepseek_model_options: tuple[str, ...] = _env_tuple(
        "DEEPSEEK_MODEL_OPTIONS",
        (
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-v4-flash",
            "deepseek-v4-pro",
        ),
    )
    deepseek_intent_model: str = _env_str("DEEPSEEK_INTENT_MODEL", deepseek_default_model)
    deepseek_answer_model: str = _env_str("DEEPSEEK_ANSWER_MODEL", deepseek_default_model)

    # ZhipuAI (智谱)
    zhipuai_api_key: str = _env_str("ZHIPUAI_API_KEY", "")
    zhipuai_model: str = _env_str("ZHIPUAI_MODEL", "glm-4-plus")

    # Ollama
    ollama_base_url: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = _env_str("OLLAMA_MODEL", "qwen:32b")


settings = Settings()
