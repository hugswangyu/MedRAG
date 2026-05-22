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


@dataclass(frozen=True)
class Settings:
    # Neo4j
    neo4j_uri: str = _env_str("NEO4J_URI", "http://localhost:7474")
    neo4j_user: str = _env_str("NEO4J_USER", "neo4j")
    neo4j_password: str = _env_str("NEO4J_PASSWORD", "all-in-rag")
    neo4j_database: str = _env_str("NEO4J_DATABASE", "neo4j")

    # Milvus
    milvus_host: str = _env_str("MILVUS_HOST", "localhost")
    milvus_port: int = _env_int("MILVUS_PORT", 19530)
    milvus_collection: str = _env_str("MILVUS_COLLECTION", "medical_qa")

    # Embedding
    embedding_model_name: str = _env_str(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-base-zh-v1.5",
    )

    # Data paths
    toyhom_dataset_path: Path = _env_path(
        "TOYHOM_DATASET_PATH",
        "data/toyhom",
    )
    user_upload_case_path: Path = _env_path(
        "USER_UPLOAD_CASE_PATH",
        "user_uploads/cases",
    )

    # Retrieval
    retrieval_top_k: int = _env_int("RETRIEVAL_TOP_K", 10)
    rerank_top_k: int = _env_int("RERANK_TOP_K", 5)

    # LLM
    llm_type: str = _env_str("LLM_TYPE", "zhipuai")


settings = Settings()


# Backward-friendly constant aliases for modules that prefer direct imports.
NEO4J_URI = settings.neo4j_uri
NEO4J_USER = settings.neo4j_user
NEO4J_PASSWORD = settings.neo4j_password
NEO4J_DATABASE = settings.neo4j_database

MILVUS_HOST = settings.milvus_host
MILVUS_PORT = settings.milvus_port
MILVUS_COLLECTION = settings.milvus_collection

EMBEDDING_MODEL_NAME = settings.embedding_model_name
TOYHOM_DATASET_PATH = settings.toyhom_dataset_path
USER_UPLOAD_CASE_PATH = settings.user_upload_case_path

RETRIEVAL_TOP_K = settings.retrieval_top_k
RERANK_TOP_K = settings.rerank_top_k
LLM_TYPE = settings.llm_type
