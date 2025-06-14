from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):

    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    qdrant_host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_timeout: float = Field(default_factory=lambda: float(os.getenv("QDRANT_TIMEOUT", "30.0")))
    qdrant_prefer_grpc: bool = Field(default_factory=lambda: os.getenv("QDRANT_PREFER_GRPC", "False").upper() == "TRUE")
    qdrant_max_retries: int = Field(default_factory=lambda: int(os.getenv("QDRANT_MAX_RETRIES", "3")))
    qdrant_retry_delay: float = Field(default_factory=lambda: float(os.getenv("QDRANT_RETRY_DELAY", "2.0")))
    qdrant_collection_name: str = Field(default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "templates"))

    embedding_dim: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1536")))

    server_host: str = Field(default_factory=lambda: os.getenv("SERVER_HOST", "0.0.0.0"))
    server_port: int = Field(default_factory=lambda: int(os.getenv("SERVER_PORT", "8000")))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "FALSE").upper() == "TRUE")

    bing_required_policies: List[str] = Field(
        default_factory=lambda: [
            "privacy policy", "terms of service", "return policy",
            "payment policy", "shipping policy"
        ],
        description="Required policies for Bing Ads compliance."
    )
    bing_forbidden_keywords: List[str] = Field(
        default_factory=lambda: [
            "crypto", "trading", "investment", "financial advice",
            "medical", "supplements", "beauty masks", "perfume",
            "news", "games", "weapons", "killing", "solar panels",
            "aviation", "food", "restaurant"
        ],
        description="Forbidden keywords for Bing Ads compliance."
    )

    html_required_tags: List[str] = Field(
        default_factory=lambda: ['html', 'head', 'body', 'title'],
        description="Required HTML tags for basic structure validation."
    )
    html_required_meta_tags: List[str] = Field(
        default_factory=lambda: ['charset', 'viewport'],
        description="Required meta tags for HTML validation."
    )
    html_forbidden_tags: List[str] = Field(
        default_factory=lambda: ['script', 'iframe', 'embed', 'object'],
        description="Forbidden HTML tags for security/compliance validation."
    )
    css_forbidden_properties: List[str] = Field(
        default_factory=lambda: ['position: fixed', 'position: absolute'],
        description="Forbidden CSS properties for layout/security validation."
    )

    model_settings: Dict[str, Any] = {
        "model": "o4-mini",
        "temperature": 1.0,
        "max_tokens": 36000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = 'ignore'

settings = Settings()