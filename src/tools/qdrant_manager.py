from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
from pathlib import Path
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from src.models.pydantic_models import FullPageTemplate
from src.config.settings import settings

logger = logging.getLogger(__name__)

class AsyncQdrantManager:
    def __init__(self, host: str = "localhost", port: int = 6333, api_key: Optional[str] = None, timeout: float = 30.0, prefer_grpc: bool = False, max_retries: int = 3, retry_delay: float = 2.0, collection_name: str = "templates", embedding_dim: int = 1536):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.timeout = timeout
        self.prefer_grpc = prefer_grpc
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = None

    async def get_client(self) -> AsyncQdrantClient:
        if self.client is None:
            self.client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=self.timeout,
                prefer_grpc=self.prefer_grpc
            )
        return self.client

    async def close(self) -> None:
        if self.client:
            await self.client.close()
            self.client = None

    async def _ensure_collection(self, client: AsyncQdrantClient, collection_name: str) -> None:
        try:
            collection_info = await client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists: {collection_info}")
        except UnexpectedResponse:
            logger.info(f"Creating collection '{collection_name}'")
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    async def health_check(self) -> bool:
        try:
            client = await self.get_client()
            await client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def search_templates(self, query_embedding: List[float], limit: int = 5, collection_name: Optional[str] = None) -> List[Tuple[FullPageTemplate, float]]:
        """Search for templates using vector similarity."""
        try:
            client = await self.get_client()
            collection_name = collection_name or self.collection_name
            
            search_results = await client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for result in search_results:
                payload = result.payload
                template = FullPageTemplate(
                    name=payload["name"],
                    html=payload["html"],
                    css=payload["css"],
                    description=payload["description"],
                    tags=payload["tags"]
                )
                results.append((template, result.score))
            
            logger.info(f"Found {len(results)} templates with similarity search")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search templates: {e}")
            return []

    async def add_template(self, template: FullPageTemplate, embedding: List[float], collection_name: Optional[str] = None) -> bool:
        """Adds a template to the Qdrant collection."""
        try:
            client = await self.get_client()
            collection_name = collection_name or self.collection_name
            await self._ensure_collection(client, collection_name)

            point_id = str(uuid4())

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "name": template.name,
                    "html": template.html,
                    "css": template.css,
                    "description": template.description,
                    "tags": template.tags
                }
            )

            await client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            logger.info(f"Template '{template.name}' added to Qdrant with ID: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add template '{template.name}' to Qdrant: {e}")
            return False

    async def get_template(self, template_name: str, collection_name: Optional[str] = None) -> Optional[FullPageTemplate]:
        """Retrieves a template from the Qdrant collection by name."""
        try:
            client = await self.get_client()
            collection_name = collection_name or self.collection_name

            search_result = await client.search(
                collection_name=collection_name,
                query_vector=[0.0] * self.embedding_dim,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchValue(value=template_name)
                        )
                    ]
                ),
                limit=1
            )

            if search_result:
                payload = search_result[0].payload
                return FullPageTemplate(
                    name=payload["name"],
                    html=payload["html"],
                    css=payload["css"],
                    description=payload["description"],
                    tags=payload["tags"]
                )
            else:
                logger.info(f"Template '{template_name}' not found in Qdrant")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve template '{template_name}' from Qdrant: {e}")
            return None