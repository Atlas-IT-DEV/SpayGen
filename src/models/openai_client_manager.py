import os
import httpx
from openai import AsyncOpenAI
from typing import Optional
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class OpenAIClientManager:
    _instance: Optional['OpenAIClientManager'] = None
    _client: Optional[AsyncOpenAI] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenAIClientManager, cls).__new__(cls)
        return cls._instance

    def _get_proxy_url(self) -> Optional[str]:
        """Constructs the proxy URL from environment variables."""
        proxy_host = os.getenv("PROXY_HOST")
        proxy_port = os.getenv("PROXY_PORT")
        proxy_user = os.getenv("PROXY_USER")
        proxy_pass = os.getenv("PROXY_PASS")

        if not all([proxy_host, proxy_port, proxy_user, proxy_pass]):
            logger.info("Proxy environment variables not fully set. Skipping proxy configuration.")
            return None

        return f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"

    def _set_proxy_env_vars(self, proxy_url: str) -> None:
        """Sets HTTP_PROXY and HTTPS_PROXY environment variables."""
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        logger.info(f"Set HTTP_PROXY and HTTPS_PROXY to {proxy_url}")

    def get_client(self) -> AsyncOpenAI:
        """Returns the OpenAI client, initializing it if necessary."""
        if self._client is None:
            logger.info("Initializing AsyncOpenAI client...")
            http_client_kwargs = {}
            proxy_url = self._get_proxy_url()

            if proxy_url:
                http_client_kwargs["proxy"] = proxy_url
                self._set_proxy_env_vars(proxy_url)

            try:
                self._client = AsyncOpenAI(
                    api_key=settings.openai_api_key,
                    base_url="https://api.openai.com/v1",
                    http_client=httpx.AsyncClient(**http_client_kwargs)
                )
                logger.info("AsyncOpenAI client initialized.")
            except Exception as e:
                logger.critical(f"Failed to initialize AsyncOpenAI client: {e}")
                raise
        return self._client

    async def close(self) -> None:
        """Closes the OpenAI client if it exists."""
        if self._client and hasattr(self._client, 'close'):
            try:
                await self._client.close()
                logger.info("AsyncOpenAI client closed successfully.")
            except Exception as e:
                logger.warning(f"Error closing AsyncOpenAI client: {e}")
            finally:
                self._client = None

# Singleton instance of OpenAIClientManager
openai_client_manager = OpenAIClientManager()

# Export the OpenAI client for use in other modules
openai_client: AsyncOpenAI = openai_client_manager.get_client()