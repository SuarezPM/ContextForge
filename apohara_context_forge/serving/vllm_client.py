"""Async HTTP client for vLLM OpenAI-compatible API."""
import logging
from typing import Any

import httpx

from apohara_context_forge.config import settings

logger = logging.getLogger(__name__)


class vLLMClient:
    """Async client for vLLM server."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self._base_url = base_url or settings.vllm_base_url
        self._api_key = api_key or settings.vllm_api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=60.0,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
        """Send completion request to vLLM."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60.0,
            )

        payload = {
            "model": settings.vllm_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            response = await self._client.post("/v1/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"vLLM request failed: {e}")
            return {"error": str(e)}

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
        """Send chat completion request."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=60.0,
            )

        payload = {
            "model": settings.vllm_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"vLLM chat request failed: {e}")
            return {"error": str(e)}
