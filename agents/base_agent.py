"""Base agent with ContextForge and vLLM integration."""
from abc import ABC, abstractmethod
from typing import Any
import logging
import time

import httpx

from apohara_context_forge.config import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract agent with ContextForge integration."""

    def __init__(self, agent_id: str, role: str, thinking: bool = False):
        self.agent_id = agent_id
        self.role = role
        self.thinking = thinking

    @abstractmethod
    async def process(self, input_data: Any) -> dict[str, Any]:
        """Process input and return result with metrics."""
        pass

    async def call_contextforge_register(self, context: str) -> dict[str, Any]:
        """Register context with ContextForge MCP server."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:{settings.contextforge_port}/tools/register_context",
                json={"agent_id": self.agent_id, "context": context},
            )
            return response.json()

    async def call_contextforge_optimize(self, context: str) -> dict[str, Any]:
        """Get optimized context from ContextForge."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:{settings.contextforge_port}/tools/get_optimized_context",
                json={"agent_id": self.agent_id, "context": context},
            )
            return response.json()

    async def call_vllm(
        self,
        prompt: str,
        thinking: bool | None = None,
    ) -> tuple[str, float]:
        """
        Call vLLM for completion with optional thinking mode.
        
        Args:
            prompt: The input prompt
            thinking: Override thinking mode (default: self.thinking)
            
        Returns:
            tuple of (response_text, ttft_ms)
        """
        use_thinking = thinking if thinking is not None else self.thinking
        
        start = time.perf_counter()
        payload = {
            "model": settings.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0 if not use_thinking else 0.6,
            "top_p": 0.95 if use_thinking else 1.0,
            "extra_body": {
                "thinking": use_thinking,
            },
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{settings.vllm_base_url}/v1/chat/completions",
                json=payload,
            )
            r.raise_for_status()
        
        ttft_ms = (time.perf_counter() - start) * 1000
        content = r.json()["choices"][0]["message"]["content"]
        return content, ttft_ms