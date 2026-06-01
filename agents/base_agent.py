"""Base agent with ContextForge and vLLM integration."""
from abc import ABC, abstractmethod
from typing import Any
import logging
import time

import httpx

from apohara_context_forge.config import settings
from apohara_context_forge.token_counter import TokenCounter

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
            response.raise_for_status()
            return response.json()

    async def call_contextforge_optimize(self, context: str) -> dict[str, Any]:
        """Get optimized context from ContextForge."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:{settings.contextforge_port}/tools/get_optimized_context",
                json={"agent_id": self.agent_id, "context": context},
            )
            response.raise_for_status()
            data = response.json()
            # When the server reports original_tokens=0 on a non-empty context
            # (e.g. coordinator_unavailable passthrough), fall back to a local
            # count so downstream metrics stay accurate.
            #
            # Bug fix: previously ``len(context.split())`` (whitespace
            # word count) was used; that under-counts for code and multibyte
            # text by a factor of ~1.3-3x.  We now route through
            # ``TokenCounter``, which uses the same Qwen3 tokenizer as the
            # registry and LSH engine.  This keeps the client-side fallback
            # consistent with what the server would have reported.
            if not data.get("original_tokens") and context:
                data["original_tokens"] = TokenCounter.get().count(context)
            return data

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
            tuple of (response_text, request_latency_ms)

        Notes:
            TTFT (time-to-first-token) requires streaming and a
            per-token callback.  This method awaits the *full*
            non-streaming response, so the returned latency is the
            request-total wall time, not first-token latency.
            Renamed to stop mislabelling the value.
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

        # TTFT requires streaming; this is request-total latency, not first-token latency.
        request_latency_ms = (time.perf_counter() - start) * 1000
        content = r.json()["choices"][0]["message"]["content"]
        return content, request_latency_ms