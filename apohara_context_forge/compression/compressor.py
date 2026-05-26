"""LLMLingua-2 async wrapper - runs in ThreadPoolExecutor."""
import asyncio
import logging
import os
from typing import Literal

from llmlingua import PromptCompressor

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Async wrapper for LLMLingua-2 compression."""

    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                 device_map: str | None = None):
        self._model_name = model_name
        # LLMLingua's PromptCompressor defaults to CUDA. When ContextForge's
        # coordinator runs on a host without an NVIDIA GPU (e.g. alongside an
        # AMD model server, or CPU-only), that raises "Found no NVIDIA driver".
        # Default to CPU; override via CONTEXTFORGE_COMPRESSOR_DEVICE.
        self._device_map = device_map or os.environ.get("CONTEXTFORGE_COMPRESSOR_DEVICE", "cpu")
        self._model: PromptCompressor | None = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Lazy load the compressor model."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    logger.info(f"Loading compressor: {self._model_name} (device={self._device_map})")
                    self._model = PromptCompressor(self._model_name, device_map=self._device_map)

    async def compress(self, context: str, rate: float = 0.5) -> tuple[str, float]:
        """
        Compress context at given rate.
        Returns (compressed_text, actual_compression_ratio).
        """
        await self.load()
        loop = asyncio.get_event_loop()
        
        def sync_compress():
            assert self._model is not None
            result = self._model.compress_prompt(
                context,
                rate=rate,
                force_tokens=[".", "!", "?", ",", "\n"],
            )
            return result["compressed_prompt"]

        compressed = await loop.run_in_executor(None, sync_compress)
        original_tokens = len(context.split())
        compressed_tokens = len(compressed.split())
        actual_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
        logger.debug(f"Compressed {original_tokens} -> {compressed_tokens} tokens (rate={rate})")
        return compressed, actual_ratio

    async def compress_batch(
        self, contexts: list[str], rate: float = 0.5
    ) -> list[tuple[str, float]]:
        """Compress multiple contexts concurrently."""
        tasks = [self.compress(ctx, rate) for ctx in contexts]
        results = await asyncio.gather(*tasks)
        return list(results)
