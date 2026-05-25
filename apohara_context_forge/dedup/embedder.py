"""Sentence-transformers wrapper for async embedding generation."""
import asyncio
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Async-safe wrapper for sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Load the embedding model (lazy initialization)."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    logger.info(f"Loading embedder model: {self._model_name}")
                    self._model = SentenceTransformer(self._model_name)

    async def encode(self, text: str) -> list[float]:
        """Encode text to embedding vector."""
        await self.load()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self._model.encode, text
        )
        return embedding.tolist()

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts."""
        await self.load()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._model.encode, texts
        )
        return [e.tolist() for e in embeddings]
