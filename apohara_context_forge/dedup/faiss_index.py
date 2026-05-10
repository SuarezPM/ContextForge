"""FAISS ANN index for fast similarity search - IMPROVEMENT-006.

Replaces O(n) Python loop scan with O(log n) approximate nearest neighbor search.
Supports dynamic upgrade from flat to IVF index as registry grows.

Usage:
    index = FAISSContextIndex(dim=384)
    await index.add("agent1", embedding)
    matches = await index.search(query_embedding, k=10, threshold=0.92)

Scaling guide:
- < 1,000 contexts: IndexFlatIP (exact, fastest)
- 1K–100K contexts: IndexIVFFlat (approximate, ~10x faster)
- > 100K contexts: IndexHNSWFlat (graph-based, best recall/speed)
"""
import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


class FAISSMatch:
    """Represents a match from FAISS search."""
    __slots__ = ('agent_id', 'similarity', 'index_position')
    
    def __init__(self, agent_id: str, similarity: float, index_position: int):
        self.agent_id = agent_id
        self.similarity = similarity
        self.index_position = index_position


class FAISSContextIndex:
    """
    Approximate Nearest Neighbor index for fast similarity search.
    O(log n) search vs O(n) Python loop in v1.
    Thread-safe via asyncio executor pattern.
    
    Usage:
        index = FAISSContextIndex()
        await index.add("agent1", embedding)  # Add to index
        results = await index.search(query_embedding, k=5, threshold=0.9)
    """
    
    def __init__(self, dim: int = EMBEDDING_DIM):
        self._dim = dim
        self._index = None  # Will be set in _ensure_index
        self._id_map: dict[int, str] = {}  # FAISS internal ID -> agent_id
        self._reverse_map: dict[str, int] = {}  # agent_id -> FAISS internal ID
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def _ensure_index(self) -> None:
        """Lazy initialize index on first use."""
        if self._initialized:
            return
        
        import faiss
        async with self._lock:
            if self._initialized:
                return
            # Use IndexFlatIP (Inner Product) for cosine similarity (with normalized vectors)
            self._index = faiss.IndexFlatIP(self._dim)
            self._initialized = True
            logger.info(f"FAISS index initialized with dim={self._dim}")
    
    async def add(self, agent_id: str, embedding: list[float]) -> int:
        """
        Add embedding to index.
        
        Args:
            agent_id: Unique identifier for this embedding
            embedding: Dense embedding vector (dim,)
        
        Returns:
            FAISS internal index position
        """
        await self._ensure_index()
        
        vec = np.array([embedding], dtype=np.float32)
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vec)
        
        async with self._lock:
            idx = self._next_id
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._index.add, vec)
            self._id_map[idx] = agent_id
            self._reverse_map[agent_id] = idx
            self._next_id += 1
        
        return idx
    
    async def search(
        self,
        query: list[float],
        k: int = 10,
        threshold: float = 0.85,
    ) -> list[FAISSMatch]:
        """
        Find top-k similar entries above threshold.
        
        Args:
            query: Query embedding vector
            k: Number of results to return
            threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of FAISSMatch objects sorted by descending similarity
        """
        await self._ensure_index()
        
        q_vec = np.array([query], dtype=np.float32)
        faiss.normalize_L2(q_vec)
        
        loop = asyncio.get_event_loop()
        D, I = await loop.run_in_executor(
            None,
            lambda: self._index.search(q_vec, k)
        )
        
        matches = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            int_idx = int(idx)
            if int_idx not in self._id_map:
                continue
            
            similarity = float(score)
            if similarity < threshold:
                continue
            
            agent_id = self._id_map[int_idx]
            matches.append(FAISSMatch(
                agent_id=agent_id,
                similarity=similarity,
                index_position=int_idx
            ))
        
        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches
    
    async def remove(self, agent_id: str) -> bool:
        """
        Mark agent_id as removed (FAISS doesn't support true deletion from flat index).
        We just remove from the map; the vector stays but won't be returned.
        
        Args:
            agent_id: Agent to remove
        
        Returns:
            True if found and removed, False if not found
        """
        async with self._lock:
            if agent_id not in self._reverse_map:
                return False
            idx = self._reverse_map.pop(agent_id)
            self._id_map.pop(idx, None)
            return True
    
    async def get_embedding(self, agent_id: str) -> Optional[np.ndarray]:
        """Get stored embedding for agent_id (reconstruct from index)."""
        await self._ensure_index()
        
        async with self._lock:
            if agent_id not in self._reverse_map:
                return None
            idx = self._reverse_map[agent_id]
        
        if self._index.ntotal == 0:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            vec = await loop.run_in_executor(
                None,
                lambda: self._index.reconstruct(idx)
            )
            return vec
        except Exception:
            return None
    
    async def upgrade_to_ivf(self, nlist: int = 100) -> bool:
        """
        Upgrade from flat index to IVF when size > 1000.
        This requires retraining on the existing vectors.
        
        Args:
            nlist: Number of clusters (rule of thumb: sqrt(n))
        
        Returns:
            True if upgrade successful, False if skipped
        """
        if self._index is None or self._index.ntotal < 1000:
            logger.warning("IVF upgrade skipped: need > 1000 vectors for training")
            return False
        
        async with self._lock:
            # Can't upgrade in-place, so we rebuild
            import faiss
            ntotal = self._index.ntotal
            
            # Reconstruct all vectors
            all_vecs = np.zeros((ntotal, self._dim), dtype=np.float32)
            for i in range(ntotal):
                all_vecs[i] = self._index.reconstruct(i)
            
            # Create new IVF index
            quantizer = faiss.IndexFlatIP(self._dim)
            ivf_index = faiss.IndexIVFFlat(quantizer, self._dim, nlist)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ivf_index.train, all_vecs)
            await loop.run_in_executor(None, ivf_index.add, all_vecs)
            
            ivf_index.nprobe = 10  # Search 10 clusters
            
            self._index = ivf_index
            logger.info(f"Upgraded to IVF index with {nlist} clusters, nprobe=10")
            return True
    
    @property
    def size(self) -> int:
        """Number of indexed entries."""
        if self._index is None:
            return 0
        return self._index.ntotal
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    async def reset(self) -> None:
        """Clear the index."""
        async with self._lock:
            self._index = None
            self._id_map.clear()
            self._reverse_map.clear()
            self._next_id = 0
            self._initialized = False