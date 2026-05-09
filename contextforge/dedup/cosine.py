"""NumPy-vectorized cosine similarity - fixes BUG-005.

Replaces Python-level for-loop with O(dim) iteration with NumPy vectorized
operations. 384-dim embeddings: 1000 comparisons go from 384,000 Python ops
to ~20 NumPy calls under GIL release.

Usage:
    similarity = cosine_similarity(query_embedding, candidate_embedding)
    batch_scores = batch_cosine_similarity(query_embedding, list_of_embeddings)
"""
import asyncio
from typing import Optional

import numpy as np


def normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a vector or matrix."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1, norm)
    return vec / norm


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec_a: First vector (any shape)
        vec_b: Second vector (must match vec_a shape)
    
    Returns:
        Cosine similarity in range [-1, 1]
    """
    a_norm = normalize(vec_a.reshape(1, -1))
    b_norm = normalize(vec_b.reshape(1, -1))
    return float(np.dot(a_norm, b_norm.T).item())


def batch_cosine_similarity(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between one query and N candidates.
    Vectorized NumPy - no Python loops.
    
    Args:
        query: Query vector (dim,) or (1, dim)
        candidates: Candidate matrix (N, dim)
    
    Returns:
        Array of N similarity scores
    """
    # Ensure 2D
    if query.ndim == 1:
        query = query.reshape(1, -1)
    
    # Normalize
    q_norm = normalize(query)
    c_norm = normalize(candidates)
    
    # Inner product = cosine similarity (after normalization)
    scores = np.dot(q_norm, c_norm.T).flatten()
    
    return scores


async def batch_cosine_similarity_async(
    query: list[float],
    candidates: list[list[float]],
) -> np.ndarray:
    """
    Async wrapper for batch cosine similarity.
    Runs CPU-bound computation in ThreadPoolExecutor.
    
    Args:
        query: Query embedding vector
        candidates: List of candidate embedding vectors
    
    Returns:
        Array of similarity scores
    """
    loop = asyncio.get_event_loop()
    
    q_arr = np.array(query, dtype=np.float32)
    c_arr = np.array(candidates, dtype=np.float32)
    
    return await loop.run_in_executor(
        None, batch_cosine_similarity, q_arr, c_arr
    )


class VectorizedSimilarity:
    """
    Pre-compiled similarity engine for repeated queries.
    Avoids repeated normalization of candidates.
    """
    
    def __init__(self, dim: int = 384):
        self._dim = dim
        self._candidates: Optional[np.ndarray] = None
        self._candidate_ids: list[str] = []
    
    def index(self, agent_id: str, embedding: list[float]) -> None:
        """Add embedding to index."""
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = normalize(vec)
        
        if self._candidates is None:
            self._candidates = norm
        else:
            self._candidates = np.vstack([self._candidates, norm])
        
        self._candidate_ids.append(agent_id)
    
    def search(
        self,
        query: list[float],
        k: int = 10,
        threshold: float = 0.85,
    ) -> list[tuple[str, float]]:
        """
        Find top-k similar entries above threshold.
        
        Args:
            query: Query embedding
            k: Return top k results
            threshold: Minimum similarity score
        
        Returns:
            List of (agent_id, similarity) tuples
        """
        if self._candidates is None:
            return []
        
        q_arr = np.array(query, dtype=np.float32)
        scores = batch_cosine_similarity(q_arr, self._candidates)
        
        # Get top k indices
        top_k_idx = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            score = float(scores[idx])
            if score < threshold:
                continue
            agent_id = self._candidate_ids[idx]
            results.append((agent_id, score))
        
        return results
    
    @property
    def size(self) -> int:
        """Number of indexed entries."""
        return len(self._candidate_ids)
    
    def clear(self) -> None:
        """Clear index."""
        self._candidates = None
        self._candidate_ids = []
