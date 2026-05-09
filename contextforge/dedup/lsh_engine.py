"""LSH Token-Level Matching Engine - IMPROVEMENT-001.

Token-level fuzzy matching using SimHash for KV cache block reuse.
Operates on actual token IDs from Qwen3 tokenizer, not word-level strings.
Aligns to vLLM PagedAttention block boundaries (default block_size=16).

Architecture:
    Incoming prompt (text)
        │
        ▼
   Qwen3 Tokenizer         ← Real token IDs, not word splits
        │
        ▼
  LSH Block Hashing        ← SimHash on token blocks
        │
        ▼
  Block Alignment          ← Align to PagedAttention blocks (16 tokens)
        │
        ▼
  Match Candidates         ← Find blocks with hamming distance < threshold
        │
        ▼
  Reuse Decision           → List of reusable block indices

Usage:
    matcher = LSHTokenMatcher()
    await matcher.index_prompt("agent1", "shared system prompt...")
    matches = await matcher.find_reusable_blocks("new incoming prompt...")
"""
import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from contextforge.token_counter import TokenCounter

logger = logging.getLogger(__name__)

# vLLM PagedAttention default block size
VLLM_BLOCK_SIZE = 16


@dataclass
class TokenBlockMatch:
    """A matching block found in the LSH index."""
    block_index: int          # Which block position in the new prompt
    cached_block_hash: int  # 64-bit SimHash of the matching cached block
    hamming_distance: int   # Lower = more similar (0 = identical)
    reuse_confidence: float  # 0.0-1.0 derived from hamming distance
    cached_agent_id: str     # Which agent owns the cached block


class LSHTokenMatcher:
    """
    Token-level fuzzy matching using SimHash for KV cache block reuse.
    Operates on actual token IDs from Qwen3 tokenizer.
    
    Key insight: vLLM PagedAttention shares KV cache for identical token blocks.
    Two prompts with 95% SBERT similarity but different wording may share ZERO cache.
    LSH finds actual token-level matches at block boundaries.
    
    Usage:
        matcher = LSHTokenMatcher()
        await matcher.index_prompt("agent1", system_prompt)
        matches = await matcher.find_reusable_blocks(new_prompt)
    """
    
    def __init__(
        self,
        block_size: int = VLLM_BLOCK_SIZE,
        hash_bits: int = 64,
        hamming_threshold: int = 8,  # <8 bits different = high confidence
    ):
        self._block_size = block_size
        self._hash_bits = hash_bits
        self._hamming_threshold = hamming_threshold
        self._token_counter = TokenCounter.get()
        self._block_store: dict[int, tuple[tuple[int, ...], str]] = {}  # hash → (tokens, agent_id)
        self._agent_blocks: dict[str, list[int]] = {}  # agent_id → list of block hashes
        self._lock = asyncio.Lock()
    
    @staticmethod
    def _hamming(a: int, b: int) -> int:
        """Compute Hamming distance between two 64-bit integers."""
        return bin(a ^ b).count('1')
    
    async def index_prompt(
        self,
        agent_id: str,
        text: str,
    ) -> list[int]:
        """
        Tokenize, blockify, and index a prompt for future reuse.
        Stores block hashes in LSH index.
        
        Args:
            agent_id: Owner of this prompt
            text: Full prompt text
        
        Returns:
            List of block hashes that were indexed
        """
        loop = asyncio.get_event_loop()
        token_ids = await loop.run_in_executor(
            None, self._token_counter.encode, text
        )
        
        hashes = []
        blocks = []
        
        # Create blocks aligned to vLLM PagedAttention boundaries
        for i in range(0, len(token_ids), self._block_size):
            block = tuple(token_ids[i:i + self._block_size])
            
            # Skip partial blocks (no cache guarantee for < block_size)
            if len(block) < self._block_size:
                continue
            
            block_hash = self._simhash_block(block)
            self._block_store[block_hash] = (block, agent_id)
            hashes.append(block_hash)
            blocks.append(block_hash)
        
        async with self._lock:
            self._agent_blocks[agent_id] = hashes
        
        logger.debug(f"Indexed {len(hashes)} blocks for agent {agent_id}")
        return hashes
    
    async def find_reusable_blocks(
        self,
        text: str,
        exclude_agent: Optional[str] = None,
    ) -> list[TokenBlockMatch]:
        """
        Find cached blocks that can be reused for this prompt.
        
        Args:
            text: New prompt text
            exclude_agent: Optionally exclude blocks from a specific agent
        
        Returns:
            List of TokenBlockMatch sorted by hamming distance (best first)
        """
        loop = asyncio.get_event_loop()
        token_ids = await loop.run_in_executor(
            None, self._token_counter.encode, text
        )
        
        matches = []
        
        for i in range(0, len(token_ids), self._block_size):
            block = tuple(token_ids[i:i + self._block_size])
            
            if len(block) < self._block_size:
                continue
            
            new_hash = self._simhash_block(block)
            
            # Search for similar blocks
            for cached_hash, (cached_tokens, agent_id) in self._block_store.items():
                if exclude_agent and agent_id == exclude_agent:
                    continue
                
                hd = self._hamming(new_hash, cached_hash)
                
                if hd <= self._hamming_threshold:
                    confidence = 1.0 - (hd / self._hash_bits)
                    matches.append(TokenBlockMatch(
                        block_index=i // self._block_size,
                        cached_block_hash=cached_hash,
                        hamming_distance=hd,
                        reuse_confidence=confidence,
                        cached_agent_id=agent_id,
                    ))
        
        # Sort by hamming distance (best = lowest)
        matches.sort(key=lambda m: m.hamming_distance)
        return matches
    
    async def get_shared_prefix_hash(self, text: str) -> str:
        """
        Compute a stable hash of the shared prefix (first block).
        Used for routing hints to llm-d/vLLM.
        
        Args:
            text: Prompt text
        
        Returns:
            SHA256 hex string of first block's tokens
        """
        loop = asyncio.get_event_loop()
        token_ids = await loop.run_in_executor(
            None, self._token_counter.encode, text
        )
        
        if len(token_ids) < self._block_size:
            first_block = token_ids
        else:
            first_block = token_ids[:self._block_size]
        
        # Create deterministic hash
        hash_input = str(tuple(first_block)).encode()
        return hashlib.sha256(hash_input).hexdigest()[:32]  # First 32 chars
    
    def _simhash_block(self, token_ids: tuple[int, ...]) -> int:
        """
        Compute 64-bit SimHash fingerprint for a token block.
        
        Uses stable pseudo-random projection per token ID.
        Deterministic: same block always produces same hash.
        
        Args:
            token_ids: Tuple of token IDs
        
        Returns:
            64-bit integer hash
        """
        v = np.zeros(self._hash_bits, dtype=np.float32)
        
        for tid in token_ids:
            # Deterministic pseudo-random projection
            # Using xorshift for speed (avoids numpy RNG object creation)
            h = int(tid)
            for _ in range(4):  # Mix well
                h ^= h << 13
                h ^= h >> 7
                h ^= h << 17
                h = h & 0xFFFFFFFF
            
            # Project onto hash bits
            for bit in range(self._hash_bits):
                if (h >> (bit % 32)) & 1:
                    v[bit] += 1
                else:
                    v[bit] -= 1
        
        # Binarize
        bits = (v > 0).astype(np.uint8)
        
        # Pack into int64
        result = 0
        for i, b in enumerate(bits):
            result |= (int(b) << i)
        
        return result
    
    async def stats(self) -> dict:
        """Return index statistics."""
        async with self._lock:
            return {
                "total_blocks": len(self._block_store),
                "total_agents": len(self._agent_blocks),
                "block_size": self._block_size,
                "hash_bits": self._hash_bits,
                "hamming_threshold": self._hamming_threshold,
            }
    
    async def clear_agent(self, agent_id: str) -> int:
        """
        Remove all blocks indexed for an agent.
        
        Args:
            agent_id: Agent to clear
        
        Returns:
            Number of blocks removed
        """
        async with self._lock:
            hashes = self._agent_blocks.pop(agent_id, [])
            for h in hashes:
                if h in self._block_store:
                    del self._block_store[h]
            return len(hashes)