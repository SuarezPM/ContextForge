"""Tests for LSHTokenMatcher and FAISSContextIndex - v2.0 deduplication components."""
import numpy as np
import pytest

from apohara_context_forge.dedup.faiss_index import FAISSContextIndex, FAISSMatch
from apohara_context_forge.dedup.lsh_engine import LSHTokenMatcher, TokenBlockMatch

pytestmark = pytest.mark.skipif(
    not __import__('importlib').util.find_spec('faiss'),
    reason="faiss-cpu not installed — run: pip install faiss-cpu"
)


@pytest.fixture
def lsh_matcher():
    """Create a fresh LSHTokenMatcher for each test."""
    return LSHTokenMatcher()


@pytest.fixture
def faiss_index():
    """Create a fresh FAISSContextIndex for each test."""
    return FAISSContextIndex(dim=384)


class TestLSHTokenMatcher:
    """Tests for LSHTokenMatcher - token-level SimHash matching."""

    @pytest.mark.asyncio
    async def test_index_prompt(self, lsh_matcher):
        """Index a prompt, verify blocks are stored."""
        # Create a prompt long enough to produce at least one full block (block_size=16)
        text = "This is a test prompt that should produce multiple token blocks for indexing."
        
        hashes = await lsh_matcher.index_prompt("agent1", text)
        
        # Verify blocks were indexed
        assert isinstance(hashes, list)
        
        # Check stats reflect the indexing
        stats = await lsh_matcher.stats()
        assert stats["total_blocks"] >= 1
        assert stats["total_agents"] == 1
        assert "agent1" in lsh_matcher._agent_blocks

    @pytest.mark.asyncio
    async def test_find_reusable_blocks(self, lsh_matcher):
        """Index one prompt, find matches in another with similar tokens."""
        # Index a prompt for agent1
        text1 = "You are a helpful assistant. You provide accurate and detailed responses."
        await lsh_matcher.index_prompt("agent1", text1)
        
        # Index another prompt for agent2 with identical beginning
        text2 = "You are a helpful assistant. Tell me about quantum physics."
        await lsh_matcher.index_prompt("agent2", text2)
        
        # Find reusable blocks in a new prompt with same prefix
        text3 = "You are a helpful assistant. What is machine learning?"
        matches = await lsh_matcher.find_reusable_blocks(text3)
        
        # Should find some matches since the prefix is the same
        assert isinstance(matches, list)
        # Matches should be sorted by hamming distance (best first)
        if len(matches) > 1:
            assert matches[0].hamming_distance <= matches[1].hamming_distance

    @pytest.mark.asyncio
    async def test_find_reusable_blocks_exclude_agent(self, lsh_matcher):
        """Verify exclude_agent parameter filters correctly."""
        text1 = "You are a helpful assistant. This is agent1's unique content here."
        await lsh_matcher.index_prompt("agent1", text1)
        
        text2 = "You are a helpful assistant. This is agent2's unique content here."
        await lsh_matcher.index_prompt("agent2", text2)
        
        # Search excluding agent1
        text3 = "You are a helpful assistant. This is agent1's unique content here."
        matches = await lsh_matcher.find_reusable_blocks(text3, exclude_agent="agent1")
        
        # Should not find any matches from agent1
        for match in matches:
            assert match.cached_agent_id != "agent1"

    @pytest.mark.asyncio
    async def test_get_shared_prefix_hash(self, lsh_matcher):
        """Compute stable hash of shared prefix."""
        text = "This is a test prompt for hashing."
        
        hash1 = await lsh_matcher.get_shared_prefix_hash(text)
        hash2 = await lsh_matcher.get_shared_prefix_hash(text)
        
        # Same text should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # First 32 chars of SHA256

    @pytest.mark.asyncio
    async def test_get_shared_prefix_hash_different_texts(self, lsh_matcher):
        """Different texts should produce different hashes."""
        text1 = "Hello world"
        text2 = "Goodbye world"
        
        hash1 = await lsh_matcher.get_shared_prefix_hash(text1)
        hash2 = await lsh_matcher.get_shared_prefix_hash(text2)
        
        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_lsh_stats(self, lsh_matcher):
        """Verify index statistics."""
        text = "This is a test prompt that should produce multiple token blocks."
        await lsh_matcher.index_prompt("agent1", text)
        await lsh_matcher.index_prompt("agent2", text)
        
        stats = await lsh_matcher.stats()
        
        assert "total_blocks" in stats
        assert "total_agents" in stats
        assert "block_size" in stats
        assert "hash_bits" in stats
        assert "hamming_threshold" in stats
        
        assert stats["total_agents"] == 2
        assert stats["block_size"] == 16
        assert stats["hash_bits"] == 64

    @pytest.mark.asyncio
    async def test_clear_agent(self, lsh_matcher):
        """Remove all blocks for an agent."""
        text = "This is a test prompt for clearing agent blocks."
        await lsh_matcher.index_prompt("agent1", text)
        
        stats_before = await lsh_matcher.stats()
        assert stats_before["total_agents"] == 1
        
        removed_count = await lsh_matcher.clear_agent("agent1")
        
        assert removed_count >= 0
        stats_after = await lsh_matcher.stats()
        assert stats_after["total_agents"] == 0
        assert stats_after["total_blocks"] == 0

    @pytest.mark.asyncio
    async def test_clear_agent_not_found(self, lsh_matcher):
        """Clearing non-existent agent returns 0."""
        removed = await lsh_matcher.clear_agent("nonexistent")
        assert removed == 0


class TestFAISSContextIndex:
    """Tests for FAISSContextIndex - approximate nearest neighbor search."""

    @pytest.mark.asyncio
    async def test_add_and_search(self, faiss_index):
        """Add embeddings, search, verify matches above threshold."""
        # Add two agents with embeddings
        emb1 = np.random.randn(384).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)  # Normalize
        
        emb2 = np.random.randn(384).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        idx1 = await faiss_index.add("agent1", emb1.tolist())
        idx2 = await faiss_index.add("agent2", emb2.tolist())
        
        assert idx1 == 0
        assert idx2 == 1
        
        # Search with nearly identical query
        query = emb1.tolist()  # Same as agent1's embedding
        matches = await faiss_index.search(query, k=10, threshold=0.85)
        
        assert isinstance(matches, list)
        assert len(matches) >= 1
        
        # Best match should be agent1 (highest similarity to itself)
        best = matches[0]
        assert isinstance(best, FAISSMatch)
        assert best.agent_id == "agent1"
        assert best.similarity > 0.99

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, faiss_index):
        """Verify threshold filtering works."""
        # Add an agent
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        await faiss_index.add("agent1", emb.tolist())
        
        # Search with very different query
        random_query = np.random.randn(384).astype(np.float32)
        random_query = random_query / np.linalg.norm(random_query)
        
        # High threshold should filter out dissimilar results
        matches = await faiss_index.search(random_query.tolist(), k=5, threshold=0.99)
        
        # Should either be empty or only contain very high similarity matches
        for match in matches:
            assert match.similarity >= 0.99

    @pytest.mark.asyncio
    async def test_search_returns_sorted_by_similarity(self, faiss_index):
        """Verify results are sorted by descending similarity."""
        # Add multiple agents with different embeddings
        for i in range(5):
            emb = np.random.randn(384).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            await faiss_index.add(f"agent{i}", emb.tolist())
        
        # Search
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        matches = await faiss_index.search(query, k=5, threshold=0.0)
        
        # Should be sorted by similarity descending
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].similarity >= matches[i + 1].similarity

    @pytest.mark.asyncio
    async def test_remove(self, faiss_index):
        """Remove agent from index."""
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        await faiss_index.add("agent1", emb.tolist())
        
        assert faiss_index.size == 1
        
        removed = await faiss_index.remove("agent1")
        assert removed is True
        
        # Size stays the same (FAISS limitation), but agent should not be found
        assert faiss_index.size == 1

    @pytest.mark.asyncio
    async def test_remove_not_found(self, faiss_index):
        """Removing non-existent agent returns False."""
        removed = await faiss_index.remove("nonexistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_size(self, faiss_index):
        """Verify index size tracking."""
        assert faiss_index.size == 0
        
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        await faiss_index.add("agent1", emb.tolist())
        assert faiss_index.size == 1
        
        await faiss_index.add("agent2", emb.tolist())
        assert faiss_index.size == 2
        
        await faiss_index.remove("agent1")
        assert faiss_index.size == 2  # FAISS doesn't actually remove

    @pytest.mark.asyncio
    async def test_multiple_searches(self, faiss_index):
        """Verify multiple searches work correctly."""
        # Add multiple agents
        embeddings = []
        for i in range(3):
            emb = np.random.randn(384).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            await faiss_index.add(f"agent{i}", emb.tolist())
        
        # Multiple searches should all work
        for emb in embeddings:
            matches = await faiss_index.search(emb.tolist(), k=3, threshold=0.5)
            assert len(matches) >= 1


class TestTokenBlockMatch:
    """Tests for TokenBlockMatch dataclass."""

    def test_token_block_match_creation(self):
        """Verify TokenBlockMatch has all required fields."""
        match = TokenBlockMatch(
            block_index=0,
            cached_block_hash=12345,
            hamming_distance=2,
            reuse_confidence=0.97,
            cached_agent_id="agent1"
        )
        
        assert match.block_index == 0
        assert match.cached_block_hash == 12345
        assert match.hamming_distance == 2
        assert match.reuse_confidence == 0.97
        assert match.cached_agent_id == "agent1"


class TestFAISSMatch:
    """Tests for FAISSMatch dataclass."""

    def test_faiss_match_creation(self):
        """Verify FAISSMatch has all required fields."""
        match = FAISSMatch(
            agent_id="agent1",
            similarity=0.95,
            index_position=5
        )
        
        assert match.agent_id == "agent1"
        assert match.similarity == 0.95
        assert match.index_position == 5
