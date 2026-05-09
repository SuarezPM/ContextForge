"""5 concrete demo agents simulating a RAG pipeline."""
import asyncio
import logging
from typing import Any

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

AGENT_CONFIGS = [
    {
        "id": "retriever",
        "role": "retrieve relevant documents from the corpus",
        "context_overlap": 0.6,
        "thinking": False,   # speed-critical, no CoT needed
    },
    {
        "id": "reranker",
        "role": "rerank retrieved documents by relevance",
        "context_overlap": 0.7,
        "thinking": False,   # deterministic ranking, no CoT needed
    },
    {
        "id": "summarizer",
        "role": "summarize retrieved documents into coherent context",
        "context_overlap": 0.6,
        "thinking": False,   # structured output, no CoT needed
    },
    {
        "id": "critic",
        "role": "verify factual accuracy and flag hallucinations",
        "context_overlap": 0.5,
        "thinking": True,    # reasoning-heavy, CoT improves accuracy
    },
    {
        "id": "responder",
        "role": "generate final user-facing response",
        "context_overlap": 0.4,
        "thinking": True,    # quality-critical final output
    },
]


class RetrieverAgent(BaseAgent):
    """Agent 1: Retrieves relevant documents."""

    def __init__(self):
        super().__init__("retriever", "retrieve relevant documents", thinking=False)

    async def process(self, input_data: Any) -> dict[str, Any]:
        shared_context = self._build_shared_context(input_data)
        
        try:
            await self.call_contextforge_register(shared_context)
            decision = await self.call_contextforge_optimize(shared_context)
        except Exception as e:
            logger.warning(f"ContextForge unavailable, using raw context: {e}")
            decision = {"strategy": "passthrough", "original_tokens": len(shared_context.split())}

        result = f"[{self.agent_id}] Retrieved docs for query: {input_data.get('query', 'unknown')}"
        return {
            "agent_id": self.agent_id,
            "result": result,
            "strategy": decision.get("strategy", "passthrough"),
            "tokens_before": decision.get("original_tokens", 0),
            "tokens_after": decision.get("final_tokens", 0),
        }

    def _build_shared_context(self, input_data: Any) -> str:
        return f"""System: You are a retriever agent.
Query: {input_data.get('query', '')}
Knowledge base: Document 1 about AI, Document 2 about ML, Document 3 about NLP.
Role: {self.role}
Instruction: Retrieve the most relevant documents."""


class RerankerAgent(BaseAgent):
    """Agent 2: Reranks documents by relevance."""

    def __init__(self):
        super().__init__("reranker", "rerank by relevance", thinking=False)

    async def process(self, input_data: Any) -> dict[str, Any]:
        prev_output = input_data.get("retriever_output", "")
        shared_context = self._build_shared_context(input_data, prev_output)

        try:
            await self.call_contextforge_register(shared_context)
            decision = await self.call_contextforge_optimize(shared_context)
        except Exception as e:
            logger.warning(f"ContextForge unavailable: {e}")
            decision = {"strategy": "passthrough", "original_tokens": len(shared_context.split())}

        result = f"[{self.agent_id}] Reranked documents by relevance scores"
        return {
            "agent_id": self.agent_id,
            "result": result,
            "strategy": decision.get("strategy", "passthrough"),
            "tokens_before": decision.get("original_tokens", 0),
            "tokens_after": decision.get("final_tokens", 0),
        }

    def _build_shared_context(self, input_data: Any, prev_output: str) -> str:
        return f"""System: You are a reranker agent.
Previous: {prev_output}
Query: {input_data.get('query', '')}
Role: {self.role}
Instruction: Rerank documents by relevance scores."""


class SummarizerAgent(BaseAgent):
    """Agent 3: Summarizes retrieved documents."""

    def __init__(self):
        super().__init__("summarizer", "summarize retrieved docs", thinking=False)

    async def process(self, input_data: Any) -> dict[str, Any]:
        prev_output = input_data.get("reranker_output", "")
        shared_context = self._build_shared_context(input_data, prev_output)

        try:
            await self.call_contextforge_register(shared_context)
            decision = await self.call_contextforge_optimize(shared_context)
        except Exception as e:
            logger.warning(f"ContextForge unavailable: {e}")
            decision = {"strategy": "passthrough", "original_tokens": len(shared_context.split())}

        result = f"[{self.agent_id}] Summarized documents into key points"
        return {
            "agent_id": self.agent_id,
            "result": result,
            "strategy": decision.get("strategy", "passthrough"),
            "tokens_before": decision.get("original_tokens", 0),
            "tokens_after": decision.get("final_tokens", 0),
        }

    def _build_shared_context(self, input_data: Any, prev_output: str) -> str:
        return f"""System: You are a summarizer agent.
Previous: {prev_output}
Query: {input_data.get('query', '')}
Role: {self.role}
Instruction: Summarize the retrieved documents into key points."""


class CriticAgent(BaseAgent):
    """Agent 4: Verifies factual accuracy."""

    def __init__(self):
        super().__init__("critic", "verify factual accuracy", thinking=True)

    async def process(self, input_data: Any) -> dict[str, Any]:
        prev_output = input_data.get("summarizer_output", "")
        shared_context = self._build_shared_context(input_data, prev_output)

        try:
            await self.call_contextforge_register(shared_context)
            decision = await self.call_contextforge_optimize(shared_context)
        except Exception as e:
            logger.warning(f"ContextForge unavailable: {e}")
            decision = {"strategy": "passthrough", "original_tokens": len(shared_context.split())}

        result = f"[{self.agent_id}] Verified factual accuracy of summary"
        return {
            "agent_id": self.agent_id,
            "result": result,
            "strategy": decision.get("strategy", "passthrough"),
            "tokens_before": decision.get("original_tokens", 0),
            "tokens_after": decision.get("final_tokens", 0),
        }

    def _build_shared_context(self, input_data: Any, prev_output: str) -> str:
        return f"""System: You are a critic agent.
Previous: {prev_output}
Query: {input_data.get('query', '')}
Role: {self.role}
Instruction: Verify factual accuracy and identify issues."""


class ResponderAgent(BaseAgent):
    """Agent 5: Generates final response."""

    def __init__(self):
        super().__init__("responder", "generate final response", thinking=True)

    async def process(self, input_data: Any) -> dict[str, Any]:
        prev_output = input_data.get("critic_output", "")
        shared_context = self._build_shared_context(input_data, prev_output)

        try:
            await self.call_contextforge_register(shared_context)
            decision = await self.call_contextforge_optimize(shared_context)
        except Exception as e:
            logger.warning(f"ContextForge unavailable: {e}")
            decision = {"strategy": "passthrough", "original_tokens": len(shared_context.split())}

        result = f"[{self.agent_id}] Generated final response to query"
        return {
            "agent_id": self.agent_id,
            "result": result,
            "strategy": decision.get("strategy", "passthrough"),
            "tokens_before": decision.get("original_tokens", 0),
            "tokens_after": decision.get("final_tokens", 0),
        }

    def _build_shared_context(self, input_data: Any, prev_output: str) -> str:
        return f"""System: You are a responder agent.
Previous: {prev_output}
Query: {input_data.get('query', '')}
Role: {self.role}
Instruction: Generate the final response based on all prior agent outputs."""


def create_agents() -> list[BaseAgent]:
    """Create all 5 demo agents."""
    return [
        RetrieverAgent(),
        RerankerAgent(),
        SummarizerAgent(),
        CriticAgent(),
        ResponderAgent(),
    ]