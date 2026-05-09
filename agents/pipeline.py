"""Pipeline orchestrator - runs 5 agents, collects metrics."""
import asyncio
import logging
import time
from typing import Any

from agents.demo_agents import create_agents

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates 5-agent pipeline with metrics collection."""

    def __init__(self, enable_contextforge: bool = True):
        self.agents = create_agents()
        self.enable_contextforge = enable_contextforge
        self.metrics = {
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "agent_ttft_ms": [],
            "strategies_used": {},
        }

    async def run(self, query: str) -> dict[str, Any]:
        """Run the full pipeline for a query."""
        logger.info(f"Starting pipeline for query: {query[:50]}...")
        
        input_data = {"query": query}
        pipeline_output = {}
        start_time = time.time()

        for i, agent in enumerate(self.agents):
            agent_start = time.time()
            result = await agent.process(input_data)
            agent_duration = (time.time() - agent_start) * 1000

            pipeline_output[f"{agent.agent_id}_output"] = result["result"]
            pipeline_output[f"{agent.agent_id}_metrics"] = {
                "ttft_ms": agent_duration,
                "strategy": result["strategy"],
                "tokens_before": result["tokens_before"],
                "tokens_after": result["tokens_after"],
            }

            self.metrics["total_tokens_before"] += result["tokens_before"]
            self.metrics["total_tokens_after"] += result["tokens_after"]
            self.metrics["agent_ttft_ms"].append(agent_duration)
            self.metrics["strategies_used"][result["strategy"]] = \
                self.metrics["strategies_used"].get(result["strategy"], 0) + 1

            input_data[f"{agent.agent_id}_output"] = result["result"]

        total_duration = (time.time() - start_time) * 1000

        return {
            "query": query,
            "final_output": pipeline_output.get("responder_output", ""),
            "pipeline_duration_ms": total_duration,
            "agent_metrics": pipeline_output,
            "summary": {
                "total_tokens_before": self.metrics["total_tokens_before"],
                "total_tokens_after": self.metrics["total_tokens_after"],
                "avg_ttft_ms": sum(self.metrics["agent_ttft_ms"]) / len(self.metrics["agent_ttft_ms"]),
                "strategies": self.metrics["strategies_used"],
                "token_savings_pct": (
                    (self.metrics["total_tokens_before"] - self.metrics["total_tokens_after"])
                    / self.metrics["total_tokens_before"] * 100
                    if self.metrics["total_tokens_before"] > 0 else 0
                ),
            },
        }


async def run_pipeline_dry():
    """Dry run - prints agent plan without execution."""
    agents = create_agents()
    print("\n=== ContextForge Pipeline - Dry Run ===")
    print(f"Total agents: {len(agents)}\n")
    for i, agent in enumerate(agents, 1):
        print(f"{i}. {agent.agent_id.upper()} ({agent.role})")
    print("\nPipeline flow:")
    print("  Query -> Retriever -> Reranker -> Summarizer -> Critic -> Responder")
    print("\nEach agent will:")
    print("  1. Register context with ContextForge")
    print("  2. Get optimized context (compression decision)")
    print("  3. Use optimized context for processing")
    print("  4. Return result with metrics\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ContextForge Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--query", default="What is machine learning?", help="Query to process")
    args = parser.parse_args()

    if args.dry_run:
        asyncio.run(run_pipeline_dry())
    else:
        pipeline = Pipeline()
        result = asyncio.run(pipeline.run(args.query))
        print(f"\n=== Pipeline Result ===")
        print(f"Token savings: {result['summary']['token_savings_pct']:.1f}%")
        print(f"Avg TTFT: {result['summary']['avg_ttft_ms']:.1f}ms")
        print(f"Strategies: {result['summary']['strategies']}")