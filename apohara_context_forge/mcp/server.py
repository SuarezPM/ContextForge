"""FastAPI MCP-compatible server exposing ContextForge tools."""
import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from apohara_context_forge.config import settings
from apohara_context_forge.metrics.collector import MetricsCollector
from apohara_context_forge.models import (
    CompressionDecision,
    ContextEntry,
    ContextMatch,
    MetricsSnapshot,
)
from apohara_context_forge.registry.context_registry import ContextRegistry

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="ContextForge", version="0.1.0")

# Global instances
registry = ContextRegistry()
metrics = MetricsCollector()


# Request/Response models
class ContextRegistration(BaseModel):
    agent_id: str
    context: str


class OptimizedContextRequest(BaseModel):
    agent_id: str
    context: str


# Tool endpoints
@app.post("/tools/register_context")
async def register_context(registration: ContextRegistration) -> ContextEntry:
    """Register an agent's context in the registry."""
    logger.info(f"Registering context for agent: {registration.agent_id}")
    entry = await registry.register(registration.agent_id, registration.context)
    
    # Update metrics
    await metrics.record_tokens(entry.token_count, entry.token_count)
    active_count = len(await registry.get_all_active())
    await metrics.set_active_agents(active_count)
    
    return entry


@app.post("/tools/get_optimized_context")
async def get_optimized_context(request: OptimizedContextRequest) -> CompressionDecision:
    """Get compression decision for an agent's context."""
    logger.info(f"Optimizing context for agent: {request.agent_id}")
    
    from apohara_context_forge.compression.coordinator import CompressionCoordinator
    coordinator = CompressionCoordinator()
    decision = await coordinator.decide(request.agent_id, request.context)
    
    # Update metrics
    await metrics.record_tokens(decision.original_tokens, decision.final_tokens)
    
    return decision


@app.get("/metrics/snapshot")
async def get_metrics() -> MetricsSnapshot:
    """Get current metrics snapshot."""
    return await metrics.snapshot()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "gpu": "MI300X", "service": "ContextForge"}


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "ContextForge",
        "version": "0.1.0",
        "description": "The shared context compiler for multi-agent LLM systems",
        "docs": "/docs",
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"ContextForge started on {settings.contextforge_host}:{settings.contextforge_port}")
    logger.info(f"vLLM: {settings.vllm_base_url}")
    logger.info(f"Model: {settings.vllm_model}")


# Background metrics loop
async def metrics_loop():
    while True:
        try:
            await asyncio.sleep(30)
            snapshot = await metrics.snapshot()
            logger.info(
                f"Metrics: VRAM={snapshot.vram_used_gb:.1f}GB, "
                f"TTFT={snapshot.ttft_ms:.1f}ms, "
                f"Dedup={snapshot.dedup_rate:.1f}%"
            )
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")