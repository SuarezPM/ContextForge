"""Entry point - starts ContextForge server and metrics collector."""
import asyncio
import logging
import uvicorn

from apohara_context_forge.config import settings
from apohara_context_forge.metrics.collector import MetricsCollector
from apohara_context_forge.mcp.server import app, metrics_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Start ContextForge server."""
    logger.info("Starting ContextForge...")
    logger.info(f"Host: {settings.contextforge_host}:{settings.contextforge_port}")
    logger.info(f"vLLM: {settings.vllm_base_url}")
    logger.info(f"Model: {settings.vllm_model}")

    # Start background metrics collector
    metrics_task = asyncio.create_task(metrics_loop())
    
    try:
        config = uvicorn.Config(
            app,
            host=settings.contextforge_host,
            port=settings.contextforge_port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    finally:
        metrics_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())