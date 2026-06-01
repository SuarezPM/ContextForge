"""Confirms vLLMClient forwards cache_salt as an extra request-body field.

ROMY Fase 1 relies on vLLM Automatic Prefix Caching, which keys prefix blocks
by ``cache_salt``. The salt reaches vLLM as a top-level field of the JSON body.
vLLMClient already merges arbitrary ``**kwargs`` into the payload (see
``vllm_client.py`` ``complete``/``chat``), so no client change is needed — this
test pins that behaviour so a future refactor can't silently drop the salt.

Hermetic: uses httpx.MockTransport (no TCP, no server), matching the repo's
existing test style in tests/test_benchmark.py.
"""
from __future__ import annotations

import json

import httpx
import pytest

from apohara_context_forge.serving.vllm_client import vLLMClient


def _capturing_client(captured: dict) -> httpx.AsyncClient:
    """An AsyncClient whose MockTransport records the last request body."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"text": "ok"}], "id": "cmpl-test"},
        )

    return httpx.AsyncClient(
        base_url="http://vllm.test",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_complete_forwards_cache_salt():
    captured: dict = {}
    client = vLLMClient()
    # Inject the mock transport directly (client lazily builds its own client;
    # we pre-seed it so no real socket is opened).
    client._client = _capturing_client(captured)

    await client.complete(prompt="hello", cache_salt="shared:abc123")
    await client.aclose()

    assert captured["path"] == "/v1/completions"
    # cache_salt is a TOP-LEVEL body field, exactly where vLLM APC reads it.
    assert captured["body"]["cache_salt"] == "shared:abc123"
    assert captured["body"]["prompt"] == "hello"


@pytest.mark.asyncio
async def test_chat_forwards_cache_salt():
    captured: dict = {}
    client = vLLMClient()
    client._client = _capturing_client(captured)

    await client.chat(
        messages=[{"role": "user", "content": "hi"}],
        cache_salt="iso:deadbeef",
    )
    await client.aclose()

    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["cache_salt"] == "iso:deadbeef"


@pytest.mark.asyncio
async def test_no_cache_salt_when_not_passed():
    """Absence is honest too: no salt kwarg => no cache_salt in the body."""
    captured: dict = {}
    client = vLLMClient()
    client._client = _capturing_client(captured)

    await client.complete(prompt="hello")
    await client.aclose()

    assert "cache_salt" not in captured["body"]
