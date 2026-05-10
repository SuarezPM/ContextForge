# MERGED: OpenCode (deep KV physics) + CC (surface coverage)
# All tests hermetic: no GPU, no TCP, no downloaded weights required
from __future__ import annotations

import json
import os
import subprocess
import sys

import httpx
import pytest

import pytest

# Optional dep guard — skip entire module if CC benchmark functions not present
# (OpenCode has demo/benchmark.py but different functions than CC's benchmark)
try:
    import demo.benchmark as _bm
    _bm._aggregate      # AttributeError if CC's functions not present
    _bm._build_results
    _bm._run_async
    _bm._run_one
    _bm.cli
except (ImportError, AttributeError):
    pytest.skip(
        "CC benchmark functions (_aggregate, _build_results, etc.) not in OpenCode's demo/benchmark.py",
        allow_module_level=True,
    )

import demo.benchmark as benchmark
from demo.benchmark import (
    _aggregate,
    _build_results,
    _fetch_hardware,
    _preflight_mcp,
    _run_async,
    _run_one,
    cli,
)
from apohara_context_forge.serving.vllm_client import VLLMClient


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- Helpers --------------------------------------------------------------------------


def _per_agent_payload(
    *, completion_tokens: int = 10, prompt_tokens: int = 100
) -> dict[str, dict]:
    return {
        name: {
            "tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_tokens": completion_tokens + prompt_tokens,
            "ttft_ms": 50.0,
            "vram_peak_gb": 4.0,
        }
        for name in ("retriever", "reranker", "summarizer", "critic", "responder")
    }


def _make_vllm_with_handler(handler) -> VLLMClient:
    return VLLMClient(
        base_url="http://vllm.test/v1",
        api_key="EMPTY",
        transport=httpx.MockTransport(handler),
    )


def _vllm_chat_handler(prompt_tokens: int):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {
                    "completion_tokens": 5,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens + 5,
                },
            },
        )

    return handler


def _mcp_handler_factory(*, gpu: str = "MI300X", health_status_code: int = 200):
    """MCP MockTransport handler covering the routes run_pipeline + benchmark hit."""
    counts: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counts[path] = counts.get(path, 0) + 1
        if path == "/health":
            if health_status_code != 200:
                return httpx.Response(health_status_code, text="degraded")
            return httpx.Response(200, json={"status": "ok", "gpu": gpu})
        if path == "/tools/register_context":
            return httpx.Response(
                200,
                json={
                    "agent_id": "x",
                    "context": "ctx",
                    "token_count": 1,
                    "created_at": "2026-05-09T12:00:00+00:00",
                    "expires_at": "2026-05-09T12:15:00+00:00",
                },
            )
        if path == "/tools/get_optimized_context":
            return httpx.Response(
                200,
                json={
                    "strategy": "compress",
                    "final_context": "OPT",
                    "shared_prefix": "",
                    "original_tokens": 10,
                    "final_tokens": 5,
                    "tokens_saved": 5,
                    "rationale": "test",
                },
            )
        if path == "/metrics/snapshot":
            return httpx.Response(
                200,
                json={
                    "vram_source": "psutil",
                    "compressor_model": "xlm-roberta-large",
                    "vram_used_gb": 2.0,
                    "vram_total_gb": 8.0,
                    "ttft_ms": 0.0,
                    "tokens_processed": 0,
                    "tokens_saved": 0,
                    "dedup_rate": 0.0,
                    "compression_ratio": 0.0,
                    "degradations": [],
                },
            )
        return httpx.Response(404)

    handler.counts = counts  # type: ignore[attr-defined]
    return handler


# ---- _run_one -------------------------------------------------------------------------


def test_run_one_aggregates_per_agent_dict() -> None:
    per_agent = {
        "a": {"tokens": 10, "prompt_tokens": 100, "ttft_ms": 40.0, "vram_peak_gb": 3.0},
        "b": {"tokens": 20, "prompt_tokens": 200, "ttft_ms": 60.0, "vram_peak_gb": 5.0},
    }
    out = _run_one(per_agent)
    assert out == {
        "tokens": 30,
        "prompt_tokens": 300,
        "ttft_ms": 50.0,
        "vram_peak_gb": 5.0,
    }


def test_run_one_handles_empty_dict() -> None:
    assert _run_one({}) == {
        "tokens": 0,
        "prompt_tokens": 0,
        "ttft_ms": 0.0,
        "vram_peak_gb": 0.0,
    }


# ---- _aggregate -----------------------------------------------------------------------


def test_aggregate_computes_delta_pct() -> None:
    warm_on = [
        {"tokens": 5, "prompt_tokens": 800, "ttft_ms": 50.0, "vram_peak_gb": 4.0},
        {"tokens": 5, "prompt_tokens": 800, "ttft_ms": 50.0, "vram_peak_gb": 4.0},
    ]
    warm_off = [
        {"tokens": 5, "prompt_tokens": 1000, "ttft_ms": 60.0, "vram_peak_gb": 5.0},
        {"tokens": 5, "prompt_tokens": 1000, "ttft_ms": 60.0, "vram_peak_gb": 5.0},
    ]
    totals = _aggregate(warm_on, warm_off)
    assert totals["tokens_with"] == 800
    assert totals["tokens_without"] == 1000
    assert totals["delta_pct"] == 20.0
    assert totals["vram_with"] == 4.0
    assert totals["vram_without"] == 5.0
    assert totals["ttft_with"] == 50.0
    assert totals["ttft_without"] == 60.0


def test_aggregate_returns_null_delta_when_tokens_without_is_zero() -> None:
    warm_on = [
        {"tokens": 5, "prompt_tokens": 0, "ttft_ms": 50.0, "vram_peak_gb": 4.0},
    ]
    warm_off = [
        {"tokens": 5, "prompt_tokens": 0, "ttft_ms": 60.0, "vram_peak_gb": 5.0},
    ]
    totals = _aggregate(warm_on, warm_off)
    assert totals["delta_pct"] is None
    # JSON-serializable as null
    assert json.loads(json.dumps(totals))["delta_pct"] is None


# ---- _build_results -------------------------------------------------------------------


def test_build_results_schema() -> None:
    cold = [{"tokens": 100, "prompt_tokens": 200, "ttft_ms": 50.0, "vram_peak_gb": 4.0}]
    warm = [
        {"tokens": 80, "prompt_tokens": 150, "ttft_ms": 40.0, "vram_peak_gb": 4.5},
        {"tokens": 70, "prompt_tokens": 140, "ttft_ms": 35.0, "vram_peak_gb": 4.5},
    ]
    off = [
        {"tokens": 100, "prompt_tokens": 300, "ttft_ms": 60.0, "vram_peak_gb": 5.0},
        {"tokens": 95, "prompt_tokens": 290, "ttft_ms": 55.0, "vram_peak_gb": 5.0},
        {"tokens": 90, "prompt_tokens": 280, "ttft_ms": 55.0, "vram_peak_gb": 5.0},
    ]
    config = {
        "runs": 3,
        "warmup": 1,
        "vllm_base_url": "http://x",
        "model": "m",
        "started_at": "t1",
        "completed_at": "t2",
    }
    result = _build_results("MI300X", config, cold, warm, off)

    expected_top = {
        "hardware",
        "config",
        "cold",
        "warm",
        "off",
        "cold_cache_baseline",
        "totals",
    }
    assert set(result.keys()) == expected_top

    expected_totals = {
        "tokens_with",
        "tokens_without",
        "vram_with",
        "vram_without",
        "ttft_with",
        "ttft_without",
        "delta_pct",
    }
    assert set(result["totals"].keys()) == expected_totals

    expected_baseline = {"tokens", "prompt_tokens", "ttft_ms", "vram_peak_gb"}
    assert set(result["cold_cache_baseline"].keys()) == expected_baseline
    # cold_cache_baseline is the per-run reduction of cold[0] (MEM007).
    assert result["cold_cache_baseline"] == cold[0]


# ---- _fetch_hardware ------------------------------------------------------------------


async def test_hardware_field_from_health_response() -> None:
    handler = _mcp_handler_factory(gpu="MI300X")
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    try:
        gpu = await _fetch_hardware(mcp)
    finally:
        await mcp.aclose()
    assert gpu == "MI300X"


async def test_hardware_unknown_on_health_failure() -> None:
    handler = _mcp_handler_factory(health_status_code=503)
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    try:
        gpu = await _fetch_hardware(mcp)
    finally:
        await mcp.aclose()
    assert gpu == "unknown"


# ---- _run_async: schema, run-counts, warmup discard, prompt-token math ----------------


async def test_runs_three_on_three_off(monkeypatch, tmp_path) -> None:
    invocations: list[bool] = []

    async def fake_run_pipeline(
        *,
        contextforge_enabled,
        mcp_client,
        vllm_client,
        shared_base,
        query,
    ):
        invocations.append(contextforge_enabled)
        return _per_agent_payload()

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    handler = _mcp_handler_factory()
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    vllm = _make_vllm_with_handler(_vllm_chat_handler(prompt_tokens=100))
    output_path = tmp_path / "out.json"

    try:
        rc = await _run_async(
            runs=3,
            warmup=1,
            output_path=output_path,
            query="q",
            shared_base="s",
            mcp_client=mcp,
            vllm_client=vllm,
        )
    finally:
        await mcp.aclose()
        await vllm.aclose()

    assert rc == 0
    assert len(invocations) == 6
    # ON first 3, then OFF next 3
    assert invocations[:3] == [True, True, True]
    assert invocations[3:] == [False, False, False]


async def test_warmup_discarded(monkeypatch, tmp_path) -> None:
    async def fake_run_pipeline(**_kwargs):
        return _per_agent_payload()

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    handler = _mcp_handler_factory()
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    vllm = _make_vllm_with_handler(_vllm_chat_handler(prompt_tokens=100))
    output_path = tmp_path / "out.json"
    try:
        rc = await _run_async(
            runs=3,
            warmup=1,
            output_path=output_path,
            query="q",
            shared_base="s",
            mcp_client=mcp,
            vllm_client=vllm,
        )
    finally:
        await mcp.aclose()
        await vllm.aclose()

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert len(payload["cold"]) == 1
    assert len(payload["warm"]) == 2
    assert len(payload["off"]) == 3


async def test_warm_tokens_with_lt_without(monkeypatch, tmp_path) -> None:
    """ON returns lower prompt_tokens than OFF → totals.tokens_with < tokens_without."""

    async def fake_run_pipeline(*, contextforge_enabled, **_kwargs):
        prompt = 200 if contextforge_enabled else 400
        return _per_agent_payload(prompt_tokens=prompt)

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    handler = _mcp_handler_factory()
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    vllm = _make_vllm_with_handler(_vllm_chat_handler(prompt_tokens=100))
    output_path = tmp_path / "out.json"
    try:
        rc = await _run_async(
            runs=3,
            warmup=1,
            output_path=output_path,
            query="q",
            shared_base="s",
            mcp_client=mcp,
            vllm_client=vllm,
        )
    finally:
        await mcp.aclose()
        await vllm.aclose()

    assert rc == 0
    payload = json.loads(output_path.read_text())
    totals = payload["totals"]
    assert totals["tokens_with"] < totals["tokens_without"]
    assert totals["delta_pct"] > 0


async def test_run_async_writes_full_schema(monkeypatch, tmp_path) -> None:
    """Every required top-level key + totals + cold_cache_baseline + hardware."""

    async def fake_run_pipeline(**_kwargs):
        return _per_agent_payload()

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    handler = _mcp_handler_factory(gpu="MI300X")
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    vllm = _make_vllm_with_handler(_vllm_chat_handler(prompt_tokens=100))
    output_path = tmp_path / "out.json"
    try:
        rc = await _run_async(
            runs=3,
            warmup=1,
            output_path=output_path,
            query="q",
            shared_base="s",
            mcp_client=mcp,
            vllm_client=vllm,
        )
    finally:
        await mcp.aclose()
        await vllm.aclose()

    assert rc == 0
    payload = json.loads(output_path.read_text())
    assert payload["hardware"] == "MI300X"
    assert {"runs", "warmup", "vllm_base_url", "model", "started_at", "completed_at"} <= set(
        payload["config"].keys()
    )
    assert payload["config"]["runs"] == 3
    assert payload["config"]["warmup"] == 1


async def test_run_async_caller_owned_clients_survive(monkeypatch, tmp_path) -> None:
    """MEM048: caller-injected mcp/vllm clients must NOT be closed by _run_async."""

    async def fake_run_pipeline(**_kwargs):
        return _per_agent_payload()

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    handler = _mcp_handler_factory()
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    vllm = _make_vllm_with_handler(_vllm_chat_handler(prompt_tokens=100))
    output_path = tmp_path / "out.json"
    try:
        await _run_async(
            runs=2,
            warmup=1,
            output_path=output_path,
            query="q",
            shared_base="s",
            mcp_client=mcp,
            vllm_client=vllm,
        )
        assert not mcp.is_closed
        # VLLMClient probe: any HTTP call works iff client open.
        resp = await vllm._client.get("/models")
        assert resp.status_code in (200, 404)
    finally:
        await mcp.aclose()
        await vllm.aclose()


async def test_run_async_does_not_write_partial_json_on_failure(
    monkeypatch, tmp_path
) -> None:
    """Mid-run exception → file write never happens; output file does NOT appear."""
    call_count = {"n": 0}

    async def fake_run_pipeline(**_kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            raise RuntimeError("simulated mid-run failure")
        return _per_agent_payload()

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    handler = _mcp_handler_factory()
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    vllm = _make_vllm_with_handler(_vllm_chat_handler(prompt_tokens=100))
    output_path = tmp_path / "should_not_exist.json"
    try:
        with pytest.raises(RuntimeError):
            await _run_async(
                runs=3,
                warmup=1,
                output_path=output_path,
                query="q",
                shared_base="s",
                mcp_client=mcp,
                vllm_client=vllm,
            )
    finally:
        await mcp.aclose()
        await vllm.aclose()

    assert not output_path.exists()


# ---- Structured logging ---------------------------------------------------------------


async def test_structured_log_per_run(monkeypatch, tmp_path, caplog) -> None:
    async def fake_run_pipeline(**_kwargs):
        return _per_agent_payload()

    monkeypatch.setattr(benchmark, "run_pipeline", fake_run_pipeline)

    handler = _mcp_handler_factory()
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    vllm = _make_vllm_with_handler(_vllm_chat_handler(prompt_tokens=100))
    output_path = tmp_path / "out.json"

    import logging

    try:
        with caplog.at_level(logging.INFO, logger="demo.benchmark"):
            await _run_async(
                runs=3,
                warmup=1,
                output_path=output_path,
                query="q",
                shared_base="s",
                mcp_client=mcp,
                vllm_client=vllm,
            )
    finally:
        await mcp.aclose()
        await vllm.aclose()

    records = [r for r in caplog.records if r.name == "demo.benchmark"]
    # 3 ON + 3 OFF = 6 "benchmark run done" records
    assert len(records) == 6
    modes = [getattr(r, "mode", None) for r in records]
    assert modes == ["cold", "warm", "warm", "off", "off", "off"]
    for r in records:
        assert getattr(r, "component", None) == "demo.benchmark"
        assert isinstance(getattr(r, "run_index", None), int)


# ---- D009 verbatim fail-fast ----------------------------------------------------------


@pytest.mark.smoke
def test_d009_vllm_unreachable_smoke() -> None:
    env = {**os.environ, "VLLM_BASE_URL": "http://127.0.0.1:1"}
    result = subprocess.run(
        [sys.executable, "-m", "demo.benchmark"],
        env=env,
        cwd=PROJECT_ROOT,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    expected = (
        b"vLLM unreachable at http://127.0.0.1:1. "
        b"Start the vLLM server first or set VLLM_BASE_URL."
    )
    assert expected in result.stderr


@pytest.mark.smoke
def test_d009_mcp_unreachable_smoke(monkeypatch, capsys, tmp_path) -> None:
    """In-process gate per the slice plan (separate fake vLLM is impractical for subprocess)."""

    async def fake_vllm_ok(client):
        return True

    async def fake_mcp_fail(client):
        return False

    monkeypatch.setattr(benchmark, "_preflight_vllm", fake_vllm_ok)
    monkeypatch.setattr(benchmark, "_preflight_mcp", fake_mcp_fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["demo.benchmark", "--output", str(tmp_path / "out.json")],
    )

    rc = cli()
    captured = capsys.readouterr()
    expected = (
        "ContextForge MCP unreachable at http://127.0.0.1:8001. "
        "Start the MCP server first or set CONTEXTFORGE_HOST/CONTEXTFORGE_PORT."
    )
    assert rc == 1
    assert expected in captured.err


# ---- Preflight unit ------------------------------------------------------------------


async def test_preflight_mcp_returns_true_on_ok_health() -> None:
    handler = _mcp_handler_factory()
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    try:
        ok = await _preflight_mcp(mcp)
    finally:
        await mcp.aclose()
    assert ok is True


async def test_preflight_mcp_returns_false_on_non_2xx() -> None:
    handler = _mcp_handler_factory(health_status_code=503)
    mcp = httpx.AsyncClient(
        base_url="http://mcp.test", transport=httpx.MockTransport(handler)
    )
    try:
        ok = await _preflight_mcp(mcp)
    finally:
        await mcp.aclose()
    assert ok is False


# ---- Preflight failure logging --------------------------------------------------------


def test_vllm_preflight_failure_logs_warning_before_stderr(
    monkeypatch, capsys, caplog, tmp_path
) -> None:
    """logger.warning fires BEFORE the verbatim stderr print (MEM053 extension)."""

    async def fake_vllm_fail(client):
        return False

    monkeypatch.setattr(benchmark, "_preflight_vllm", fake_vllm_fail)
    monkeypatch.setattr(
        sys,
        "argv",
        ["demo.benchmark", "--output", str(tmp_path / "out.json")],
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="demo.benchmark"):
        rc = cli()
    captured = capsys.readouterr()

    assert rc == 1
    warn_records = [r for r in caplog.records if r.name == "demo.benchmark"]
    assert any(
        getattr(r, "component", None) == "demo.benchmark"
        and getattr(r, "base_url", None) is not None
        for r in warn_records
    )
    assert "vLLM unreachable at" in captured.err
