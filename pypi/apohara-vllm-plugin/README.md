# apohara-vllm-plugin

Multi-agent KV-cache coordination as a [vLLM V1](https://vllm.ai)
plugin. Drop it next to vLLM and it self-registers through the
`vllm.general_plugins` entry-point group: no patching, no fork.

```bash
pip install apohara-vllm-plugin
```

The plugin's job inside vLLM is:

1. **Anchor-aware KV-block routing** via SimHash LSH lookup against the
   ContextForge registry (cross-agent block reuse).
2. **RotateKV pre-RoPE INT4 quantization hooks** (INVARIANT 10:
   pre-RoPE only).
3. **JCR Safety Gate (INV-15) enforcement** — judge / critic agents
   with `JCR risk > 0.7` are forced into dense prefill, bypassing the
   shared cache. See [arXiv:2601.08343](https://arxiv.org/abs/2601.08343).
4. **Honest metrics** — every flag in the hook's return dict reflects
   state (what actually ran), not intent (what the config asked for).

This is the thin published shim over the in-tree implementation at
[`apohara_context_forge.serving.atom_plugin`](https://github.com/SuarezPM/Apohara_Context_Forge/blob/main/apohara_context_forge/serving/atom_plugin.py).

## Quick usage

### Inside vLLM (automatic)

vLLM walks `vllm.general_plugins` at worker startup. No code change:

```bash
pip install vllm apohara-vllm-plugin
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-235B-A22B
```

You should see in the vLLM startup log:

```
ATOM plugin initialised: worker=… deps={…}
ATOM plugin: vLLM platform hooks registered
```

### Manually (for tests / inspection)

```python
from apohara_vllm_plugin import register

plugin = register()
assert plugin.is_initialized()
print(plugin.get_stats())
```

The plugin is constructible without vLLM installed; the kernel-level
hook installation is silently skipped in that case.

### Wiring real ContextForge dependencies

By default the plugin runs as a no-op telemetry surface (every flag in
the metadata dict reports `False` / `None` honestly). Inject the real
subsystems through `vLLMAtomPlugin(...)`:

```python
from apohara_vllm_plugin import vLLMAtomPlugin, ATOMConfig
from apohara_context_forge.quantization.rotate_kv import (
    RotateKVConfig, RotateKVQuantizer,
)
from apohara_context_forge.dedup.lsh_engine import LSHTokenMatcher
from apohara_context_forge.safety.jcr_gate import JCRSafetyGate
from apohara_context_forge.metrics.collector import MetricsCollector

plugin = vLLMAtomPlugin(
    ATOMConfig(),
    quantizer=RotateKVQuantizer(RotateKVConfig()),
    lsh_matcher=LSHTokenMatcher(),
    jcr_gate=JCRSafetyGate(),
    metrics=MetricsCollector(),
)
plugin.initialize("worker_0", vllm_config={})
```

Pass that plugin's `pre_attention_hook` / `post_attention_hook` to
your custom vLLM platform if you're not relying on the entry-point
auto-discovery.

## Honest semantics

V6.1+ flags in the pre-attention hook's return dict:

| Flag                        | True iff                                                                 |
|----------------------------|---------------------------------------------------------------------------|
| `quantization_attempted`    | `enable_quantization=True` *and* a quantizer was wired                    |
| `quantization_applied`      | a quantizer was wired *and* it actually executed without raising         |
| `quantized` *(alias)*       | same as `quantization_applied` — kept for back-compat                    |
| `pre_rope`                  | always `True` — INV-10: this hook never operates on post-RoPE tensors    |
| `anchor_match`              | `None` if no LSH matcher wired; else lookup descriptor                   |
| `jcr_dense`                 | `True` iff JCR Safety Gate fired INV-15 for this call                    |

Returning `True` when nothing happened is the pattern we're explicitly
fixing in V6.1 — see the project root [`AUDIT.md`](https://github.com/SuarezPM/Apohara_Context_Forge/blob/main/AUDIT.md).

## Citation

If this plugin or the underlying mechanisms help your work, please cite:

```bibtex
@misc{contextforge,
  author    = {Suarez, Pablo M.},
  title     = {{ContextForge: A Unified KV-Cache Coordination Layer
                for Multi-Agent LLM Pipelines on AMD Instinct MI300X}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20114594},
  url       = {https://doi.org/10.5281/zenodo.20114594}
}
```

## License

Apache-2.0.
