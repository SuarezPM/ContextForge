# Sprint 4 follow-ups (V7.0.0 final blockers / nice-to-haves)

## Blockers for V7.0.0 final
- [ ] arXiv submission (paper v2.0 PDF ready in V7.0.0-rc.1, needs user endorsement dance)
- [ ] Zenodo deposit V7.0.0-rc.1 release (gated on arXiv ID for paper cross-reference)

## V7.0.0 nice-to-haves (could ship in V7.0.0-rc.2)
- [ ] Per-nibble independent scales codec rewrite (would reclaim FWHT benefit + literature 3.97x at cost of ~0.5x storage)
- [ ] vLLM end-to-end test on MI300X (would need actual model load — Sprint 4 Wave B work)
- [ ] V6.2 adversarial benchmark with GPU-aware queueing (currently CPU sim)
- [ ] Multi-worker LMCache cross-hit test (gated on W2 non-CUDA adapter shipping)

## V8 work (deferred)
- [ ] Plugin marketplace SDK (still LOW leverage; plugin count = 1)
- [ ] @sha256: digest pinning for production operator deployment
- [ ] govulncheck CI run + dependency refresh
