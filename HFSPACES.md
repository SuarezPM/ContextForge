# Deploying the ContextForge demo to HuggingFace Spaces

This repo ships the live-demo Gradio app under `demo/app.py`. The
`hf_spaces/` directory wraps it for HuggingFace Spaces with the YAML
frontmatter, the entrypoint shim, and a scoped `requirements.txt`.
A GH Actions workflow keeps the Space in sync with `main` on every
push.

End state when this is fully wired:

* https://huggingface.co/spaces/SuarezPM/apohara-contextforge — public
  Gradio dashboard, anyone can click "Run with ContextForge" / "Run
  without ContextForge" and see real token-savings + INV-15 firing
  on the same Python code that runs in CI.
* Every push to `main` that touches `demo/app.py`,
  `apohara_context_forge/**`, or `hf_spaces/**` re-syncs the Space
  via [`.github/workflows/sync-hfspaces.yml`](.github/workflows/sync-hfspaces.yml).

---

## 0. One-time setup (≈ 5 min)

1. Sign in at <https://huggingface.co/>.
2. Create a new Space:
   * **Name:** `apohara-contextforge`
   * **Owner:** `SuarezPM` (or your HF username)
   * **SDK:** Gradio
   * **Hardware:** CPU basic (free tier — the demo is CPU-only)
   * **Visibility:** Public
3. Generate a **write** access token at
   <https://huggingface.co/settings/tokens>.
4. Add two GitHub repository settings at
   <https://github.com/SuarezPM/Apohara_Context_Forge/settings>:
   * **Secrets and variables → Actions → Secrets** →
     `HF_TOKEN` = the write token above.
   * **Secrets and variables → Actions → Variables** →
     `HF_SPACE` = `SuarezPM/apohara-contextforge`
     (or whatever owner/name you chose).

That's it. The next push to `main` that touches one of the
trigger paths runs the sync; if you want to deploy right now without
waiting for a code change, go to
**Actions → sync-hfspaces → Run workflow**.

## 1. Local preview (without HF)

```bash
cd hf_spaces
pip install -r requirements.txt
python app.py
# → http://0.0.0.0:7860
```

The shim imports `demo.app` from the parent repo, so all the
dashboard tabs (Live Demo, Real-time Metrics, Benchmark Results,
Architecture + V6 snapshot) work as on a normal `python demo/app.py`
run.

## 2. What gets pushed to the Space

The workflow flattens the repo into a single tree before pushing:

```
/tmp/space/
├── README.md                # from hf_spaces/README.md (with YAML)
├── app.py                   # from hf_spaces/app.py (the shim)
├── requirements.txt         # from hf_spaces/requirements.txt (scoped)
├── demo/                    # copied from repo root
├── apohara_context_forge/   # copied from repo root
├── agents/                  # copied from repo root (if present)
├── AUDIT.md
├── CHANGELOG.md
└── LICENSE
```

The HF Space sees this as a flat repo. The shim's `sys.path` insert
puts both the Space root and the repo parent on the import path, so
`from demo.app import app` resolves whether the Space was assembled
through the workflow or run locally.

## 3. What's NOT on the Space

* **vLLM** — no GPU on free tier. The demo measures token
  deduplication / registry routing / INV-15 firing, which are all
  CPU-only paths. The MI300X performance numbers in the paper come
  from `logs/benchmark_v6_final.txt` (DevCloud ATL1).
* **sentence-transformers**, **llmlingua**, **torch** — kept out of
  the Space's `requirements.txt` to keep cold-start fast. The
  EmbeddingEngine falls back to xorshift pseudo-embeddings when
  these are missing; the demo's token-savings number is unaffected
  (it uses LSH on token IDs, not on dense embeddings).
* **HF private datasets, ssh keys, the GitHub release tarball** —
  the workflow only pushes the public payload listed in §2.

## 4. Operating the Space

* **Logs:** Space settings → "Logs" tab. Gradio prints the same
  output you'd see from `python demo/app.py`.
* **Restart:** Space settings → "Restart" (no code change needed
  for the runtime to refresh).
* **Rollback:** push a revert commit to GitHub `main`; the workflow
  re-syncs automatically.
* **Pause:** Space settings → "Pause Space". Cold-start ~30 s when
  resumed.

## 5. Troubleshooting

| Symptom | Likely cause |
|--------|--------------|
| Space stuck on "Building" > 5 min | First sync is downloading the dep set; check Logs tab. |
| `ModuleNotFoundError: apohara_context_forge` in Space logs | The flatten step in the workflow failed; check Actions run log for the `cp -r apohara_context_forge` step. |
| "Fallback app shown" (the degraded-mode UI from app.py) | A required dep is missing from `hf_spaces/requirements.txt`. The fallback page shows the traceback — copy it into an issue. |
| 401 on push | `HF_TOKEN` secret missing or expired; regenerate at huggingface.co/settings/tokens. |
| Workflow not triggering | Either the path filter didn't match (only `hf_spaces/**`, `demo/app.py`, and `apohara_context_forge/**` trigger), or the `HF_SPACE` variable is unset (workflow has a guard for that). |

## 6. Why a sync workflow instead of HF's "Sync from GitHub" feature

HF's native "Sync from GitHub" syncs the entire repo as-is. That's
unfortunate for us because:

* The Space needs YAML frontmatter on `README.md`. We don't want
  that on the GitHub `README.md` (we tried; see
  commit `6680370 fix: restore original README (revert HF Spaces YAML
  overwrite)`).
* The Space's `requirements.txt` should be different from the parent
  repo's `requirements.txt` (no torch / sentence-transformers on the
  Space).
* The Space wants a top-level `app.py`, but the canonical demo lives
  at `demo/app.py`.

A 30-line sync workflow handles all three concerns cleanly: the
GitHub repo stays GitHub-shaped, the Space stays HF-shaped, and
neither has to compromise.
