"""HuggingFace Spaces entry point for the ContextForge demo.

HF Spaces expects a top-level ``app.py`` that, when executed,
instantiates and launches a Gradio Blocks app. We import the
canonical app from ``demo/app.py`` so there's a single source of
truth — this file is a 5-line shim, nothing else.

If the import fails (a dep missing on Spaces but present in the
parent repo's dev env), we fall back to a minimal Gradio app that
explains the failure mode and links to the GitHub issues page.
"""
from __future__ import annotations

import os
import sys
import traceback

# Make the parent repo importable. HF Spaces clones this directory
# alone, so we have to reach back up to the repo root that contains
# ``demo/app.py`` and ``apohara_context_forge/``. When deployed via
# the GH Actions sync workflow (.github/workflows/sync-hfspaces.yml)
# the parent directory is at ../, so prepend that to sys.path.
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# In the GH-Actions deployment mode, the sync workflow flattens the
# repo into a single tree where everything lives at the Space root.
# The fallback path here points at the same directory.
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _build_fallback_app(exc: BaseException) -> "object":
    """Tiny single-tab app shown when the real demo fails to import.
    Surfaces the error string + link so a Space visitor isn't faced
    with a blank traceback page."""
    import gradio as gr

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    with gr.Blocks(title="APOHARA · ContextForge (degraded)") as demo:
        gr.Markdown(
            "# APOHARA · ContextForge — temporarily degraded\n\n"
            "The full demo failed to import in this environment.\n"
            "Please file an issue at "
            "[github.com/SuarezPM/Apohara_Context_Forge/issues]"
            "(https://github.com/SuarezPM/Apohara_Context_Forge/issues) "
            "and paste the traceback below."
        )
        gr.Code(value=tb, label="traceback", language="python")
    return demo


try:
    # The real demo. Importing this module instantiates ``app`` as a
    # gr.Blocks at import time — that's the object HF Spaces hosts.
    from demo.app import app  # noqa: F401
except BaseException as exc:  # noqa: BLE001
    app = _build_fallback_app(exc)


if __name__ == "__main__":
    # HF Spaces ignores host/port (their PaaS routes to the Gradio
    # default 7860 anyway), but explicit is friendlier for local
    # `python hf_spaces/app.py` runs.
    app.launch(server_name="0.0.0.0", server_port=7860)
