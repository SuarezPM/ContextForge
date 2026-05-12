# Publishing apohara-vllm-plugin

Two-stage release: **TestPyPI** first (validates the upload, the
classifiers, and the entry-point), then **PyPI** for real. The Apohara
repo has a GitHub Actions workflow that automates step 4 below
([`.github/workflows/release-plugin.yml`](../../.github/workflows/release-plugin.yml));
this doc covers both the automated and the manual path.

---

## 0. One-time prerequisites

### PyPI / TestPyPI accounts

1. Create accounts at <https://pypi.org/account/register/> and
   <https://test.pypi.org/account/register/>.
2. Enable 2FA on both.
3. Generate **API tokens** (Account Settings → API tokens):
   * One scoped to project `apohara-vllm-plugin` on TestPyPI.
   * One scoped to project `apohara-vllm-plugin` on PyPI (create the
     project on PyPI *after* the first successful upload to claim the
     name; until then use a "Account" scope token).

### Trusted publishing (recommended over API tokens)

PyPI now supports OIDC-based [trusted publishers](https://docs.pypi.org/trusted-publishers/),
which removes the need to store long-lived tokens in GitHub secrets.

1. On PyPI / TestPyPI → project settings → **Publishing** →
   **Add trusted publisher** → GitHub Actions.
2. Owner: `SuarezPM`. Repository: `Apohara_Context_Forge`.
3. Workflow filename: `release-plugin.yml`.
4. Environment name: `pypi` (matches the workflow's `environment:`
   block).

That's it — the workflow uses `pypa/gh-action-pypi-publish@release/v1`
with no token. No secrets needed.

If you prefer API tokens, store them as repo secrets:

  * `TEST_PYPI_API_TOKEN`
  * `PYPI_API_TOKEN`

…and update the workflow to pass `password: ${{ secrets.PYPI_API_TOKEN }}`.

---

## 1. Build the wheel + sdist locally

```bash
cd pypi/apohara-vllm-plugin
rm -rf dist build *.egg-info src/*.egg-info
python -m build           # produces dist/*.whl and dist/*.tar.gz
twine check dist/*        # validates metadata, README rendering, classifiers
```

Expected output:

```
dist/apohara_vllm_plugin-0.1.0-py3-none-any.whl
dist/apohara_vllm_plugin-0.1.0.tar.gz
```

`twine check` must report `PASSED` for both files. If the README
fails to render, that's typically a Markdown / RST mismatch — the
`pyproject.toml` declares `readme = "README.md"` so it must be
GitHub-flavoured Markdown.

## 2. Smoke-test the wheel in a clean venv

```bash
python -m venv /tmp/venv_pypi
source /tmp/venv_pypi/bin/activate
pip install --no-deps -e ../../                              # apohara-context-forge editable
pip install --no-deps dist/apohara_vllm_plugin-0.1.0-*.whl   # the wheel under test
pip install numpy prometheus-client pydantic pydantic-settings pytest
python -c "
import importlib.metadata as md
eps = list(md.entry_points(group='vllm.general_plugins'))
print([(e.name, e.value) for e in eps])
assert any(e.name == 'apohara_contextforge' for e in eps)
"
pytest tests/
```

The smoke test must print:

```
[('apohara_contextforge', 'apohara_vllm_plugin:register')]
```

and `pytest tests/` must report **6 passed, 1 skipped** (the
`test_vllm_integration_smoke` skip is correct unless vLLM is also
installed in the venv).

## 3. Upload to TestPyPI

```bash
pip install -U twine
twine upload --repository testpypi dist/*
```

Verify the project page renders correctly at
<https://test.pypi.org/project/apohara-vllm-plugin/0.1.0/>. Check:

* The README displays with badges + headings.
* "Project links" shows DOI, AUDIT, CHANGELOG.
* Entry-point section lists `vllm.general_plugins → apohara_contextforge`.
* Classifiers include Apache-2.0 + Python 3.11/3.12.

Test-install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            apohara-vllm-plugin
python -c "from apohara_vllm_plugin import register; print(register().is_initialized())"
```

The `--extra-index-url` is required because TestPyPI does not host
mirror copies of standard PyPI deps (numpy, etc.).

## 4. Upload to PyPI (real)

```bash
twine upload dist/*
```

Or via GitHub Actions: tag a release `vllm-plugin-v0.1.0`, which the
workflow `.github/workflows/release-plugin.yml` watches.

Verify:

* <https://pypi.org/project/apohara-vllm-plugin/0.1.0/>
* `pip install apohara-vllm-plugin` works on a third machine.

## 5. Post-release checks

* Create a GitHub Release at
  <https://github.com/SuarezPM/Apohara_Context_Forge/releases/new> with
  tag `vllm-plugin-v0.1.0`, attaching the `.whl` and `.tar.gz` as
  release assets (the workflow does this automatically).
* Tag the main repo with the matching commit:
  ```bash
  git tag -a vllm-plugin-v0.1.0 -m "apohara-vllm-plugin 0.1.0 (V6.x #1)"
  git push origin vllm-plugin-v0.1.0
  ```
* Update the project root README badge row with a `pip install` badge:
  ```
  [![PyPI](https://img.shields.io/pypi/v/apohara-vllm-plugin?logo=pypi&logoColor=white)](https://pypi.org/project/apohara-vllm-plugin/)
  ```

## 6. Yank procedure (only if a release is broken)

```bash
twine upload --skip-existing dist/*    # cannot replace files
# Yank via the PyPI web UI: project → release page → "Yank release"
```

A yanked release stays installable via explicit version pin but
disappears from `pip install apohara-vllm-plugin` (the implicit
latest). Do this if 0.1.0 has a serious bug; otherwise ship 0.1.1.

## 7. Version bump checklist (for 0.1.1, 0.2.0, etc.)

* Bump `version = "0.1.0"` in `pyproject.toml`.
* Bump `__version__ = "0.1.0"` in
  `src/apohara_vllm_plugin/__init__.py`.
* Add a `CHANGELOG` section if you have one in the plugin dir (we
  currently delegate to the project-root `CHANGELOG.md`).
* `python -m build` + `twine check` + venv smoke test (step 2).
* Tag `vllm-plugin-vX.Y.Z` and let the workflow do the upload.

---

## Why we publish as a thin shim

The plugin's behaviour lives in
[`apohara_context_forge.serving.atom_plugin`](../../apohara_context_forge/serving/atom_plugin.py)
(the in-tree implementation tested by
[`tests/test_atom_plugin.py`](../../tests/test_atom_plugin.py)
— 19 tests, all passing as of V6.1). The `apohara-vllm-plugin`
distribution exists for one reason: to declare the
`vllm.general_plugins` entry point on a package that vLLM users
discover via `pip search`. Putting that declaration on the heavy
`apohara-context-forge` package would force every vLLM user to take a
full ContextForge dep (torch, faiss, etc.) just to register the entry
point. The shim package keeps the declaration where vLLM users look,
and the shim itself only depends on `apohara-context-forge` so a real
deploy still pulls in the heavy machinery on demand.
