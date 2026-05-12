# Contributing to Apohara ContextForge

Thank you for your interest in contributing to Apohara ContextForge. This is an academic-adjacent project with a published paper (DOI 10.5281/zenodo.20114594), and we welcome external contributions from researchers, practitioners, and developers. All contributions are valued.

## Code of Conduct

Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming and inspiring community for all.

## Development Setup

### Prerequisites

- Python 3.11+ (3.14 is the current development default)
- Git

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/apohara-contextforge.git
cd apohara-contextforge
```

2. Install the project in development mode:
```bash
pip install -e .
```

3. Run the test suite to verify your setup:
```bash
PYTHONPATH=. pytest tests/ -v
```

All tests should pass before you begin making changes.

## Honesty Discipline

This project follows a V6.1 honesty discipline: **every claim in code, docstrings, and documentation must match what the code actually does at runtime.** This is not optional.

- **AUDIT.md is the source of truth.** It contains status flags for all modules and features.
- **Flags reflect runtime state, not intent.** 🟢 = implemented + tested. 🟡 = implemented, no tests. 🟠 = stubbed. 🔴 = broken. Do not introduce flags that claim a status your code does not actually achieve.
- **No fabricated metrics.** Hardcoded benchmark values, demo results, or simulated measurements in production code will be rejected.
- **Documentation must match code.** If a docstring says the function does X but the code does Y, fix the docstring to match the code.

Contributors who introduce inconsistencies between claims and actual behavior will be asked to revise their PR.

## Pre-PR Checklist

Before opening a pull request, verify all of the following:

- [ ] All existing tests pass: `PYTHONPATH=. pytest tests/ -v`
- [ ] Honesty script runs without errors: `scripts/check_honesty.sh`
- [ ] All new code paths have corresponding tests
- [ ] If a new module is introduced, AUDIT.md is updated with its status flag:
  - 🟢 = implemented + tested
  - 🟡 = implemented, no tests yet
  - 🟠 = stubbed (interface only, no implementation)
  - 🔴 = known broken (document the issue in AUDIT.md)
- [ ] No hardcoded metrics, demo values, or fabricated benchmark numbers in production code
- [ ] Type hints are provided for all function signatures
- [ ] Code follows the project's style (match adjacent code in the module)

## DCO: Developer Certificate of Origin

All commits must be signed off with a Developer Certificate of Origin. This affirms that you have the right to submit the contribution under the project's license.

When committing, use the `-s` flag:
```bash
git commit -s -m "Your commit message"
```

This adds a `Signed-off-by: Your Name <your.email@example.com>` trailer to your commit. We do not use a CLA; DCO sign-off is our only requirement.

## Pull Request Process

1. **Fork and branch.** Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make focused changes.** Small, surgical PRs are strongly preferred over large ones. Each PR should have a single, clear purpose.

3. **Push to your fork** and open a PR against the main repository.

4. **PR review.** A maintainer will review your PR, request changes if needed, or merge it.

5. **Merge strategy.** We use squash merges by default to keep a clean history.

## Reporting Bugs

If you find a bug, please open a GitHub issue with:
- A clear, concise description of the bug
- Steps to reproduce it
- Your Python version and platform (output of `python --version` and `uname -s`)
- Relevant test output or error messages
- Expected vs. actual behavior

## Reporting Security Issues

**Do not open a public GitHub issue for security vulnerabilities.** Instead, email the maintainer directly:
- **suarezpm@csnat.unt.edu.ar**

Please include a description of the vulnerability and steps to reproduce it. We will work with you to address the issue before disclosure.

## Questions or Need Help?

If you have questions about the project or need guidance on contributing, feel free to open an issue and ask. We are here to help.

---

**Thank you for contributing!**
