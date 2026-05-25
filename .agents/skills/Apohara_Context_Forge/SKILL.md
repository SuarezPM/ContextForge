```markdown
# Apohara_Context_Forge Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns and conventions used in the Apohara_Context_Forge Python codebase. You'll learn about the project's file organization, import/export styles, commit message conventions, and how to write and run tests. This guide is designed to help contributors maintain consistency and quality across the codebase.

## Coding Conventions

### File Naming
- Use **snake_case** for all file and module names.
  - Example: `context_manager.py`, `data_loader.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import parse_context
    from ..models import ContextModel
    ```

### Export Style
- Use **named exports** (explicitly define what is exported from a module).
  - Example:
    ```python
    __all__ = ["ContextManager", "parse_context"]
    ```

### Commit Messages
- Follow the **conventional commit** style.
- Use prefixes such as `test` for test-related commits.
- Keep commit messages concise (average ~52 characters).
  - Example:
    ```
    test: add unit tests for context parsing
    ```

## Workflows

### Adding a New Feature
**Trigger:** When implementing a new functionality.
**Command:** `/add-feature`

1. Create a new Python file using snake_case.
2. Use relative imports for internal modules.
3. Define `__all__` for explicit exports.
4. Write clear, conventional commit messages.
5. Add or update tests as needed.

### Writing and Running Tests
**Trigger:** When verifying code correctness.
**Command:** `/run-tests`

1. Create test files following the `*.test.ts` pattern (note: legacy or cross-language pattern).
2. Implement test cases for new or modified code.
3. Use the project's preferred test runner (framework is unknown; check project docs or ask maintainers).
4. Run the tests and ensure all pass before committing.

## Testing Patterns

- Test files should match the `*.test.ts` pattern, though the project is in Python (this may indicate cross-language testing or legacy patterns).
- The specific testing framework is unknown; check with maintainers or inspect existing test files for guidance.
- Example test file name: `context_manager.test.ts`
- Example (Python-style test):
  ```python
  def test_parse_context():
      assert parse_context("input") == "expected_output"
  ```

## Commands
| Command        | Purpose                                      |
|----------------|----------------------------------------------|
| /add-feature   | Start workflow for adding a new feature      |
| /run-tests     | Run all tests in the codebase                |
```
