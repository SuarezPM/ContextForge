```markdown
# Apohara_Context_Forge Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches the core development patterns used in the Apohara_Context_Forge Python codebase. You'll learn about file organization, code style, import/export conventions, and how to write and structure tests. These practices ensure consistency and maintainability when contributing to or extending the repository.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `context_manager.py`, `data_loader.py`

### Import Style
- Use **relative imports** within the package.
  - Example:
    ```python
    from .utils import parse_context
    from .models import ContextModel
    ```

### Export Style
- Use **named exports**; explicitly define what is exported from modules.
  - Example:
    ```python
    __all__ = ["ContextManager", "parse_context"]
    ```

### Commit Patterns
- Commit messages are freeform, typically around 45 characters.
- No strict prefixing required, but keep messages concise and descriptive.

## Workflows

### Adding a New Module
**Trigger:** When you need to introduce new functionality.
**Command:** `/add-module`

1. Create a new Python file using snake_case (e.g., `my_feature.py`).
2. Implement the required classes or functions.
3. Use relative imports for any internal dependencies.
4. Define `__all__` to specify exported symbols.
5. Add or update corresponding test files (see Testing Patterns).

### Refactoring Existing Code
**Trigger:** When improving or restructuring code.
**Command:** `/refactor`

1. Identify the target file(s) and functionality.
2. Update code following snake_case and relative import conventions.
3. Update `__all__` if exported symbols change.
4. Run or update tests to ensure nothing is broken.

### Writing Tests
**Trigger:** When adding new features or fixing bugs.
**Command:** `/write-test`

1. Create a test file matching the pattern `*.test.*` (e.g., `context_manager.test.py`).
2. Write test functions for each public method or function.
3. Use assert statements to validate expected behavior.
4. Run tests to verify correctness.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `module.test.py`).
- The testing framework is unspecified; use standard `assert` statements or your preferred framework.
- Place tests alongside the modules they cover or in a dedicated `tests/` directory if present.

**Example test file:**
```python
# context_manager.test.py

from .context_manager import ContextManager

def test_context_creation():
    ctx = ContextManager()
    assert ctx is not None
```

## Commands
| Command        | Purpose                                       |
|----------------|-----------------------------------------------|
| /add-module    | Scaffold a new module with conventions        |
| /refactor      | Refactor code following style guidelines      |
| /write-test    | Create or update a test file for a module     |
```
