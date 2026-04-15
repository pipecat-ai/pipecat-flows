# CLAUDE.md

## Project overview

Pipecat Flows is a conversation flow management framework for [Pipecat](https://www.pipecat.ai) applications. It provides structured conversations with runtime-determined transitions, state management, action execution, and LLM function calling across multiple providers (OpenAI, Anthropic, Google, AWS Bedrock).

Package name: `pipecat-ai-flows`
Python: 3.11+
License: BSD 2-Clause

## Architecture

Source code lives in `src/pipecat_flows/`:

- **`manager.py`** - Core `FlowManager` class. Orchestrates conversation flows, manages state transitions, registers/executes functions, handles actions.
- **`types.py`** - Type definitions: `NodeConfig`, `FlowsFunctionSchema`, `FlowResult`, `FlowArgs`, `ContextStrategy`, handler protocols, the `@flows_direct_function()` decorator.
- **`actions.py`** - `ActionManager` for pre/post-transition actions. Built-in actions: `tts_say`, `end_conversation`, `function`. Supports custom action registration.
- **`adapters.py`** - `LLMAdapter` bridges FlowManager and Pipecat's `LLMContext`. Handles function formatting and cross-provider compatibility.
- **`exceptions.py`** - Exception hierarchy: `FlowError` (base), `FlowInitializationError`, `FlowTransitionError`, `InvalidFunctionError`, `ActionError`.

Public API is exported from `__init__.py`.

## Development commands

```bash
# Install dependencies
uv sync --group dev

# Run tests
uv run pytest tests/

# Run a specific test
uv run pytest tests/test_manager.py -k test_flow_initialization

# Run tests with coverage
uv run pytest tests/ --cov=pipecat_flows

# Lint
uv run ruff check .

# Lint and auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .

# Format check (no changes)
uv run ruff format --check .

# Install pre-commit hooks
uv run pre-commit install

# Preview changelog
towncrier build --draft --version Unreleased
```

## Code style

- **Formatter/linter:** Ruff. Line length: 100.
- **Lint rules:** `D` (docstrings, Google convention), `I` (imports). `D105` is ignored (magic method docstrings).
- **Docstring exemptions:** `examples/`, `tests/`, `docs/` are excluded from docstring checks.
- **Docstrings:** Google-style. See CONTRIBUTING.md for full conventions including dataclasses (`Parameters:` not `Args:`), enums, properties, code examples, and deprecation directives.
- **Pre-commit hooks:** Ruff format + Ruff lint with `--fix`.

## Testing

Tests are in `tests/` using pytest with pytest-asyncio (`asyncio_mode = "auto"`).

Key test files:
- `test_manager.py` - FlowManager functionality, state transitions, function registration
- `test_context_strategies.py` - Context management strategies
- `test_flows_direct_functions.py` - Direct function support
- `test_actions.py` - Action execution
- `test_adapters.py` - LLM adapter
- `test_helpers.py` - Shared utilities (`make_mock_task()`, assertion helpers)

## Changelog

Every user-facing PR needs a changelog fragment in `changelog/`. Uses towncrier.

Format: `changelog/<PR_number>.<type>.md` where type is one of: `added`, `changed`, `deprecated`, `removed`, `fixed`, `performance`, `security`, `other`.

Content should be a markdown bullet point starting with `- `. Multiple changes of the same type use numbered files: `1234.changed.md`, `1234.changed.2.md`.

## CI

GitHub Actions workflows:
- **test.yml** - Runs `pytest tests/ --cov=pipecat_flows` on push to main and PRs
- **format.yaml** - Ruff format check + lint on push to main and PRs
- **publish-pypi.yml** - Build and publish to PyPI (manual trigger)
- **update-docs.yml** - Update ReadTheDocs on doc changes

## Examples

12 example applications in `examples/`. All support multiple LLM providers via `LLM_PROVIDER` env var. Run with:

```bash
cp env.example .env  # Fill in API keys
uv run examples/food_ordering.py
```
