<h1><div align="center">
 <img alt="pipecat" width="500px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/pipecat-flows.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-flows)](https://pypi.org/project/pipecat-ai-flows) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai/guides/features/pipecat-flows) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Flows is an add-on framework for [Pipecat](https://github.com/pipecat-ai/pipecat/tree/main#readme) that allows you to build structured conversations in your AI applications. It enables you to create both predefined conversation paths and dynamically generated flows while handling the complexities of state management and LLM interactions.

The framework consists of:

- A Python module for building conversation flows with Pipecat
- A [visual editor](#Pipecat-Flows-Editor) for designing and exporting flow configurations

## Dependencies

- Python 3.10 or higher
- [Pipecat](https://github.com/pipecat-ai/pipecat?tab=readme-ov-file#-getting-started)

## Installation

1. Install uv

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   > **Need help?** Refer to the [uv install documentation](https://docs.astral.sh/uv/getting-started/installation/).

2. Install the module

   ```bash
   # For new projects
   uv init my-pipecat-flows-app
   cd my-pipecat-flows-app
   uv add pipecat-ai-flows

   # Or for existing projects
   uv add pipecat-ai-flows
   ```

> **Using pip?** You can still use `pip install pipecat-ai-flows` to get set up.

## Quick Start

See [Quick Start README](./examples/quickstart/README.md).

For more detailed examples and guides, visit our [documentation](https://docs.pipecat.ai/guides/features/pipecat-flows).

## Examples

The repository includes several complete example implementations demonstrating various features of Pipecat Flows.

### Available Examples

The examples demonstrate various conversation flows including food ordering, restaurant reservations, patient intake, insurance quotes, and warm transfers. All examples support multiple LLM providers (OpenAI, Anthropic, Google Gemini, AWS Bedrock) to demonstrate cross-platform compatibility.

### Getting Started with Examples

For detailed setup instructions, configuration, and running examples, see the **[Examples README](examples/README.md)**.

Quick start:

```bash
# Install dependencies
uv sync
uv pip install "pipecat-ai[daily,openai,deepgram,cartesia,silero,examples]"

# Configure environment
cp env.example .env  # Add your API keys

# Run an example
uv run examples/food_ordering.py
```

## Contributing to the framework

1. Clone the repository and navigate to it:

   ```bash
   git clone https://github.com/pipecat-ai/pipecat-flows.git
   cd pipecat-flows
   ```

2. Install development dependencies:

   ```bash
   uv sync --group dev
   ```

3. Install the git pre-commit hooks (these help ensure your code follows project rules):

   ```bash
   uv run pre-commit install
   ```

   > The package is automatically installed in editable mode when you run `uv sync`.

## Tests

The package includes a comprehensive test suite covering the core functionality.

### Setup Test Environment

Install venv and dependencies:

```bash
uv sync --group dev
```

### Running Tests

Run all tests:

```bash
uv run pytest tests/
```

Run specific test file:

```bash
uv run pytest tests/test_state.py
```

Run specific test:

```bash
uv run pytest tests/test_state.py -k test_initialization
```

Run with coverage report:

```bash
uv run pytest tests/ --cov=pipecat_flows
```

## Pipecat Flows Editor

A visual editor for creating and managing Pipecat conversation flows.

![Food ordering flow example](https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/images/flows-food-ordering.png)

Visit the [Pipecat Flows Editor](https://github.com/pipecat-ai/pipecat-flows-editor) repo to learn more.

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, improving documentation, or adding new features, here's how you can help:

- **Found a bug?** Open an [issue](https://github.com/pipecat-ai/pipecat-flows/issues)
- **Have a feature idea?** Start a [discussion](https://discord.gg/pipecat)
- **Want to contribute code?** Check our [CONTRIBUTING.md](CONTRIBUTING.md) guide
- **Documentation improvements?** [Docs](https://github.com/pipecat-ai/docs) PRs are always welcome

Before submitting a pull request, please check existing issues and PRs to avoid duplicates.

We aim to review all contributions promptly and provide constructive feedback to help get your changes merged.

## Getting help

➡️ [Join our Discord](https://discord.gg/pipecat)

➡️ [Pipecat Flows Guide](https://docs.pipecat.ai/guides/pipecat-flows)

➡️ [Reach us on X](https://x.com/pipecat_ai)
