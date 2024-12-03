<h1><div align="center">
 <img alt="pipecat" width="500px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/pipecat-flows.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-flows)](https://pypi.org/project/pipecat-ai-flows) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Flows provides a framework for building structured conversations in your AI applications. It enables you to create both predefined conversation paths and dynamically generated flows while handling the complexities of state management and LLM interactions.

The framework consists of:

- A Python module for building conversation flows with Pipecat
- A visual editor for designing and exporting flow configurations

### When to Use Pipecat Flows

- **Static Flows**: When your conversation structure is known upfront and follows predefined paths. Perfect for customer service scripts, intake forms, or guided experiences.
- **Dynamic Flows**: When conversation paths need to be determined at runtime based on user input, external data, or business logic. Ideal for personalized experiences or complex decision trees.

## Installation

If you're already using Pipecat:

```bash
pip install pipecat-ai-flows
```

If you're starting fresh:

```bash
# Basic installation
pip install pipecat-ai-flows

# Install Pipecat with specific LLM provider options:
pip install "pipecat-ai[daily,openai,deepgram]"     # For OpenAI
pip install "pipecat-ai[daily,anthropic,deepgram]"  # For Anthropic
pip install "pipecat-ai[daily,google,deepgram]"     # For Google
```

## Quick Start

Here's a basic example of setting up a conversation flow:

```python
from pipecat_flows import FlowManager

# Initialize flow manager with static configuration
flow_manager = FlowManager(task, llm, tts, flow_config=flow_config)

# Or with dynamic flow handling
flow_manager = FlowManager(
    task,
    llm,
    tts,
    transition_callback=handle_transitions
)

@transport.event_handler("on_first_participant_joined")
async def on_first_participant_joined(transport, participant):
    await transport.capture_participant_transcription(participant["id"])
    await flow_manager.initialize(messages)
    await task.queue_frames([context_aggregator.user().get_context_frame()])
```

For more detailed examples and guides, visit our [documentation](https://docs.pipecat.ai/guides/pipecat-flows).

## Core Concepts

### Flow Configuration

Each conversation flow consists of nodes that define the conversation structure. A node includes:

#### Messages

Messages set the context for the LLM at each state:

```python
"messages": [
    {
        "role": "system",
        "content": "You are handling pizza orders. Ask for size selection."
    }
]
```

#### Functions

Functions come in two types:

1. **Node Functions**: Execute operations within the current state

```python
{
    "type": "function",
    "function": {
        "name": "select_size",
        "handler": select_size_handler,  # Required for node functions
        "description": "Select pizza size",
        "parameters": {
            "type": "object",
            "properties": {
                "size": {"type": "string", "enum": ["small", "medium", "large"]}
            }
        }
    }
}
```

2. **Edge Functions**: Create transitions between states

```python
{
    "type": "function",
    "function": {
        "name": "next_node",  # Must match a node name
        "description": "Move to next state",
        "parameters": {"type": "object", "properties": {}}
    }
}
```

#### Actions

Actions execute during state transitions:

```python
"pre_actions": [
    {
        "type": "tts_say",
        "text": "Processing your order..."
    }
]
```

#### Provider-Specific Formats

Pipecat Flows automatically handles format differences between LLM providers:

**OpenAI Format**

```python
"functions": [{
    "type": "function",
    "function": {
        "name": "function_name",
        "description": "description",
        "parameters": {...}
    }
}]
```

**Anthropic Format**

```python
"functions": [{
    "name": "function_name",
    "description": "description",
    "input_schema": {...}
}]
```

**Google (Gemini) Format**

```python
"functions": [{
    "function_declarations": [{
        "name": "function_name",
        "description": "description",
        "parameters": {...}
    }]
}]
```

### Flow Management

The FlowManager handles both static and dynamic flows through a unified interface:

#### Static Flows

```python
# Define flow configuration upfront
flow_config = {
    "initial_node": "greeting",
    "nodes": {
        "greeting": {
            "messages": [...],
            "functions": [...]
        }
    }
}

# Initialize with static configuration
flow_manager = FlowManager(task, llm, tts, flow_config=flow_config)
```

#### Dynamic Flows

```python
# Define transition handling
async def handle_transitions(function_name: str, args: Dict, flow_manager):
    if function_name == "collect_age":
        await flow_manager.set_node("next_step", create_next_node())

# Initialize with transition callback
flow_manager = FlowManager(task, llm, tts, transition_callback=handle_transitions)
```

## Examples

The repository includes several complete example implementations in the `examples/` directory.

### Static

In the `examples/static` directory, you'll find these examples:

- `food_ordering.py` - A restaurant order flow demonstrating node and edge functions
- `movie_explorer_openai.py` - Movie information bot demonstrating real API integration with TMDB
- `movie_explorer_anthropic.py` - The same movie information demo adapted for Anthropic's format
- `movie_explorer_gemini.py` - The same movie explorer demo adapted for Google Gemini's format
- `patient_intake.py` - A medical intake system showing complex state management
- `restaurant_reservation.py` - A reservation system with availability checking
- `travel_planner.py` - A vacation planning assistant with parallel paths

### Dynamic

In the `examples/dynamic` directory, you'll find these examples:

- `insurance_openai.py` - An insurance quote system using OpenAI's format
- `insurance_anthropic.py` - The same insurance system adapted for Anthropic's format
- `insurance_gemini.py` - The insurance system implemented with Google's format

Each LLM provider (OpenAI, Anthropic, Google) has slightly different function calling formats, but Pipecat Flows handles these differences internally while maintaining a consistent API for developers.

To run these examples:

1. **Setup Virtual Environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Installation**:

   Install the package in development mode:

   ```bash
   pip install -e .
   ```

   Install Pipecat with required options for examples:

   ```bash
   pip install "pipecat-ai[daily,openai,deepgram,silero,examples]"
   ```

   If you're running Google or Anthropic examples, you will need to update the installed options. For example:

   ```bash
   # Install Google Gemini
   pip install "pipecat-ai[daily,google,deepgram,silero,examples]"
   # Install Anthropic
   pip install "pipecat-ai[daily,anthropic,deepgram,silero,examples]"
   ```

3. **Configuration**:

   Copy `env.example` to `.env` in the examples directory:

   ```bash
   cp env.example .env
   ```

   Add your API keys and configuration:

   - DEEPGRAM_API_KEY
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
   - GOOGLE_API_KEY
   - DAILY_API_KEY

   Looking for a Daily API key and room URL? Sign up on the [Daily Dashboard](https://dashboard.daily.co).

4. **Running**:
   ```bash
   python examples/static/food_ordering.py -u YOUR_DAILY_ROOM_URL
   ```

## Tests

The package includes a comprehensive test suite covering the core functionality.

### Setup Test Environment

1. **Create Virtual Environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Test Dependencies**:
   ```bash
   pip install -r dev-requirements.txt -r test-requirements.txt
   pip install "pipecat-ai[google,openai,anthropic]"
   pip install -e .
   ```

### Running Tests

Run all tests:

```bash
pytest tests/
```

Run specific test file:

```bash
pytest tests/test_state.py
```

Run specific test:

```bash
pytest tests/test_state.py -k test_initialization
```

Run with coverage report:

```bash
pytest tests/ --cov=pipecat_flows
```

## Pipecat Flows Editor

A visual editor for creating and managing Pipecat conversation flows.

![Food ordering flow example](https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/images/food-ordering-flow.png)

### Features

- Visual flow creation and editing
- Import/export of flow configurations
- Support for node and edge functions
- Merge node support for complex flows
- Real-time validation

### Naming Conventions

While the underlying system is flexible with node naming, the editor follows these conventions for clarity:

- **Start Node**: Named after your initial conversation state (e.g., "greeting", "welcome")
- **End Node**: Conventionally named "end" for clarity, though other names are supported
- **Flow Nodes**: Named to reflect their purpose in the conversation (e.g., "get_time", "confirm_order")

These conventions help maintain readable and maintainable flows while preserving technical flexibility.

### Online Editor

The editor is available online at [flows.pipecat.ai](https://flows.pipecat.ai).

### Local Development

#### Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

#### Installation

Clone the repository

```bash
git clone git@github.com:pipecat-ai/pipecat-flows.git
```

Navigate to project directory

```bash
cd pipecat-flows/editor
```

Install dependencies

```bash
npm install
```

Start development server

```bash
npm run dev
```

Open the page in your browser: http://localhost:5173.

#### Usage

1. Create a new flow using the toolbar buttons
2. Add nodes by right-clicking in the canvas
   - Start nodes can have descriptive names (e.g., "greeting")
   - End nodes are conventionally named "end"
3. Connect nodes by dragging from outputs to inputs
4. Edit node properties in the side panel
5. Export your flow configuration using the toolbar

#### Examples

The `editor/examples/` directory contains sample flow configurations:

- `food_ordering.json`
- `movie_booking.json`
- `movie_explorer.py`
- `patient_intake.json`
- `restaurant_reservation.json`
- `travel_planner.json`

To use an example:

1. Open the editor
2. Click "Import Flow"
3. Select an example JSON file

See the [examples directory](editor/examples/) for the complete files and documentation.

### Development

#### Available Scripts

- `npm start` - Start production server
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run preview:prod` - Preview production build with base path
- `npm run lint` - Check for linting issues
- `npm run lint:fix` - Fix linting issues
- `npm run format` - Format code with Prettier
- `npm run format:check` - Check code formatting
- `npm run docs` - Generate documentation
- `npm run docs:serve` - Serve documentation locally

#### Documentation

The Pipecat Flows Editor project uses JSDoc for documentation. To generate and view the documentation:

Generate documentation:

```bash
npm run docs
```

Serve documentation locally:

```bash
npm run docs:serve
```

View in browser by opening: http://localhost:8080

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
