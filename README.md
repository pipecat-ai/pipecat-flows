<h1><div align="center">
 <img alt="pipecat" width="500px" height="auto" src="https://raw.githubusercontent.com/pipecat-ai/pipecat-flows/main/pipecat-flows.png">
</div></h1>

[![PyPI](https://img.shields.io/pypi/v/pipecat-ai-flows)](https://pypi.org/project/pipecat-ai-flows) [![Docs](https://img.shields.io/badge/Documentation-blue)](https://docs.pipecat.ai/guides/features/pipecat-flows) [![Discord](https://img.shields.io/discord/1239284677165056021)](https://discord.gg/pipecat)

Pipecat Flows is an add-on framework for [Pipecat](https://github.com/pipecat-ai/pipecat/tree/main#readme) that allows you to build structured conversations in your AI applications. It enables you to create both predefined conversation paths and dynamically generated flows while handling the complexities of state management and LLM interactions.

The framework consists of:

- A Python module for building conversation flows with Pipecat
- A [visual editor](#Pipecat-Flows-Editor) for designing and exporting flow configurations

### When to Use Pipecat Flows

- **Static Flows**: When your conversation structure is known upfront and follows predefined paths. Perfect for customer service scripts, intake forms, or guided experiences.
- **Dynamic Flows**: When conversation paths need to be determined at runtime based on user input, external data, or business logic. Ideal for personalized experiences or complex decision trees.

## Dependencies

- Python 3.10 or higher
- [Pipecat](https://github.com/pipecat-ai/pipecat?tab=readme-ov-file#-getting-started)

## Installation

Setup virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Install

```bash
pip install pipecat-ai-flows
```

## Basic Example

This is a basic example of setting up a flow:

```python
from pipecat_flows import FlowManager, FlowsFunctionSchema

def create_initial_node() -> NodeConfig:
    collect_name_schema = FlowsFunctionSchema(
        name="collect_name",
        description="Collect user's name.",
        required=["name"],
        handler=handle_collect_name,
        properties={"name": {"type": "string"}},
    )

    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Your responses will be converted to audio."
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Start by greeting the user and asking for their name.",
            }
        ],
        "functions": [collect_name_schema]
    }

    # Initialize flow manager
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await flow_manager.initialize(create_initial_node())

```

## Quick Start

See [Quick Start README](./examples/quickstart/README.md).

For more detailed examples and guides, visit our [documentation](https://docs.pipecat.ai/guides/features/pipecat-flows).

## Core Concepts

### Flow Configuration

Each conversation flow consists of nodes that define the conversation structure. A node includes:

#### Messages

Nodes use two types of messages to control the conversation:

1. **Role Messages**: Define the bot's personality or role (optional)

```python
"role_messages": [
    {
        "role": "system",
        "content": "You are a friendly pizza ordering assistant. Keep responses casual and upbeat."
    }
]
```

2. **Task Messages**: Define what the bot should do in the current node

```python
"task_messages": [
    {
        "role": "system",
        "content": "Ask the customer which pizza size they'd like: small, medium, or large."
    }
]
```

Role messages are typically defined in your initial node and inherited by subsequent nodes, while task messages are specific to each node's purpose.

#### Functions

Functions come in two types:

1. **Node Functions**: Execute operations within the current state
2. **Edge Functions**: Create transitions between states

Functions can be defined using either the new `FlowsFunctionSchema` class (recommended) or traditional dictionary format:

```python
# Using FlowsFunctionSchema (recommended)
from pipecat_flows import FlowsFunctionSchema

select_size_schema = FlowsFunctionSchema(
    name="select_size",
    description="Select pizza size",
    properties={"size": {"type": "string", "enum": ["small", "medium", "large"]}},
    required=["size"],
    handler=select_size_handler
)

# Traditional dictionary format
{
    "type": "function",
    "function": {
        "name": "select_size",
        "handler": select_size_handler,
        "description": "Select pizza size",
        "parameters": {
            "type": "object",
            "properties": {
                "size": {"type": "string", "enum": ["small", "medium", "large"]}
            }
        },
    }
}
```

Functions behave differently based on their type:

- Node Functions execute their handler and trigger an immediate LLM completion with the result
- Edge Functions execute their handler (if provided) and transition to a new node, with the LLM completion occurring after both the function result and new node's messages are added to context

Functions can:

- Have a handler (for data processing)
- Have a transition_to or transition callback (for state changes)
- Have both (process data and transition)
- Have neither (end node functions)

For Static flows, use `transition_to`:

```python
# Using FlowsFunctionSchema
next_step_schema = FlowsFunctionSchema(
    name="next_step",
    description="Move to next state",
    properties={},
    required=[],
    handler=select_size_handler,  # Optional handler
    transition_to="target_node"   # Specify target node
)

# Using dictionary format
{
    "type": "function",
    "function": {
        "name": "next_step",
        "handler": select_size_handler, # Optional handler
        "description": "Move to next state",
        "parameters": {"type": "object", "properties": {}},
        "transition_to": "target_node"  # Required: Specify target node
    }
}
```

For Dynamic flows, use `transition_callback`:

```python
# Using FlowsFunctionSchema
collect_age_schema = FlowsFunctionSchema(
    name="collect_age",
    description="Record user's age",
    properties={"age": {"type": "integer"}},
    required=["age"],
    handler=collect_age,
    transition_callback=handle_age_collection
)

# Using dictionary format
{
    "type": "function",
    "function": {
        "name": "collect_age",
        "handler": collect_age,
        "description": "Record user's age",
        "parameters": {...},
        "transition_callback": handle_age_collection  # Specify transition handler
    }
}
```

Pipecat Flows automatically handles format differences between LLM providers (OpenAI, Anthropic, and Google Gemini), so you can focus on your conversation logic rather than provider-specific implementations.

#### Actions

There are two types of actions available:

- `pre_actions`: Run before the LLM inference. For long function calls, you can use a pre_action for the TTS to say something, like "Hold on a moment..."
- `post_actions`: Run after the LLM inference. This is handy for actions like ending or transferring a call.

Actions can be registered in two ways:

1. Via handler field in action config:

```python
"pre_actions": [
    # Built-in action (no handler needed)
    {
        "type": "tts_say",
        "text": "Processing your order..."
    },
    # Custom action with handler
    {
        "type": "check_status",
        "handler": check_status_handler
    }
]
```

2. Via manual registration:

```python
flow_manager.register_action("check_status", check_status_handler)
```

Built-in actions (`tts_say`, `end_conversation`) don't require registration.

Example custom action:

```python
async def check_status_handler(action: dict) -> None:
    """Custom action to check system status."""
    logger.info("Checking system status")
    # Perform status check
```

Learn more about built-in actions and defining your own action in the docs.

### Flow Management

The FlowManager handles both static and dynamic flows through a unified interface:

#### Static Flows

```python
# Define flow configuration upfront
flow_config = {
    "initial_node": "greeting",
    "nodes": {
        "greeting": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Your responses will be converted to audio."
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Start by greeting the user and asking for their name."
                }
            ],
            "functions": [{
                "type": "function",
                "function": {
                    "name": "collect_name",
                    "description": "Record user's name",
                    "parameters": {...},
                    "handler": collect_name_handler,     # Specify handler
                    "transition_to": "next_step"         # Specify transition
                }
            }]
        }
    }
}

# Create and initialize the FlowManager
flow_manager = FlowManager(
    task=task,
    llm=llm,
    context_aggregator=context_aggregator,
    flow_config=flow_config,
)
await flow_manager.initialize()
```

#### Dynamic Flows

Dynamic flows follow the same pattern as static flows, but use `transition_callback` instead of `transition_to` to specify runtime-determined transitions. Here's an example:

```python
# Define handlers
async def update_coverage(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Update coverage options; node function without a transition."""
    return {"coverage": args["coverage"]}

# Edge function transition handler
async def handle_age_collection(args: Dict, result: FlowResult, flow_manager: FlowManager):
    """Handle age collection transition; edge function which transitions to the next node."""
    # Use typed result directly
    flow_manager.state["age"] = result.age
    await flow_manager.set_node("next", create_next_node())

# Create nodes
def create_initial_node() -> NodeConfig:
    return {
        "role_messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Ask the user for their age."
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "collect_age",
                    "handler": collect_age,
                    "description": "Record user's age",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "age": {"type": "integer"}
                        },
                        "required": ["age"]
                    },
                    "transition_callback": handle_age_collection  # Specify transition handler
                }
            }
        ]
    }

# Initialize flow manager
flow_manager = FlowManager(
    task=task,
    llm=llm,
    context_aggregator=context_aggregator,
)
await flow_manager.initialize()

@transport.event_handler("on_first_participant_joined")
async def on_first_participant_joined(transport, participant):
    await transport.capture_participant_transcription(participant["id"])
    await flow_manager.initialize()
    await flow_manager.set_node("initial", create_initial_node())
```

### Context Management

The `FlowManager` provides three strategies for managing conversation context during node transitions:

- **APPEND** (default): Adds new messages to the existing context, maintaining the full conversation history
- **RESET**: Clears the context and starts fresh with the new node's messages, including the previous function call results
- **RESET_WITH_SUMMARY**: Resets the context but includes an AI-generated summary of the previous conversation and the new node's messages

Strategies can be set globally or per-node:

```python
# Global strategy
flow_manager = FlowManager(
    task=task,
    llm=llm,
    context_aggregator=context_aggregator,
    context_strategy=ContextStrategyConfig(
        strategy=ContextStrategy.RESET
    )
)

# Per-node strategy
node_config = {
    "task_messages": [...],
    "functions": [...],
    "context_strategy": ContextStrategyConfig(
        strategy=ContextStrategy.RESET_WITH_SUMMARY,
        summary_prompt="Summarize the key points discussed so far."
    )
}
```

When using `RESET_WITH_SUMMARY`, the system automatically falls back to `RESET` if summary generation fails or times out.

## Examples

The repository includes several complete example implementations in the `examples/` directory.

### Static

In the `examples/static` directory, you'll find these examples:

- `food_ordering.py` - A restaurant order flow demonstrating node and edge functions
- `movie_explorer_openai.py` - Movie information bot demonstrating real API integration with TMDB
- `movie_explorer_anthropic.py` - The same movie information demo adapted for Anthropic's format
- `movie_explorer_gemini.py` - The same movie explorer demo adapted for Google Gemini's format
- `patient_intake_openai.py` - A medical intake system showing complex state management
- `patient_intake_anthropic.py` - The same medical intake demo adapted for Anthropic's format
- `patient_intake_gemini.py` - The same medical intake demo adapted for Gemini's format
- `travel_planner.py` - A vacation planning assistant with parallel paths

### Dynamic

In the `examples/dynamic` directory, you'll find these examples:

- `insurance_openai.py` - An insurance quote system using OpenAI's format
- `insurance_anthropic.py` - The same insurance system adapted for Anthropic's format
- `insurance_gemini.py` - The insurance system implemented with Google's format
- `restaurant_reservation.py` - A reservation system with availability checking

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
   pip install "pipecat-ai[daily,openai,deepgram,cartesia,silero,examples]"
   ```

   If you're running Google or Anthropic examples, you will need to update the installed options. For example:

   ```bash
   # Install Google Gemini
   pip install "pipecat-ai[daily,google,deepgram,cartesia,silero,examples]"
   # Install Anthropic
   pip install "pipecat-ai[daily,anthropic,deepgram,cartesia,silero,examples]"
   ```

3. **Configuration**:

   Copy `env.example` to `.env` in the examples directory:

   ```bash
   cp env.example .env
   ```

   Add your API keys and configuration:

   - DEEPGRAM_API_KEY
   - CARTESIA_API_KEY
   - OPENAI_API_KEY
   - ANTHROPIC_API_KEY
   - GOOGLE_API_KEY
   - DAILY_API_KEY

   Looking for a Daily API key and room URL? Sign up on the [Daily Dashboard](https://dashboard.daily.co).

4. **Running**:
   ```bash
   python examples/static/food_ordering.py -u YOUR_DAILY_ROOM_URL
   ```

## Hacking on the framework

1. Set up a virtual environment before following these instructions. From the root of the repo:

   ```shell
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the development dependencies:

   ```shell
   pip install -r dev-requirements.txt
   ```

3. Install the git pre-commit hooks (these help ensure your code follows project rules):

   ```shell
   pre-commit install
   ```

4. Install the `pipecat-ai-flows` package locally in editable mode:

   ```shell
   pip install -e .
   ```

   > The `-e` or `--editable` option allows you to modify the code without reinstalling.

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
   pip install -r dev-requirements.txt
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
