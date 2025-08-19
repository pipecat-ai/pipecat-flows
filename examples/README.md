# Pipecat Flows Examples

This directory contains complete example implementations demonstrating various features of Pipecat Flows.

## Available Examples

### Quickstart

See [Quick Start README](./examples/quickstart/README.md).

### Static Flows

In the `static/` directory, you'll find these examples:

- `food_ordering.py` - A restaurant order flow demonstrating node and edge functions
- `food_ordering_direct_functions.py` - Same flow using direct function registration
- `movie_explorer_openai.py` - Movie information bot demonstrating real API integration with TMDB
- `movie_explorer_anthropic.py` - The same movie information demo adapted for Anthropic's format
- `movie_explorer_gemini.py` - The same movie explorer demo adapted for Google Gemini's format
- `patient_intake_openai.py` - A medical intake system showing complex state management
- `patient_intake_anthropic.py` - The same medical intake demo adapted for Anthropic's format
- `patient_intake_aws_bedrock.py` - The same medical intake demo adapted for AWS Bedrock
- `patient_intake_gemini.py` - The same medical intake demo adapted for Gemini's format
- `travel_planner.py` - A vacation planning assistant with parallel paths

### Dynamic Flows

In the `dynamic/` directory, you'll find these examples:

- `insurance_openai.py` - An insurance quote system using OpenAI's format
- `insurance_anthropic.py` - The same insurance system adapted for Anthropic's format
- `insurance_aws_bedrock.py` - The same insurance system adapted for AWS Bedrock
- `insurance_gemini.py` - The insurance system implemented with Google's format
- `restaurant_reservation.py` - A reservation system with availability checking
- `restaurant_reservation_direct_functions.py` - Same reservation system using direct function registration
- `warm_transfer.py` - Demonstrates transferring calls between different flows

Each LLM provider (OpenAI, Anthropic, Google, AWS Bedrock) has slightly different function calling formats, but Pipecat Flows handles these differences internally while maintaining a consistent API for developers.

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### 1. Installation

Install the package:

```bash
uv sync
```

Install Pipecat with required options for examples:

```bash
uv pip install "pipecat-ai[daily,openai,deepgram,cartesia,silero,examples]"
```

If you're running Google or Anthropic examples, you will need to update the installed options. For example:

```bash
# Install Google Gemini
uv pip install "pipecat-ai[daily,google,deepgram,cartesia,silero,examples]"
# Install Anthropic
uv pip install "pipecat-ai[daily,anthropic,deepgram,cartesia,silero,examples]"
# Install AWS Bedrock
uv pip install "pipecat-ai[daily,aws,deepgram,cartesia,silero,examples]"
```

### 2. Configuration

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
- AWS_ACCESS_KEY_ID (for Bedrock)
- AWS_SECRET_ACCESS_KEY (for Bedrock)
- AWS_REGION (for Bedrock)
- DAILY_API_KEY

Looking for a Daily API key and room URL? Sign up on the [Daily Dashboard](https://dashboard.daily.co).

### 3. Running Examples

Run any example using:

```bash
uv run python examples/static/food_ordering.py -u YOUR_DAILY_ROOM_URL
```

Replace `food_ordering.py` with any other example file, and provide your Daily room URL.
