# Pipecat Flows Examples

This directory contains complete example implementations demonstrating various features of Pipecat Flows.

## Available Examples

### Quickstart

See [quickstart/README.md](./quickstart/README.md) for a beginner-friendly introduction to Pipecat Flows.

### Main Examples

#### Core Flow Examples

- `food_ordering.py` - Restaurant order flow demonstrating node and edge functions
- `restaurant_reservation.py` - Reservation system with availability checking
- `patient_intake.py` - Medical intake system showing complex state management
- `insurance_quote.py` - Insurance quote system with data collection

#### Implementation Patterns

- `food_ordering_direct_functions.py` - Food ordering using direct function registration
- `restaurant_reservation_direct_functions.py` - Reservation system using direct function registration
- `llm_switching.py` - Switching between different LLM providers during conversation

#### Advanced Features

- `warm_transfer.py` - Transferring calls between different flows (DailyTransport only)

**Note:** All examples support multiple LLM providers (OpenAI, Anthropic, Google, AWS Bedrock) and transport options unless otherwise specified. Pipecat Flows handles the different function calling formats internally while maintaining a consistent API for developers.

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
uv run examples/food_ordering.py
```

Open http://localhost:7860/client in your browser to talk to your bot.

## Other Transports

The examples use Pipecat development runner, which supports using multiple clients. Join using either the SmallWebRTCTransport, DailyTransport, or FastAPIWebsocketTransport with Twilio/Telnyx/Plivo/Exotel:

- SmallWebRTCTransport:

  ```bash
  uv run examples/food_ordering.py
  ```

- DailyTransport:

  ```bash
  uv run examples/food_ordering.py --transport daily
  ```

- Twilio (or other telephony provider):

  Start an ngrok tunnel:

  ```bash
  ngrok http 7860
  ```

  > Tip: Use `--subdomain` for a reusable ngrok URL.

  Run the bot:

  ```bash
  uv run examples/food_ordering.py --transport twilio --proxy your-ngrok.ngrok.io
  ```

  replacing `your-ngrok` with your ngrok subdomain.
