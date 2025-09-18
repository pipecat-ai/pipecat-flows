# Pipecat Flows: Quickstart

This quickstart example will help you set up your first Flow and explain to you the basics of using Flows.

## Prerequisites

- Python 3.10 or higher
- `uv` package manager

## Installation

1. Setup your virtual environment and install dependencies:

   ```bash
   uv sync
   ```

2. Create an .env file with API keys for

   - [Cartesia](https://play.cartesia.ai/sign-up)
   - [Google Gemini](https://ai.google.dev/)

   You can find the environment variable names in the [env.example](../../env.example).

## Run the example

The example makes use of:

- SmallWebRTCTransport: A free peer-to-peer WebRTC transport, which sends audio to/from the bot
- Cartesia: A service provider for STT and TTS
- Google Gemini: An LLM inference provider

1. From within the `/examples/quickstart` directory, run:

   ```bash
   uv run hello_world.py
   ```

2. Connect to http://localhost:7860 using your web browser.

3. Press "Connect" to start the example.

The example is a simple bot that asks for your favorite color and leaves. While this is a simple example, it shows all of the basics of building with Flows.

Check out the [source code](/examples/quickstart/hello_world.py) to learn more.
