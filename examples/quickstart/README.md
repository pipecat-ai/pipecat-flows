# Pipecat Flows: Quickstart

This quickstart example will help you set up your first Flow and explain to you the basics of using Flows.

## Dependencies

- Python 3.10 or higher
- [Pipecat](https://github.com/pipecat-ai/pipecat?tab=readme-ov-file#-getting-started)

## Installation

1. Setup your virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install Pipecat Flows:

```bash
pip install pipecat-ai-flows
```

3. Install the quickstart requirements:

```bash
pip install -r requirements.txt
```

4. Create a .env file with your API keys. For this demo, keys are required for [Cartesia](https://play.cartesia.ai/sign-up) and [Google Gemini](https://ai.google.dev/).

## Run the example

The example makes use of:

- SmallWebRTCTransport: A free peer-to-peer WebRTC transport, which sends audio to/from the bot
- Cartesia: A service provider for STT and TTS
- Google Gemini: An LLM inference provider

1. From within the `/examples/quickstart` directory, run:

```bash
python hello_world.py
```

2. Connect to http://localhost:7860 using your web browser.

3. Press connect to start the example.

The example is a simple bot that asks for your favorite color and leaves. While this is a simple example, it shows all of the basics of building with Flows.

Check out the source code to learn more.
