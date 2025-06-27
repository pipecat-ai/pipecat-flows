#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""A 'Hello-World' introduction to Pipecat Flows

Requirements:
- Google API key
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Literal, TypedDict, Union

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_flows import (
    FlowArgs,
    FlowConfig,
    FlowManager,
    FlowResult,
    FlowsFunctionSchema,
    NodeConfig,
)

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True, audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer()
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True, audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer()
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True, audio_out_enabled=True, vad_analyzer=SileroVADAnalyzer()
    ),
}

# Flow nodes
def create_initial_node() -> NodeConfig:
    """Start here."""
    get_favorite_color_func = FlowsFunctionSchema(
                name="get_favorite_color",
                description="Record the color the user said is their favorite.",
                required=["color"],
                handler=print_favorite_color_and_set_next_node,
                properties={"color": {"type": "string"}},
            )

    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": "You are an inquisitive child. Use very simple language. Ask simple questions. You must ALWAYS use one of the available functions to progress the conversation. Your responses will be converted to audio. Avoid outputting special characters and emojis.",
            }
        ],
        "task_messages": [
            {"role": "system", "content": "Say 'Hello world' and ask what is the user's favorite color."}
        ],
        "functions": [get_favorite_color_func],
    }

async def print_favorite_color_and_set_next_node(args: FlowArgs, flow_manager: FlowManager) -> tuple[str, NodeConfig]:
    print(f"Your favorite color is: {args["color"]}")
    return args["color"], end_conversation()

def end_conversation() -> NodeConfig:
    return NodeConfig(
        name="end_conversation",
        task_messages=[
            {
                "role": "system",
                "content": "Thank the user for answering and end the conversation",
            }
        ],
        post_actions=[{"type": "end_conversation"}]
    )

async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="32b3f3c5-7171-46aa-abe7-b598964aa793",
        text_filter=MarkdownTextFilter(),
    )
    # llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-exp")
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    context = OpenAILLMContext()
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Initialize flow manager
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await flow_manager.initialize(create_initial_node())

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


if __name__ == "__main__":
    from run import main

    main(run_example, transport_params=transport_params)
