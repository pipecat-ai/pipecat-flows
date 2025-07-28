#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowManager, FlowResult, NodeConfig

sys.path.append(str(Path(__file__).parent.parent))

from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

current_llm = "OpenAI"
# current_llm = "Google"


async def switch_llm(flow_manager: FlowManager, llm: str) -> tuple[FlowResult, None]:
    """Switch the current LLM service.

    Args:
        llm: The name of the LLM service to switch to (must be "OpenAI" or "Google").
    """
    global current_llm
    current_llm = llm
    return FlowResult(status="success"), None


async def openai_filter(frame) -> bool:
    return current_llm == "OpenAI"


async def google_filter(frame) -> bool:
    return current_llm == "Google"


async def get_current_weather(flow_manager: FlowManager) -> tuple[FlowResult, None]:
    """Get the current weather information."""
    # This is a placeholder for the actual implementation
    # In a real scenario, you would call an API to get the weather data
    weather_info = "The current weather is sunny with a temperature of 75 degrees Fahrenheit."
    return FlowResult(status="success", response=weather_info), None


def create_initial_node() -> NodeConfig:
    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            }
        ],
        "task_messages": [
            {
                # TODO: should be able to specify "system" for OpenAI and "user" for Google
                "role": "system",
                "content": "Say a brief hello.",
            }
        ],
        "functions": [switch_llm, get_current_weather],
    }


# Main setup
async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "LLM Failover Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )

        # Shared context and aggregators for both LLMs
        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair.create(context)

        # Primary LLM service
        llm_openai = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # Secondary LLM service for failover
        llm_google = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                ParallelPipeline(
                    [
                        FunctionFilter(openai_filter, direction=FrameDirection.DOWNSTREAM),
                        llm_openai,
                        FunctionFilter(openai_filter, direction=FrameDirection.UPSTREAM),
                    ],
                    [
                        FunctionFilter(google_filter, direction=FrameDirection.DOWNSTREAM),
                        llm_google,
                        FunctionFilter(google_filter, direction=FrameDirection.UPSTREAM),
                    ],
                ),
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llms=[llm_openai, llm_google],
            context_aggregator=context_aggregator,
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            logger.debug("Initializing flow manager")
            await flow_manager.initialize(create_initial_node())

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
