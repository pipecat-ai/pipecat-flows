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
from pipecat.pipeline.llm_switcher import LLMSwitcher, LLMSwitcherStrategyManual
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowManager, FlowResult, NodeConfig

sys.path.append(str(Path(__file__).parent.parent))

from runner import configure

from pipecat_flows.types import ContextStrategy, ContextStrategyConfig

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def switch_llm(flow_manager: FlowManager, llm: str) -> tuple[FlowResult, None]:
    """Switch the current LLM service.

    Args:
        llm: The name of the LLM service to switch to (must be "OpenAI" or "Google").
    """
    if llm == "OpenAI":
        llm_switcher.strategy.set_active(llm_openai)
    elif llm == "Google":
        llm_switcher.strategy.set_active(llm_google)
    return FlowResult(status="success"), None


async def get_current_weather(flow_manager: FlowManager) -> tuple[FlowResult, None]:
    """Get the current weather information."""
    # This is a placeholder for the actual implementation
    # In a real scenario, you would call an API to get the weather data
    weather_info = "The current weather is sunny with a temperature of 75 degrees Fahrenheit."
    return FlowResult(status="success", response=weather_info), None


async def summarize_conversation(flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Summarize the conversation so far."""
    return None, create_main_node(summarize=True)


def create_main_node(summarize: bool = False) -> NodeConfig:
    return NodeConfig(
        name="main",
        role_messages=[
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            }
        ],
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY,
            summary_prompt="Summarize the conversation so far in a concise way.",
        )
        if summarize
        else ContextStrategyConfig(strategy=ContextStrategy.APPEND),
        task_messages=[
            {
                # TODO: this isn't handled correctly by the Google LLM yet
                "role": "system",
                "content": "Say the conversation summary, which was already retrieved (do not invoke the summarize_conversation function again)."
                if summarize
                else "Say a brief hello.",
            }
        ],
        functions=[switch_llm, get_current_weather, summarize_conversation],
    )


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

        # Shared context and aggregators for LLM services
        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(context)

        # LLM services
        global llm_openai, llm_google, llm_switcher
        llm_openai = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
        llm_google = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"))
        llm_switcher = LLMSwitcher(
            llms=[llm_openai, llm_google], strategy_type=LLMSwitcherStrategyManual
        )

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm_switcher,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm_switcher,
            context_aggregator=context_aggregator,
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            logger.debug("Initializing flow manager")
            await flow_manager.initialize(create_main_node())

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
