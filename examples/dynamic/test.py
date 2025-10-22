#
# Copyright (c) 2024-2025, Daily
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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from pipecat_flows import FlowManager, FlowResult, NodeConfig
from pipecat_flows.types import ActionConfig

sys.path.append(str(Path(__file__).parent.parent))
import argparse

from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Function handlers
async def dummy_function(flow_manager: FlowManager) -> tuple[FlowResult, NodeConfig]:
    """A dummy function that does nothing."""
    print("Dummy function called")
    return FlowResult(status="success"), create_end_node()

# Actions
async def noop_action(action: dict, flow_manager: FlowManager) -> None:
    """A no-operation action."""
    pass


# Node configurations
def create_initial_node(wait_for_user: bool) -> NodeConfig:
    """Create initial node."""
    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": "You friendly assistant.",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Say hello. If the user happens to ask about the weather, say that you unfortunately can't help, respond with a random bit of trivia, then call dummy_function.",
            }
        ],
        "functions": [dummy_function],
        "respond_immediately": not wait_for_user,
    }


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": "Say goodbye in a clever way that very specifically references the previous thing you said. Then, end the conversation.",
            }
        ],
        # The below forces the node setup to wait for the action, which in turn
        # waits for the pipeline to finish the work ahead of it. That way, the
        # last assistant message has a chance to get written to context before
        # we trigger another LLM run.
        "pre_actions": [
            ActionConfig(type="function", handler=noop_action),
        ],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }


# Main setup
async def main(wait_for_user: bool):
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Dummy bot",
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
        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash")

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            logger.debug("Initializing flow manager")
            await flow_manager.initialize(create_initial_node(wait_for_user))

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant reservation bot")
    parser.add_argument(
        "--wait-for-user",
        action="store_true",
        help="If set, the bot will wait for the user to speak first",
    )
    args = parser.parse_args()

    asyncio.run(main(args.wait_for_user))
