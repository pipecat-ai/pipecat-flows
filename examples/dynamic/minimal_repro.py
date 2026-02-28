import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

import aiohttp
from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowManager, FlowResult, NodeConfig
from pipecat_flows.types import ActionConfig

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

# Transitions


async def hello(flow_manager: FlowManager):
    """Transition to the "hello" node"""
    await flow_manager.set_node("hello", create_start_node())


async def transition_to_goodbye(args: Dict, result: FlowResult, flow_manager: FlowManager):
    await flow_manager.set_node("goodbye", create_goodbye_node())


# Node configuration


def create_start_node() -> NodeConfig:
    return NodeConfig(
        task_messages=[
            {
                "role": "system",
                "content": """Count from 1 to 9.
                If the user wants to end the conversation, call the goodbye function""",
            }
        ],
        functions=[
            {
                "type": "function",
                "function": {
                    "name": "goodbye",
                    "description": "Say goodbye and end the conversation",
                    "transition_callback": transition_to_goodbye,
                },
            },
        ],
    )


# Actions


async def dummy_post_action(action: dict, flow_manager: FlowManager):
    print("Dummy post-action!")


def create_goodbye_node() -> NodeConfig:
    return NodeConfig(
        task_messages=[
            {
                "role": "system",
                "content": """Say goodbye warmly""",
            }
        ],
        functions=[],
        post_actions=[
            ActionConfig(type="function", handler=dummy_post_action),
            ActionConfig(type="end_conversation"),
        ],
    )


async def main():
    """Main function to set up and run the bot."""

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Initialize services
        transport = DailyTransport(
            room_url=room_url,
            token=token,
            bot_name="ABC Widget Company Bot",
            params=DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="d46abd1d-2d02-43e8-819f-51fb652c1c61",  # Newsman
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Initialize context
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        # Create pipeline
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
        task = PipelineTask(pipeline=pipeline, params=PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            tts=tts,
            transport=transport,
        )

        # Set up event handlers
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: DailyTransport, participant: Dict[str, Any]
        ):
            """Start the flow.
            We're assuming the first participant is the customer and not the human agent.
            """
            await transport.capture_participant_transcription(participant["id"])
            # Initialize flow
            await flow_manager.initialize()
            # Set initial node
            await hello(flow_manager=flow_manager)

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
