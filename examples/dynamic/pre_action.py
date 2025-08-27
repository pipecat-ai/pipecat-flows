#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Simple Pre-Action Example using Pipecat Dynamic Flows with OpenAI.

This example demonstrates a simple 3-node flow:
1. Start node: LLM introduces itself and asks for participant's name
2. Middle node: Pre-action says "Ok!" before LLM responds
3. End node: LLM says goodbye

This example uses:
- Pipecat Flows for conversation management
- OpenAI gpt-4o as the LLM
- Daily as the transport service

Requirements:
- Daily room URL
- Daily API key
- OpenAI API key
- Deepgram API key
- Cartesia API key
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.filters.frame_filter import FrameFilter
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaHttpTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowManager, FlowResult, NodeConfig
from pipecat_flows.types import ActionConfig, FlowArgs, FlowsFunctionSchema

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Custom processor to track TTSSpeakFrame and BotStoppedSpeakingFrame
class CustomMuteFilter(FrameProcessor):
    """Processor to control custom muting based on TTSSpeakFrame and BotStoppedSpeakingFrame."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_muted = False

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        """Process frames to control muting state."""
        if isinstance(frame, TTSSpeakFrame):
            logger.debug(f"CustomMuteFilter: TTSSpeakFrame detected, enabling muting")
            self._is_muted = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug(f"CustomMuteFilter: BotStoppedSpeakingFrame detected, disabling muting")
            self._is_muted = False

        if isinstance(
            frame,
            (
                StartInterruptionFrame,
                StopInterruptionFrame,
                VADUserStartedSpeakingFrame,
                VADUserStoppedSpeakingFrame,
                UserStartedSpeakingFrame,
                UserStoppedSpeakingFrame,
                InputAudioRawFrame,
                InterimTranscriptionFrame,
                TranscriptionFrame,
            ),
        ):
            # Only pass VAD-related frames when not muted
            if not self._is_muted:
                await self.push_frame(frame, direction)
            else:
                logger.trace(f"{frame.__class__.__name__} suppressed - CustomSTT currently muted")
        else:
            # Pass all other frames through
            await self.push_frame(frame, direction)


# Function handlers
async def collect_name(args: FlowArgs, flow_manager: FlowManager) -> tuple[FlowResult, NodeConfig]:
    """Collect the participant's name and move to the greeting node."""
    name = args.get("name", "")

    # Store the name in flow manager state
    flow_manager._state["participant_name"] = name

    result: FlowResult = {"status": "success"}
    next_node = create_greeting_node()
    return result, next_node


async def finish_conversation(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[FlowResult, NodeConfig]:
    """End the conversation."""
    result: FlowResult = {"status": "success"}
    next_node = create_goodbye_node()
    return result, next_node


# Node configurations
def create_start_node() -> NodeConfig:
    """Create the start node where the LLM introduces itself and asks for the participant's name."""
    return NodeConfig(
        name="start_node",
        role_messages=[
            {
                "role": "system",
                "content": "You are a friendly AI assistant. Keep your responses conversational and warm. This is a voice conversation, so avoid special characters and emojis.",
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": "Introduce yourself as an AI assistant and ask the participant for their name. Be friendly and welcoming.",
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="collect_name",
                description="Collect the participant's name",
                handler=collect_name,
                properties={"name": {"type": "string", "description": "The participant's name"}},
                required=["name"],
            )
        ],
    )


def create_greeting_node() -> NodeConfig:
    """Create the greeting node with a pre-action that says 'Ok!' before responding."""
    return NodeConfig(
        name="greeting_node",
        pre_actions=[
            ActionConfig(
                type="tts_say",
                text="Ok! Give me just one moment! This is a longer than usual pre-action, to demonstrate that things are being muted!",
            )
        ],
        task_messages=[
            {
                "role": "system",
                "content": "Greet the participant warmly using their name from the previous interaction and ask if they're ready to finish the conversation. Be friendly and personal.",
            }
        ],
        functions=[
            FlowsFunctionSchema(
                name="finish_conversation",
                description="Finish the conversation",
                handler=finish_conversation,
                properties={},
                required=[],
            )
        ],
    )


def create_goodbye_node() -> NodeConfig:
    """Create the goodbye node that says farewell."""
    return NodeConfig(
        name="goodbye_node",
        task_messages=[
            {
                "role": "system",
                "content": "Say a warm goodbye to the participant and wish them well.",
            }
        ],
        post_actions=[ActionConfig(type="end_conversation")],
    )


async def main():
    """Main function to set up and run the bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Initialize services
        transport = DailyTransport(
            room_url=room_url,
            token=token,
            bot_name="Pre-Action Example Bot",
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )
        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))
        tts = CartesiaHttpTTSService(
            api_key=os.getenv("CARTESIA_API_KEY", ""),
            voice_id="d46abd1d-2d02-43e8-819f-51fb652c1c61",  # Newsman
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Initialize context
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        # Create STT mute filter with both FUNCTION_CALL and CUSTOM strategies
        stt_mute_filter = STTMuteFilter(
            config=STTMuteConfig(
                strategies={STTMuteStrategy.FUNCTION_CALL},
            )
        )

        custom_mute_filter = CustomMuteFilter()

        # Create pipeline
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                stt_mute_filter,  # Add STTMuteFilter between STT and context aggregator
                custom_mute_filter,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )
        task = PipelineTask(
            pipeline=pipeline,
            params=PipelineParams(
                allow_interruptions=True, observers=[DebugLogObserver(frame_types=(TTSSpeakFrame,))]
            ),
        )

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            transport=transport,
        )

        # Set up event handlers
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(
            transport: DailyTransport, participant: Dict[str, Any]
        ):
            """Start the flow when the first participant joins."""
            await transport.capture_participant_transcription(participant["id"])
            # Initialize flow with the start node
            await flow_manager.initialize(create_start_node())

        @transport.event_handler("on_participant_left")
        async def on_participant_left(
            transport: DailyTransport, participant: Dict[str, Any], reason: str
        ):
            """Stop the bot when all participants leave."""
            remaining_participants = [
                p
                for p in transport.participants().values()
                if not p.get("info", {}).get("isLocal", True)
            ]
            if not remaining_participants:
                await task.cancel()

        # Print join URL
        logger.info(
            f"\n\nJOIN THE CONVERSATION:\n{room_url}{'?' if '?' not in room_url else '&'}t={token}\n"
        )

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
