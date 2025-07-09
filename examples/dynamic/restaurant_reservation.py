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
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig

sys.path.append(str(Path(__file__).parent.parent))
import argparse

from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Mock reservation system
class MockReservationSystem:
    """Simulates a restaurant reservation system API."""

    def __init__(self):
        # Mock data: Times that are "fully booked"
        self.booked_times = {"7:00 PM", "8:00 PM"}  # Changed to AM/PM format

    async def check_availability(
        self, party_size: int, requested_time: str
    ) -> tuple[bool, list[str]]:
        """Check if a table is available for the given party size and time."""
        # Simulate API call delay
        await asyncio.sleep(0.5)

        # Check if time is booked
        is_available = requested_time not in self.booked_times

        # If not available, suggest alternative times
        alternatives = []
        if not is_available:
            base_times = ["5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM"]
            alternatives = [t for t in base_times if t not in self.booked_times]

        return is_available, alternatives


# Initialize mock system
reservation_system = MockReservationSystem()


# Type definitions for function results
class PartySizeResult(FlowResult):
    size: int
    status: str


class TimeResult(FlowResult):
    status: str
    time: str
    available: bool
    alternative_times: list[str]


# Function handlers
async def collect_party_size(args: FlowArgs) -> tuple[PartySizeResult, NodeConfig]:
    """Process party size collection."""
    size = args["size"]
    result = PartySizeResult(size=size, status="success")
    next_node = create_time_selection_node()
    return result, next_node


async def check_availability(args: FlowArgs) -> tuple[TimeResult, NodeConfig]:
    """Check reservation availability and return result."""
    time = args["time"]
    party_size = args["party_size"]

    # Check availability with mock API
    is_available, alternative_times = await reservation_system.check_availability(party_size, time)

    result = TimeResult(
        status="success", time=time, available=is_available, alternative_times=alternative_times
    )

    if is_available:
        logger.debug("Time is available, transitioning to confirmation node")
        next_node = create_confirmation_node()
    else:
        logger.debug(f"Time not available, storing alternatives: {result['alternative_times']}")
        next_node = create_no_availability_node(result["alternative_times"])

    return result, next_node


async def end_conversation(args: FlowArgs) -> tuple[None, NodeConfig]:
    """Handle conversation end."""
    return None, create_end_node()


# Create function schemas
party_size_schema = FlowsFunctionSchema(
    name="collect_party_size",
    description="Record the number of people in the party",
    properties={"size": {"type": "integer", "minimum": 1, "maximum": 12}},
    required=["size"],
    handler=collect_party_size,
)

availability_schema = FlowsFunctionSchema(
    name="check_availability",
    description="Check availability for requested time",
    properties={
        "time": {
            "type": "string",
            "pattern": "^([5-9]|10):00 PM$",  # Matches "5:00 PM" through "10:00 PM"
            "description": "Reservation time (e.g., '6:00 PM')",
        },
        "party_size": {"type": "integer"},
    },
    required=["time", "party_size"],
    handler=check_availability,
)

end_conversation_schema = FlowsFunctionSchema(
    name="end_conversation",
    description="End the conversation",
    properties={},
    required=[],
    handler=end_conversation,
)


# Node configurations
def create_initial_node(wait_for_user: bool) -> NodeConfig:
    """Create initial node for party size collection."""
    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": "You are a restaurant reservation assistant for La Maison, an upscale French restaurant. Be casual and friendly. This is a voice conversation, so avoid special characters and emojis.",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Warmly greet the customer and ask how many people are in their party. This is your only job for now; if the customer asks for something else, politely remind them you can't do it.",
            }
        ],
        "functions": [party_size_schema],
        "respond_immediately": not wait_for_user,
    }


def create_time_selection_node() -> NodeConfig:
    """Create node for time selection and availability check."""
    logger.debug("Creating time selection node")
    return {
        "name": "get_time",
        "task_messages": [
            {
                "role": "system",
                "content": "Ask what time they'd like to dine. Restaurant is open 5 PM to 10 PM.",
            }
        ],
        "functions": [availability_schema],
    }


def create_confirmation_node() -> NodeConfig:
    """Create confirmation node for successful reservations."""
    return {
        "name": "confirm",
        "task_messages": [
            {
                "role": "system",
                "content": "Confirm the reservation details and ask if they need anything else.",
            }
        ],
        "functions": [end_conversation_schema],
    }


def create_no_availability_node(alternative_times: list[str]) -> NodeConfig:
    """Create node for handling no availability."""
    times_list = ", ".join(alternative_times)
    return {
        "name": "no_availability",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    f"Apologize that the requested time is not available. "
                    f"Suggest these alternative times: {times_list}. "
                    "Ask if they'd like to try one of these times."
                ),
            }
        ],
        "functions": [availability_schema, end_conversation_schema],
    }


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": "Thank them and end the conversation.",
            }
        ],
        "post_actions": [{"type": "end_conversation"}],
    }


# Main setup
async def main(wait_for_user: bool):
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Reservation bot",
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
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

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
