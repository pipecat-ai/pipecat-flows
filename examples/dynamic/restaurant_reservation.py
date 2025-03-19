#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowManager, FlowResult, NodeConfig

sys.path.append(str(Path(__file__).parent.parent))
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
        await asyncio.sleep(4.0)

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


# Start callback for availability check
async def before_check_availability(function_name: str, flow_manager: FlowManager) -> None:
    """Provide feedback while checking availability.

    This is called before the availability check function executes.
    """
    logger.debug(f"Executing start callback for: {function_name}")
    await flow_manager.llm.push_frame(TTSSpeakFrame("Let's see if we have an opening then..."))


# Function handlers
async def collect_party_size(args: FlowArgs) -> PartySizeResult:
    """Process party size collection."""
    size = args["size"]
    return PartySizeResult(size=size, status="success")


async def check_availability(args: FlowArgs) -> TimeResult:
    """Check reservation availability and return result."""
    time = args["time"]
    party_size = args["party_size"]

    # Check availability with mock API
    is_available, alternative_times = await reservation_system.check_availability(party_size, time)

    result = TimeResult(
        status="success", time=time, available=is_available, alternative_times=alternative_times
    )
    return result


# Transition handlers
async def handle_party_size_collection(
    args: Dict, result: PartySizeResult, flow_manager: FlowManager
):
    """Handle party size collection and transition to time selection."""
    # Store party size in flow state
    flow_manager.state["party_size"] = result["size"]
    await flow_manager.set_node("get_time", create_time_selection_node())


async def handle_availability_check(args: Dict, result: TimeResult, flow_manager: FlowManager):
    """Handle availability check result and transition based on availability."""
    # Store reservation details in flow state
    flow_manager.state["requested_time"] = args["time"]

    # Use result directly instead of accessing state
    if result["available"]:
        logger.debug("Time is available, transitioning to confirmation node")
        await flow_manager.set_node("confirm", create_confirmation_node())
    else:
        logger.debug(f"Time not available, storing alternatives: {result['alternative_times']}")
        await flow_manager.set_node(
            "no_availability", create_no_availability_node(result["alternative_times"])
        )


async def handle_end(_: Dict, result: FlowResult, flow_manager: FlowManager):
    """Handle conversation end."""
    await flow_manager.set_node("end", create_end_node())


# Node configurations
def create_initial_node() -> NodeConfig:
    """Create initial node for party size collection."""
    return {
        "role_messages": [
            {
                "role": "system",
                "content": "You are a restaurant reservation assistant for La Maison, an upscale French restaurant. Be casual and friendly. This is a voice conversation, so avoid special characters and emojis.",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Warmly greet the customer and ask how many people are in their party.",
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "collect_party_size",
                    "handler": collect_party_size,
                    "description": "Record the number of people in the party",
                    "parameters": {
                        "type": "object",
                        "properties": {"size": {"type": "integer", "minimum": 1, "maximum": 12}},
                        "required": ["size"],
                    },
                    "transition_callback": handle_party_size_collection,
                },
            }
        ],
    }


def create_time_selection_node() -> NodeConfig:
    """Create node for time selection and availability check."""
    logger.debug("Creating time selection node")
    return {
        "task_messages": [
            {
                "role": "system",
                "content": "Ask what time they'd like to dine. Restaurant is open 5 PM to 10 PM.",
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "check_availability",
                    "handler": check_availability,
                    "start_callback": before_check_availability,
                    "description": "Check availability for requested time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time": {
                                "type": "string",
                                "pattern": "^([5-9]|10):00 PM$",  # Matches "5:00 PM" through "10:00 PM"
                                "description": "Reservation time (e.g., '6:00 PM')",
                            },
                            "party_size": {"type": "integer"},
                        },
                        "required": ["time", "party_size"],
                    },
                    "transition_callback": handle_availability_check,
                },
            }
        ],
    }


def create_confirmation_node() -> NodeConfig:
    """Create confirmation node for successful reservations."""
    return {
        "task_messages": [
            {
                "role": "system",
                "content": "Confirm the reservation details and ask if they need anything else.",
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "end_conversation",
                    "description": "End the conversation",
                    "parameters": {"type": "object", "properties": {}},
                    "transition_callback": handle_end,
                },
            }
        ],
    }


def create_no_availability_node(alternative_times: list[str]) -> NodeConfig:
    """Create node for handling no availability."""
    times_list = ", ".join(alternative_times)
    return {
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
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "check_availability",
                    "handler": check_availability,
                    "start_callback": before_check_availability,
                    "description": "Check availability for new time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time": {
                                "type": "string",
                                "pattern": "^([5-9]|10):00 PM$",
                                "description": "Reservation time (e.g., '6:00 PM')",
                            },
                            "party_size": {"type": "integer"},
                        },
                        "required": ["time", "party_size"],
                    },
                    "transition_callback": handle_availability_check,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "end_conversation",
                    "description": "End the conversation",
                    "parameters": {"type": "object", "properties": {}},
                    "transition_callback": handle_end,
                },
            },
        ],
    }


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return {
        "task_messages": [
            {
                "role": "system",
                "content": "Thank them and end the conversation.",
            }
        ],
        "functions": [],
        "post_actions": [{"type": "end_conversation"}],
    }


# Main setup
async def main():
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Reservation bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
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
            await flow_manager.initialize()
            logger.debug("Setting initial node")
            await flow_manager.set_node("initial", create_initial_node())

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
