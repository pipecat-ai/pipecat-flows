#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path
from typing import List

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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult
from pipecat_flows.types import FlowsFunctionSchema

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Travel Planner
#
# This configuration defines a vacation planning system with the following states:
#
# 1. start
#    - Initial state where user chooses between beach or mountain vacation
#    - Functions: choose_beach, choose_mountain
#    - Pre-action: Welcome message
#    - Transitions to: choose_beach or choose_mountain
#
# 2. choose_beach/choose_mountain
#    - Handles destination selection for chosen vacation type
#    - Functions:
#      * select_destination (node function with location-specific options)
#      * get_dates (transitions to date selection)
#    - Pre-action: Destination-specific welcome message
#
# 3. get_dates
#    - Handles travel date selection
#    - Functions:
#      * record_dates (node function, can be modified)
#      * get_activities (transitions to activity selection)
#
# 4. get_activities
#    - Handles activity preference selection
#    - Functions:
#      * record_activities (node function, array-based selection)
#      * verify_itinerary (transitions to verification)
#
# 5. verify_itinerary
#    - Reviews complete vacation plan
#    - Functions:
#      * revise_plan (loops back to get_dates)
#      * confirm_booking (transitions to confirmation)
#
# 6. confirm_booking
#    - Handles final confirmation and tips
#    - Functions: end
#    - Pre-action: Confirmation message
#
# 7. end
#    - Final state that closes the conversation
#    - No functions available
#    - Post-action: Ends conversation


# Type definitions
class DestinationResult(FlowResult):
    destination: str


class DatesResult(FlowResult):
    check_in: str
    check_out: str


class ActivitiesResult(FlowResult):
    activities: List[str]


# Function handlers
async def select_destination(args: FlowArgs) -> tuple[DestinationResult, str]:
    """Handler for destination selection."""
    destination = args["destination"]
    # In a real app, this would store the selection
    return DestinationResult(destination=destination), "get_dates"


async def record_dates(args: FlowArgs) -> tuple[DatesResult, str]:
    """Handler for travel date recording."""
    check_in = args["check_in"]
    check_out = args["check_out"]
    # In a real app, this would validate and store the dates
    return DatesResult(check_in=check_in, check_out=check_out), "get_activities"


async def record_activities(args: FlowArgs) -> tuple[ActivitiesResult, str]:
    """Handler for activity selection."""
    activities = args["activities"]
    # In a real app, this would validate and store the activities
    return ActivitiesResult(activities=activities), "verify_itinerary"


async def choose_beach(args: FlowArgs) -> tuple[None, str]:
    """Handler for choosing a beach vacation."""
    return None, "choose_beach"


async def choose_mountain(args: FlowArgs) -> tuple[None, str]:
    """Handler for choosing a mountain retreat."""
    return None, "choose_mountain"


async def revise_plan(args: FlowArgs) -> tuple[None, str]:
    """Handler to revise the vacation plan."""
    return None, "get_dates"


async def confirm_booking(args: FlowArgs) -> tuple[None, str]:
    """Handler to confirm the vacation booking."""
    return None, "confirm_booking"


async def end(args: FlowArgs) -> tuple[None, str]:
    """Handler to end the conversation."""
    return None, "end"


flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a travel planning assistant with Summit & Sand Getaways. You must ALWAYS use one of the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Avoid outputting special characters and emojis.",
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "For this step, ask if they're interested in planning a beach vacation or a mountain retreat, and wait for them to choose. Start with an enthusiastic greeting and be conversational; you're helping them plan their dream vacation.",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="choose_beach",
                    handler=choose_beach,
                    description="User wants to plan a beach vacation",
                    properties={},
                    required=[],
                ),
                FlowsFunctionSchema(
                    name="choose_mountain",
                    handler=choose_mountain,
                    description="User wants to plan a mountain retreat",
                    properties={},
                    required=[],
                ),
            ],
        },
        "choose_beach": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "You are handling beach vacation planning. Use the available functions:\n - Use select_destination when the user chooses their preferred beach location\n - After destination is selected, dates will be collected automatically\n\nAvailable beach destinations are: 'Maui', 'Cancun', or 'Maldives'. After they choose, confirm their selection. Be enthusiastic and paint a picture of each destination.",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="select_destination",
                    handler=select_destination,
                    description="Record the selected beach destination",
                    properties={
                        "destination": {
                            "type": "string",
                            "enum": ["Maui", "Cancun", "Maldives"],
                            "description": "Selected beach destination",
                        }
                    },
                    required=["destination"],
                ),
            ],
        },
        "choose_mountain": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "You are handling mountain retreat planning. Use the available functions:\n - Use select_destination when the user chooses their preferred mountain location\n - After destination is selected, dates will be collected automatically\n\nAvailable mountain destinations are: 'Swiss Alps', 'Rocky Mountains', or 'Himalayas'. After they choose, confirm their selection. Be enthusiastic and paint a picture of each destination.",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="select_destination",
                    handler=select_destination,
                    description="Record the selected mountain destination",
                    properties={
                        "destination": {
                            "type": "string",
                            "enum": ["Swiss Alps", "Rocky Mountains", "Himalayas"],
                            "description": "Selected mountain destination",
                        }
                    },
                    required=["destination"],
                ),
            ],
        },
        "get_dates": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Handle travel date selection. Use the available functions:\n - Use record_dates when the user specifies their travel dates (can be used multiple times if they change their mind)\n - After dates are recorded, activities will be collected automatically\n\nAsk for their preferred travel dates within the next 6 months. After recording dates, confirm the selection.",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="record_dates",
                    handler=record_dates,
                    description="Record the selected travel dates",
                    properties={
                        "check_in": {
                            "type": "string",
                            "format": "date",
                            "description": "Check-in date (YYYY-MM-DD)",
                        },
                        "check_out": {
                            "type": "string",
                            "format": "date",
                            "description": "Check-out date (YYYY-MM-DD)",
                        },
                    },
                    required=["check_in", "check_out"],
                ),
            ],
        },
        "get_activities": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Handle activity preferences. Use the available functions:\n - Use record_activities to save their activity preferences\n - After activities are recorded, verification will happen automatically\n\nFor beach destinations, suggest: snorkeling, surfing, sunset cruise\nFor mountain destinations, suggest: hiking, skiing, mountain biking\n\nAfter they choose, confirm their selections.",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="record_activities",
                    handler=record_activities,
                    description="Record selected activities",
                    properties={
                        "activities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 3,
                            "description": "Selected activities",
                        }
                    },
                    required=["activities"],
                ),
            ],
        },
        "verify_itinerary": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Review the complete itinerary with the user. Summarize their destination, dates, and chosen activities. Use revise_plan to make changes or confirm_booking if they're happy. Be thorough in reviewing all details and ask for their confirmation.",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="revise_plan",
                    handler=revise_plan,
                    description="Return to date selection to revise the plan",
                    properties={},
                    required=[],
                ),
                FlowsFunctionSchema(
                    name="confirm_booking",
                    handler=confirm_booking,
                    description="Confirm the booking and proceed to end",
                    properties={},
                    required=[],
                ),
            ],
        },
        "confirm_booking": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "The booking is confirmed. Share some relevant tips about their chosen destination, thank them warmly, and then invoke the 'end' function to complete the conversation.",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="end",
                    handler=end,
                    description="End the conversation",
                    properties={},
                    required=[],
                ),
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Fantastic! Your dream vacation is confirmed!"}
            ],
        },
        "end": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Wish them a wonderful trip and end the conversation.",
                }
            ],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}


async def main():
    """Main function to set up and run the travel planning bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Planner Bot",
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
            text_filter=MarkdownTextFilter(),
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(context)

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

        # Initialize flow manager with LLM
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            flow_config=flow_config,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Initialize the flow processor
            await flow_manager.initialize()

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
