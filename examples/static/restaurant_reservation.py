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
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Restaurant Reservation System
#
# This configuration defines a streamlined restaurant reservation system with the following states:
#
# 1. start
#    - Initial state collecting party size information
#    - Functions:
#      * record_party_size (node function, validates 1-12 people)
#      * get_time (edge function, transitions to time selection)
#    - Expected flow: Greet -> Ask party size -> Record -> Transition to time
#
# 2. get_time
#    - Collects preferred reservation time
#    - Operating hours: 5 PM - 10 PM
#    - Functions:
#      * record_time (node function, collects time in HH:MM format)
#      * confirm (edge function, transitions to confirmation)
#    - Expected flow: Ask preferred time -> Record time -> Proceed to confirmation
#
# 3. confirm
#    - Reviews reservation details with guest
#    - Functions:
#      * end (edge function, transitions to end)
#    - Expected flow: Review details -> Confirm -> End conversation
#
# 4. end
#    - Final state that closes the conversation
#    - No functions available
#    - Post-action: Ends conversation
#
# This simplified flow demonstrates both node functions (which perform operations within
# a state) and edge functions (which transition between states), while maintaining a
# clear and efficient reservation process.


async def record_party_size(args: FlowArgs) -> FlowResult:
    """Handler for recording party size."""
    size = args["size"]
    # In a real app, this would store the reservation details
    return {"status": "success", "size": size}


async def record_time(args: FlowArgs) -> FlowResult:
    """Handler for recording reservation time."""
    time = args["time"]
    # In a real app, this would validate availability and store the time
    return {"status": "success", "time": time}


flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "messages": [
                {
                    "role": "system",
                    "content": "Warmly greet the customer and ask how many people are in their party.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_party_size",
                        "handler": record_party_size,
                        "description": "Record the number of people in the party",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {"type": "integer", "minimum": 1, "maximum": 12}
                            },
                            "required": ["size"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Proceed to time selection",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "get_time": {
            "messages": [
                {
                    "role": "system",
                    "content": "Ask what time they'd like to dine. Restaurant is open 5 PM to 10 PM.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_time",
                        "handler": record_time,
                        "description": "Record the requested time",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time": {
                                    "type": "string",
                                    "pattern": "^([0-1][0-9]|2[0-3]):[0-5][0-9]$",
                                }
                            },
                            "required": ["time"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "confirm",
                        "description": "Proceed to confirmation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "confirm": {
            "messages": [
                {
                    "role": "system",
                    "content": "Confirm the reservation details and end the conversation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "End the conversation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        "end": {
            "messages": [{"role": "system", "content": "Thank them and end the conversation."}],
            "functions": [],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}


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
        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Get initial tools from the first node
        initial_tools = flow_config["nodes"]["start"]["functions"]

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are a restaurant reservation assistant for La Maison, an upscale French restaurant. You must ALWAYS use one of the available functions to progress the conversation. This is a phone conversations and your responses will be converted to audio. Avoid outputting special characters and emojis. Be causal and friendly.",
            }
        ]

        context = OpenAILLMContext(messages, initial_tools)
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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager with LLM
        flow_manager = FlowManager(task, llm, tts, flow_config)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Initialize the flow processor
            await flow_manager.initialize(messages)
            # Kick off the conversation using the context aggregator
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
