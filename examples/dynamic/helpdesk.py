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

load_dotenv(override=False)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Food ordering
#
# This configuration defines a simple food ordering system with the following states:
#
# 1. start
#    - Initial state where user chooses between pizza or sushi
#    - Functions: choose_pizza, choose_sushi
#    - Transitions to: choose_pizza or choose_sushi
#
# 2. choose_pizza
#    - Handles pizza size selection and order confirmation
#    - Functions:
#      * select_pizza_size (node function, can be called multiple times)
#      * end (transitions to end node after order confirmation)
#    - Pre-action: Immediate TTS acknowledgment
#
# 3. choose_sushi
#    - Handles sushi roll count selection and order confirmation
#    - Functions:
#      * select_roll_count (node function, can be called multiple times)
#      * end (transitions to end node after order confirmation)
#    - Pre-action: Immediate TTS acknowledgment
#
# 4. end
#    - Final state that closes the conversation
#    - No functions available
#    - Pre-action: Farewell message
#    - Post-action: Ends conversation


# Type definitions
class PizzaSizeResult(FlowResult):
    size: str


class RollCountResult(FlowResult):
    count: int


# Function handlers
async def select_pizza_size(args: FlowArgs) -> PizzaSizeResult:
    """Handle pizza size selection."""
    size = args["size"]
    return {"size": size}


async def select_roll_count(args: FlowArgs) -> RollCountResult:
    """Handle sushi roll count selection."""
    count = args["count"]
    return {"count": count}


flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "messages": [
                {
                    "role": "system",
                    "content": "For this step, ask the user if they want pizza or sushi, and wait for them to use a function to choose. Start off by greeting them. Be friendly and casual; you're taking an order for food over the phone.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_pizza",
                        "description": "User wants to order pizza",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "choose_sushi",
                        "description": "User wants to order sushi",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "choose_pizza": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are handling a pizza order. Use the available functions:\n - Use select_pizza_size when the user specifies a size (can be used multiple times if they change their mind or want to order multiple pizzas)\n - Use the end function ONLY when the user confirms they are done with their order\n\nAfter each size selection, confirm the selection and ask if they want to change it or complete their order. Only use the end function after the user confirms they are satisfied with their order.\n\nStart off by acknowledging the user's choice. Once they've chosen a size, ask if they'd like anything else. Remember to be friendly and casual.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_pizza_size",
                        "handler": select_pizza_size,
                        "description": "Record the selected pizza size",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {
                                    "type": "string",
                                    "enum": ["small", "medium", "large"],
                                    "description": "Size of the pizza",
                                }
                            },
                            "required": ["size"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "Complete the order (use only after user confirms)",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Ok, let me help you with your pizza order..."}
            ],
        },
        "choose_sushi": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are handling a sushi order. Use the available functions:\n - Use select_roll_count when the user specifies how many rolls (can be used multiple times if they change their mind or if they want to order multiple sushi rolls)\n - Use the end function ONLY when the user confirms they are done with their order\n\nAfter each roll count selection, confirm the count and ask if they want to change it or complete their order. Only use the end function after the user confirms they are satisfied with their order.\n\nStart off by acknowledging the user's choice. Once they've chosen a size, ask if they'd like anything else. Remember to be friendly and casual.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_roll_count",
                        "handler": select_roll_count,
                        "description": "Record the number of sushi rolls",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Number of rolls to order",
                                }
                            },
                            "required": ["count"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "Complete the order (use only after user confirms)",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {"type": "tts_say", "text": "Ok, let me help you with your sushi order..."}
            ],
        },
        "end": {
            "messages": [
                {
                    "role": "system",
                    "content": "The order is complete. Thank the user and end the conversation.",
                }
            ],
            "functions": [],
            "pre_actions": [{"type": "tts_say", "text": "Thank you for your order! Goodbye!"}],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}


async def main():
    """Main function to set up and run the food ordering bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        # Initialize services
        transport = DailyTransport(
            room_url,
            None,
            "Food Ordering Bot",
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

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Avoid outputting special characters and emojis.",
            }
        ]

        context = OpenAILLMContext(messages, flow_config["nodes"]["start"]["functions"])
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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager in static mode
        flow_manager = FlowManager(task, llm, tts, flow_config=flow_config)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            logger.debug("Initializing flow")
            await flow_manager.initialize(messages)
            logger.debug("Starting conversation")
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
