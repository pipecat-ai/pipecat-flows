#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
Basic Food Ordering Example (without state)

This example demonstrates:
- Basic static flow configuration
- Simple node transitions
- Function handlers
- LLM interactions

A simpler starting point for learning Pipecat Flows.

Requirements:
- Daily room URL
- OpenAI API key
- Deepgram API key
- Cartesia API key
"""

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
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Food ordering
#
# This configuration defines a food ordering system with the following states:
#
# 1. start
#    - Initial state where user chooses between pizza or sushi
#    - Functions:
#      * choose_pizza (transitions to choose_pizza)
#      * choose_sushi (transitions to choose_sushi)
#
# 2. choose_pizza
#    - Handles pizza order details
#    - Functions:
#      * select_pizza_order (node function with size and type)
#      * confirm_order (transitions to confirm)
#    - Pricing:
#      * Small: $10
#      * Medium: $15
#      * Large: $20
#
# 3. choose_sushi
#    - Handles sushi order details
#    - Functions:
#      * select_sushi_order (node function with count and type)
#      * confirm_order (transitions to confirm)
#    - Pricing:
#      * $8 per roll
#
# 4. confirm
#    - Reviews order details with the user
#    - Functions:
#      * complete_order (transitions to end)
#
# 5. end
#    - Final state that closes the conversation
#    - No functions available
#    - Post-action: Ends conversation


# Type definitions
class PizzaOrderResult(FlowResult):
    size: str
    type: str
    price: float


class SushiOrderResult(FlowResult):
    count: int
    type: str
    price: float


# Function handlers
async def select_pizza_order(args: FlowArgs) -> PizzaOrderResult:
    """Handle pizza size and type selection."""
    size = args["size"]
    pizza_type = args["type"]

    # Simple pricing
    base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
    price = base_price[size]

    return {"size": size, "type": pizza_type, "price": price}


async def select_sushi_order(args: FlowArgs) -> SushiOrderResult:
    """Handle sushi roll count and type selection."""
    count = args["count"]
    roll_type = args["type"]

    # Simple pricing: $8 per roll
    price = count * 8.00

    return {"count": count, "type": roll_type, "price": price}


flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.",
                }
            ],
            "task_messages": [
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
                        "description": "User wants to order pizza. Let's get that order started.",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "choose_pizza",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "choose_sushi",
                        "description": "User wants to order sushi. Let's get that order started.",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "choose_sushi",
                    },
                },
            ],
        },
        "choose_pizza": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """You are handling a pizza order. Use the available functions:
- Use select_pizza_order when the user specifies both size AND type
- Use confirm_order when the user confirms they are satisfied with their selection

Pricing:
- Small: $10
- Medium: $15
- Large: $20

After selection, confirm both the size and type, state the price, and ask if they want to confirm their order. Remember to be friendly and casual.""",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_pizza_order",
                        "handler": select_pizza_order,
                        "description": "Record the pizza order details",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {
                                    "type": "string",
                                    "enum": ["small", "medium", "large"],
                                    "description": "Size of the pizza",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["pepperoni", "cheese", "supreme", "vegetarian"],
                                    "description": "Type of pizza",
                                },
                            },
                            "required": ["size", "type"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "confirm_order",
                        "description": "Proceed to order confirmation",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "confirm",
                    },
                },
            ],
        },
        "choose_sushi": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """You are handling a sushi order. Use the available functions:
- Use select_sushi_order when the user specifies both count AND type
- Use confirm_order when the user confirms they are satisfied with their selection

Pricing:
- $8 per roll

After selection, confirm both the count and type, state the price, and ask if they want to confirm their order. Remember to be friendly and casual.""",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_sushi_order",
                        "handler": select_sushi_order,
                        "description": "Record the sushi order details",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Number of rolls to order",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["california", "spicy tuna", "rainbow", "dragon"],
                                    "description": "Type of sushi roll",
                                },
                            },
                            "required": ["count", "type"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "confirm_order",
                        "description": "Proceed to order confirmation",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "confirm",
                    },
                },
            ],
        },
        "confirm": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """Read back the complete order details to the user and ask for final confirmation. Use the available functions:
- Use complete_order when the user confirms
- Use revise_order if they want to change something

Be friendly and clear when reading back the order details.""",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "complete_order",
                        "description": "User confirms the order is correct",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "end",
                    },
                },
            ],
        },
        "end": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Concisely end the conversation—1-3 words is appropriate. Just say 'Bye' or something similarly short.",
                }
            ],
            "functions": [],
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
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")


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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager in static mode
        flow_manager = FlowManager(task=task, llm=llm, tts=tts, flow_config=flow_config)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            logger.debug("Initializing flow")
            await flow_manager.initialize()
            logger.debug("Starting conversation")
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
