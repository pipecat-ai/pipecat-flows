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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult
from pipecat_flows.types import FlowsFunctionSchema

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
async def check_kitchen_status(action: dict) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


async def select_pizza_order(args: FlowArgs) -> tuple[PizzaOrderResult, str]:
    """Handle pizza size and type selection."""
    size = args["size"]
    pizza_type = args["type"]

    # Simple pricing
    base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
    price = base_price[size]

    result = {"size": size, "type": pizza_type, "price": price}
    return result, "confirm"


async def select_sushi_order(args: FlowArgs) -> tuple[SushiOrderResult, str]:
    """Handle sushi roll count and type selection."""
    count = args["count"]
    roll_type = args["type"]

    # Simple pricing: $8 per roll
    price = count * 8.00

    result = {"count": count, "type": roll_type, "price": price}
    return result, "confirm"


async def choose_pizza() -> tuple[None, str]:
    """Transition to pizza order selection."""
    return None, "choose_pizza"


async def choose_sushi() -> tuple[None, str]:
    """Transition to sushi order selection."""
    return None, "choose_sushi"


async def complete_order() -> tuple[None, str]:
    """Transition to end state."""
    return None, "end"


async def revise_order() -> tuple[None, str]:
    """Transition to start for order revision."""
    return None, "start"


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
            "pre_actions": [
                {
                    "type": "check_kitchen",
                    "handler": check_kitchen_status,
                },
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="choose_pizza",
                    handler=choose_pizza,
                    description="User wants to order pizza. Let's get that order started.",
                    properties={},
                    required=[],
                ),
                FlowsFunctionSchema(
                    name="choose_sushi",
                    handler=choose_sushi,
                    description="User wants to order sushi. Let's get that order started.",
                    properties={},
                    required=[],
                ),
            ],
        },
        "choose_pizza": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """You are handling a pizza order. Use the available functions:
- Use select_pizza_order when the user specifies both size AND type

Pricing:
- Small: $10
- Medium: $15
- Large: $20

Remember to be friendly and casual.""",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="select_pizza_order",
                    handler=select_pizza_order,
                    description="Record the pizza order details",
                    properties={
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
                    required=["size", "type"],
                ),
            ],
        },
        "choose_sushi": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """You are handling a sushi order. Use the available functions:
- Use select_sushi_order when the user specifies both count AND type

Pricing:
- $8 per roll

Remember to be friendly and casual.""",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="select_sushi_order",
                    handler=select_sushi_order,
                    description="Record the sushi order details",
                    properties={
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
                    required=["count", "type"],
                ),
            ],
        },
        "confirm": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """Read back the complete order details to the user and if they want anything else or if they want to make changes. Use the available functions:
- Use complete_order when the user confirms that the order is correct and no changes are needed
- Use revise_order if they want to change something

Be friendly and clear when reading back the order details.""",
                }
            ],
            "functions": [
                FlowsFunctionSchema(
                    name="complete_order",
                    handler=complete_order,
                    description="User confirms the order is correct",
                    properties={},
                    required=[],
                ),
                FlowsFunctionSchema(
                    name="revise_order",
                    handler=revise_order,
                    description="User wants to make changes to their order",
                    properties={},
                    required=[],
                ),
            ],
        },
        "end": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Thank the user for their order and end the conversation politely and concisely.",
                }
            ],
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
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        context = LLMContext()
        context_aggregator = LLMContextAggregatorPair(context)

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

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        # Initialize flow manager in static mode
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            flow_config=flow_config,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            logger.debug("Initializing flow")
            await flow_manager.initialize()

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
