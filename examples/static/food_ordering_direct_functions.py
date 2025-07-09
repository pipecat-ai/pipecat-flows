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

from pipecat_flows import FlowConfig, FlowManager, FlowResult

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


# Actions
async def check_kitchen_status(action: dict) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


# Functions
async def choose_pizza(flow_manager: FlowManager) -> tuple[None, str]:
    """
    User wants to order pizza. Let's get that order started.
    """
    return None, "choose_pizza"


async def choose_sushi(flow_manager: FlowManager) -> tuple[None, str]:
    """
    User wants to order sushi. Let's get that order started.
    """
    return None, "choose_sushi"


async def select_pizza_order(
    flow_manager: FlowManager, size: str, pizza_type: str
) -> tuple[PizzaOrderResult, str]:
    """
    Record the pizza order details.

    Args:
        size (str): Size of the pizza. Must be one of "small", "medium", or "large".
        pizza_type (str): Type of pizza. Must be one of "pepperoni", "cheese", "supreme", or "vegetarian".
    """
    # Simple pricing
    base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
    price = base_price[size]

    return {"size": size, "type": pizza_type, "price": price}, "confirm"


async def select_sushi_order(
    flow_manager: FlowManager, count: int, roll_type: str
) -> tuple[SushiOrderResult, str]:
    """
    Record the sushi order details.

    Args:
        count (int): Number of sushi rolls to order. Must be between 1 and 10.
        roll_type (str): Type of sushi roll. Must be one of "california", "spicy tuna", "rainbow", or "dragon".
    """
    # Simple pricing: $8 per roll
    price = count * 8.00

    return {"count": count, "type": roll_type, "price": price}, "confirm"


async def complete_order(flow_manager: FlowManager) -> tuple[None, str]:
    """
    User confirms the order is correct.
    """
    return None, "end"


async def revise_order(flow_manager: FlowManager) -> tuple[None, str]:
    """
    User wants to make changes to their order.
    """
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
            "functions": [choose_pizza, choose_sushi],
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
            "functions": [select_pizza_order],
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
            "functions": [select_sushi_order],
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
            "functions": [complete_order, revise_order],
        },
        "end": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Thank the user for their order and end the conversation politely and concisely.",
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
