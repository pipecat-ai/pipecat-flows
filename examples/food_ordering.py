#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A dynamic food ordering flow example for Pipecat Flows.

This example demonstrates a food ordering system using dynamic flows where
conversation paths are determined at runtime. The flow handles:

1. Initial greeting and food type selection (pizza or sushi)
2. Order details collection based on food type
3. Order confirmation and revision
4. Order completion

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai (default), anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from utils import create_llm

from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowResult,
    FlowsFunctionSchema,
    NodeConfig,
)

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


# Type definitions
class PizzaOrderResult(FlowResult):
    size: str
    type: str
    price: float


class SushiOrderResult(FlowResult):
    count: int
    type: str
    price: float


# Pre-action handlers
async def check_kitchen_status(action: dict, flow_manager: FlowManager) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


# Node creation functions
def create_initial_node() -> NodeConfig:
    """Create the initial node for food type selection."""

    async def choose_pizza(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to pizza order selection."""
        return None, create_pizza_node()

    async def choose_sushi(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to sushi order selection."""
        return None, create_sushi_node()

    choose_pizza_func = FlowsFunctionSchema(
        name="choose_pizza",
        handler=choose_pizza,
        description="User wants to order pizza. Let's get that order started.",
        properties={},
        required=[],
    )

    choose_sushi_func = FlowsFunctionSchema(
        name="choose_sushi",
        handler=choose_sushi,
        description="User wants to order sushi. Let's get that order started.",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="initial",
        role_messages=[
            {
                "role": "system",
                "content": "You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.",
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": "For this step, ask the user if they want pizza or sushi, and wait for them to use a function to choose. Start off by greeting them. Be friendly and casual; you're taking an order for food over the phone.",
            }
        ],
        pre_actions=[
            {
                "type": "function",
                "handler": check_kitchen_status,
            },
        ],
        functions=[choose_pizza_func, choose_sushi_func],
    )


def create_pizza_node() -> NodeConfig:
    """Create the pizza ordering node."""

    async def select_pizza_order(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[PizzaOrderResult, NodeConfig]:
        """Handle pizza size and type selection."""
        size = args["size"]
        pizza_type = args["type"]

        # Simple pricing
        base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
        price = base_price[size]

        result = PizzaOrderResult(size=size, type=pizza_type, price=price)

        # Store order details in flow state
        flow_manager.state["order"] = {
            "type": "pizza",
            "size": size,
            "pizza_type": pizza_type,
            "price": price,
        }

        return result, create_confirmation_node()

    select_pizza_func = FlowsFunctionSchema(
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
    )

    return NodeConfig(
        name="choose_pizza",
        task_messages=[
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
        functions=[select_pizza_func],
    )


def create_sushi_node() -> NodeConfig:
    """Create the sushi ordering node."""

    async def select_sushi_order(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[SushiOrderResult, NodeConfig]:
        """Handle sushi roll count and type selection."""
        count = args["count"]
        roll_type = args["type"]

        # Simple pricing: $8 per roll
        price = count * 8.00

        result = SushiOrderResult(count=count, type=roll_type, price=price)

        # Store order details in flow state
        flow_manager.state["order"] = {
            "type": "sushi",
            "count": count,
            "roll_type": roll_type,
            "price": price,
        }

        return result, create_confirmation_node()

    select_sushi_func = FlowsFunctionSchema(
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
    )

    return NodeConfig(
        name="choose_sushi",
        task_messages=[
            {
                "role": "system",
                "content": """You are handling a sushi order. Use the available functions:
- Use select_sushi_order when the user specifies both count AND type

Pricing:
- $8 per roll

Remember to be friendly and casual.""",
            }
        ],
        functions=[select_sushi_func],
    )


def create_confirmation_node() -> NodeConfig:
    """Create the order confirmation node."""

    async def complete_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to end state."""
        return None, create_end_node()

    async def revise_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to start for order revision."""
        return None, create_initial_node()

    complete_order_func = FlowsFunctionSchema(
        name="complete_order",
        handler=complete_order,
        description="User confirms the order is correct",
        properties={},
        required=[],
    )

    revise_order_func = FlowsFunctionSchema(
        name="revise_order",
        handler=revise_order,
        description="User wants to make changes to their order",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "system",
                "content": """Read back the complete order details to the user and ask if they want anything else or if they want to make changes. Use the available functions:
- Use complete_order when the user confirms that the order is correct and no changes are needed
- Use revise_order if they want to change something

Be friendly and clear when reading back the order details.""",
            }
        ],
        functions=[complete_order_func, revise_order_func],
    )


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "system",
                "content": "Thank the user for their order and end the conversation politely and concisely.",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the food ordering bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        text_filters=[MarkdownTextFilter()],
    )
    # LLM service is created using the create_llm function from utils.py
    # Default is OpenAI; can be changed by setting LLM_PROVIDER environment variable
    llm = create_llm()

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

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

    # Initialize flow manager in dynamic mode
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation with the initial node
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
