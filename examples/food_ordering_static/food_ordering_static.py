#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A food ordering flow example using static JSON configuration.

This example demonstrates the same food ordering system as food_ordering.py,
but with the flow graph defined in a static JSON file. Custom computation
(pricing, state updates) still happens in Python handler functions.

JSONFlowConfig (in json_flow_config.py) bridges JSON and Pipecat Flows:
- Simple transitions (choose_pizza, choose_sushi) are auto-generated from JSON
- Functions with custom logic (select_pizza_order) use registered Python handlers
- The JSON file defines the complete flow graph without any Python code

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai_responses (default), openai, anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Add parent directory to path so we can import the shared utils module
sys.path.insert(0, str(Path(__file__).parent.parent))
from json_flow_config import JSONFlowConfig  # noqa: E402
from utils import create_llm  # noqa: E402

from pipecat_flows import FlowArgs, FlowManager, FlowResult

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# --- Handler functions ---


async def select_pizza_order(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Handle pizza size and type selection."""
    size = args["size"]
    pizza_type = args["type"]

    # Simple pricing
    base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
    price = base_price[size]

    # Store order details in flow state
    flow_manager.state["order"] = {
        "type": "pizza",
        "size": size,
        "pizza_type": pizza_type,
        "price": price,
    }

    return {"size": size, "type": pizza_type, "price": price}


async def select_sushi_order(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Handle sushi roll count and type selection."""
    count = args["count"]
    roll_type = args["type"]

    # Simple pricing: $8 per roll
    price = count * 8.00

    # Store order details in flow state
    flow_manager.state["order"] = {
        "type": "sushi",
        "count": count,
        "roll_type": roll_type,
        "price": price,
    }

    return {"count": count, "type": roll_type, "price": price}


async def get_delivery_estimate(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
    """Provide delivery estimate information."""
    delivery_time = datetime.now() + timedelta(minutes=30)
    return {"time": str(delivery_time)}


# Pre-action handler


async def check_kitchen_status(action: dict, flow_manager: FlowManager) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


# --- Pipeline setup ---


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the food ordering bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        ),
    )
    # LLM service is created using the create_llm function from utils.py
    # Default is OpenAI; can be changed by setting LLM_PROVIDER environment variable
    llm = create_llm()

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=True,
        ),
    )

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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Load flow config from JSON. Handler strings in the JSON are resolved
    # against the module namespace -- function names just need to match.
    flow_config = JSONFlowConfig(
        Path(__file__).parent / "food_ordering_static.json",
        globals(),
    )

    # Initialize flow manager with global functions from config
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        global_functions=flow_config.create_global_functions(),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation with the initial node
        await flow_manager.initialize(flow_config.create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
