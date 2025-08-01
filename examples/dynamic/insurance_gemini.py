#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Insurance Quote Example using Pipecat Dynamic Flows with Google Gemini.

This example demonstrates how to create a conversational insurance quote bot using:
- Dynamic flow management for flexible conversation paths
- Google Gemini for natural language understanding
- Simple function handlers for processing user input
- Node configurations for different conversation states
- Pre/post actions for user feedback

The flow allows users to:
1. Provide their age
2. Specify marital status
3. Get an insurance quote
4. Adjust coverage options
5. Complete the quote process

Requirements:
- Daily room URL
- Google API key
- Deepgram API key
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import TypedDict, Union

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowManager, FlowResult, FlowsFunctionSchema, NodeConfig

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Type definitions
class InsuranceQuote(TypedDict):
    monthly_premium: float
    coverage_amount: int
    deductible: int


class AgeCollectionResult(FlowResult):
    age: int


class MaritalStatusResult(FlowResult):
    marital_status: str


class QuoteCalculationResult(FlowResult, InsuranceQuote):
    pass


class CoverageUpdateResult(FlowResult, InsuranceQuote):
    pass


# Simulated insurance data
INSURANCE_RATES = {
    "young_single": {"base_rate": 150, "risk_multiplier": 1.5},
    "young_married": {"base_rate": 130, "risk_multiplier": 1.3},
    "adult_single": {"base_rate": 100, "risk_multiplier": 1.0},
    "adult_married": {"base_rate": 90, "risk_multiplier": 0.9},
}


# Function handlers
async def collect_age(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[AgeCollectionResult, NodeConfig]:
    """Process age collection."""
    age = args["age"]
    logger.debug(f"collect_age handler executing with age: {age}")

    flow_manager.state["age"] = age
    result = AgeCollectionResult(age=age)

    next_node = create_marital_status_node()

    return result, next_node


async def collect_marital_status(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[MaritalStatusResult, NodeConfig]:
    """Process marital status collection."""
    status = args["marital_status"]
    logger.debug(f"collect_marital_status handler executing with status: {status}")

    result = MaritalStatusResult(marital_status=status)

    next_node = create_quote_calculation_node(flow_manager.state["age"], status)

    return result, next_node


async def calculate_quote(args: FlowArgs) -> tuple[QuoteCalculationResult, NodeConfig]:
    """Calculate insurance quote based on age and marital status."""
    age = args["age"]
    marital_status = args["marital_status"]
    logger.debug(f"calculate_quote handler executing with age: {age}, status: {marital_status}")

    # Determine rate category
    age_category = "young" if age < 25 else "adult"
    rate_key = f"{age_category}_{marital_status}"
    rates = INSURANCE_RATES.get(rate_key, INSURANCE_RATES["adult_single"])

    # Calculate quote
    monthly_premium = rates["base_rate"] * rates["risk_multiplier"]

    result = QuoteCalculationResult(
        monthly_premium=monthly_premium,
        coverage_amount=250000,
        deductible=1000,
    )
    next_node = create_quote_results_node(result)
    return result, next_node


async def update_coverage(args: FlowArgs) -> tuple[CoverageUpdateResult, NodeConfig]:
    """Update coverage options and recalculate premium."""
    coverage_amount = args["coverage_amount"]
    deductible = args["deductible"]
    logger.debug(
        f"update_coverage handler executing with amount: {coverage_amount}, deductible: {deductible}"
    )

    # Calculate adjusted quote
    monthly_premium = (coverage_amount / 250000) * 100
    if deductible > 1000:
        monthly_premium *= 0.9  # 10% discount for higher deductible

    result = CoverageUpdateResult(
        monthly_premium=monthly_premium,
        coverage_amount=coverage_amount,
        deductible=deductible,
    )
    next_node = create_quote_results_node(result)
    return result, next_node


async def end_quote(args: FlowArgs) -> tuple[FlowResult, NodeConfig]:
    """Handle quote completion."""
    logger.debug("end_quote handler executing")
    result = {"status": "completed"}
    next_node = create_end_node()
    return result, next_node


# Node configurations using FlowsFunctionSchema
def create_initial_node() -> NodeConfig:
    """Create the initial node asking for age."""
    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly insurance agent. Your responses will be "
                    "converted to audio, so avoid special characters. "
                    "Always wait for customer responses before calling functions. "
                    "Only call functions after receiving relevant information from the customer."
                ),
            }
        ],
        "task_messages": [
            {
                "role": "user",
                "content": "Start by asking for the customer's age.",
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="collect_age",
                description="Record customer's age",
                properties={"age": {"type": "integer"}},
                required=["age"],
                handler=collect_age,
            )
        ],
    }


def create_marital_status_node() -> NodeConfig:
    """Create node for collecting marital status."""
    return {
        "name": "marital_status",
        "task_messages": [
            {
                "role": "user",
                "content": (
                    "Ask about the customer's marital status (single or married). "
                    "Wait for their response before calling collect_marital_status. "
                    "Only call the function after they provide their status."
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="collect_marital_status",
                description="Record marital status after customer provides it",
                properties={
                    "marital_status": {"type": "string", "enum": ["single", "married"]},
                },
                required=["marital_status"],
                handler=collect_marital_status,
            )
        ],
    }


def create_quote_calculation_node(age: int, marital_status: str) -> NodeConfig:
    """Create node for calculating initial quote."""
    return {
        "name": "quote_calculation",
        "task_messages": [
            {
                "role": "user",
                "content": (
                    f"Calculate a quote for {age} year old {marital_status} customer. "
                    "Call calculate_quote with their information. "
                    "After receiving the quote, explain the details and ask if they'd like to adjust coverage."
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="calculate_quote",
                description="Calculate initial insurance quote",
                properties={
                    "age": {"type": "integer"},
                    "marital_status": {"type": "string", "enum": ["single", "married"]},
                },
                required=["age", "marital_status"],
                handler=calculate_quote,
            )
        ],
    }


def create_quote_results_node(
    quote: Union[QuoteCalculationResult, CoverageUpdateResult],
) -> NodeConfig:
    """Create node for showing quote and adjustment options."""
    return {
        "name": "quote_results",
        "task_messages": [
            {
                "role": "user",
                "content": (
                    f"Quote details:\n"
                    f"Monthly Premium: ${quote['monthly_premium']:.2f}\n"
                    f"Coverage Amount: ${quote['coverage_amount']:,}\n"
                    f"Deductible: ${quote['deductible']:,}\n\n"
                    "Explain these quote details to the customer. When they request changes, "
                    "use update_coverage to recalculate their quote. Explain how their "
                    "changes affected the premium and compare it to their previous quote. "
                    "Ask if they'd like to make any other adjustments or if they're ready "
                    "to end the quote process."
                ),
            }
        ],
        "functions": [
            FlowsFunctionSchema(
                name="update_coverage",
                description="Recalculate quote with new coverage options",
                properties={
                    "coverage_amount": {"type": "integer"},
                    "deductible": {"type": "integer"},
                },
                required=["coverage_amount", "deductible"],
                handler=update_coverage,
            ),
            FlowsFunctionSchema(
                name="end_quote",
                description="Complete the quote process when customer is satisfied",
                properties={},
                required=[],
                handler=end_quote,
            ),
        ],
    }


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "user",
                "content": (
                    "Thank the customer for their time and end the conversation. "
                    "Mention that a representative will contact them about the quote."
                ),
            }
        ],
        "post_actions": [{"type": "end_conversation"}],
    }


async def main():
    """Main function to set up and run the insurance quote bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        # Initialize services
        transport = DailyTransport(
            room_url,
            None,
            "Insurance Quote Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")
        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-exp")

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

        # Initialize flow manager with transition callback
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Initialize flow
            await flow_manager.initialize(create_initial_node())

        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
