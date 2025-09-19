#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A dynamic patient intake flow example for Pipecat Flows.

This example demonstrates a medical intake system using dynamic flows where
conversation paths are determined at runtime. The flow handles:

1. Patient identity verification through birthday
2. Prescription collection
3. Allergy information gathering
4. Medical conditions collection
5. Visit reason documentation
6. Information verification and confirmation

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai (default), anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (optionalfor transport)
- LLM API key (varies by provider - see env.example)
"""

import os
from typing import List, TypedDict

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
    ContextStrategy,
    ContextStrategyConfig,
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
class Prescription(TypedDict):
    medication: str
    dosage: str


class Allergy(TypedDict):
    name: str


class Condition(TypedDict):
    name: str


class VisitReason(TypedDict):
    name: str


# Result types for each handler
class BirthdayVerificationResult(FlowResult):
    verified: bool


class PrescriptionRecordResult(FlowResult):
    count: int


class AllergyRecordResult(FlowResult):
    count: int


class ConditionRecordResult(FlowResult):
    count: int


class VisitReasonRecordResult(FlowResult):
    count: int


# Function handlers
async def verify_birthday(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[BirthdayVerificationResult, NodeConfig]:
    """Handler for birthday verification."""
    birthday = args["birthday"]
    # In a real app, this would verify against patient records
    is_valid = birthday == "1983-01-01"

    # Store verification result in flow state
    flow_manager.state["birthday_verified"] = is_valid
    flow_manager.state["birthday"] = birthday

    return BirthdayVerificationResult(verified=is_valid), create_prescriptions_node()


async def record_prescriptions(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[PrescriptionRecordResult, NodeConfig]:
    """Handler for recording prescriptions."""
    prescriptions: List[Prescription] = args["prescriptions"]

    # Store prescriptions in flow state
    flow_manager.state["prescriptions"] = prescriptions

    # In a real app, this would store in patient records
    return PrescriptionRecordResult(count=len(prescriptions)), create_allergies_node()


async def record_allergies(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[AllergyRecordResult, NodeConfig]:
    """Handler for recording allergies."""
    allergies: List[Allergy] = args["allergies"]

    # Store allergies in flow state
    flow_manager.state["allergies"] = allergies

    # In a real app, this would store in patient records
    return AllergyRecordResult(count=len(allergies)), create_conditions_node()


async def record_conditions(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[ConditionRecordResult, NodeConfig]:
    """Handler for recording medical conditions."""
    conditions: List[Condition] = args["conditions"]

    # Store conditions in flow state
    flow_manager.state["conditions"] = conditions

    # In a real app, this would store in patient records
    return ConditionRecordResult(count=len(conditions)), create_visit_reasons_node()


async def record_visit_reasons(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[VisitReasonRecordResult, NodeConfig]:
    """Handler for recording visit reasons."""
    visit_reasons: List[VisitReason] = args["visit_reasons"]

    # Store visit reasons in flow state
    flow_manager.state["visit_reasons"] = visit_reasons

    # In a real app, this would store in patient records
    return VisitReasonRecordResult(count=len(visit_reasons)), create_verification_node()


async def revise_information(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Handler to restart the information-gathering process."""
    return None, create_prescriptions_node()


async def confirm_information(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Handler to confirm all collected information."""
    return None, create_confirmation_node()


async def complete_intake(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
    """Handler to complete the intake process."""
    return None, create_end_node()


# Node creation functions
def create_initial_node() -> NodeConfig:
    """Create the initial node for patient identity verification."""
    verify_birthday_func = FlowsFunctionSchema(
        name="verify_birthday",
        handler=verify_birthday,
        description="Verify the user has provided their correct birthday. Once confirmed, the next step is to record the user's prescriptions.",
        properties={
            "birthday": {
                "type": "string",
                "description": "The user's birthdate (convert to YYYY-MM-DD format)",
            }
        },
        required=["birthday"],
    )

    return NodeConfig(
        name="start",
        role_messages=[
            {
                "role": "system",
                "content": "You are Jessica, an agent for Tri-County Health Services. You must ALWAYS use one of the available functions to progress the conversation. Be professional but friendly.",
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": "Start by introducing yourself to Chad Bailey, then ask for their date of birth, including the year. Once they provide their birthday, use verify_birthday to check it. If verified (1983-01-01), proceed to prescriptions.",
            }
        ],
        functions=[verify_birthday_func],
    )


def create_prescriptions_node() -> NodeConfig:
    """Create the prescriptions collection node."""
    record_prescriptions_func = FlowsFunctionSchema(
        name="record_prescriptions",
        handler=record_prescriptions,
        description="Record the user's prescriptions. Once confirmed, the next step is to collect allergy information.",
        properties={
            "prescriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "medication": {
                            "type": "string",
                            "description": "The medication's name",
                        },
                        "dosage": {
                            "type": "string",
                            "description": "The prescription's dosage",
                        },
                    },
                    "required": ["medication", "dosage"],
                },
            }
        },
        required=["prescriptions"],
    )

    return NodeConfig(
        name="get_prescriptions",
        role_messages=[
            {
                "role": "system",
                "content": "You are Jessica, an agent for Tri-County Health Services. You must ALWAYS use one of the available functions to progress the conversation. Be professional but friendly.",
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": "This step is for collecting prescriptions. Ask them what prescriptions they're taking, including the dosage. Get to the point by saying 'Thanks for confirming that. First up, what prescriptions are you currently taking, including the dosage for each medication?'. After recording prescriptions (or confirming none), proceed to allergies.",
            }
        ],
        context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
        functions=[record_prescriptions_func],
    )


def create_allergies_node() -> NodeConfig:
    """Create the allergies collection node."""
    record_allergies_func = FlowsFunctionSchema(
        name="record_allergies",
        handler=record_allergies,
        description="Record the user's allergies. Once confirmed, then next step is to collect medical conditions.",
        properties={
            "allergies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "What the user is allergic to",
                        },
                    },
                    "required": ["name"],
                },
            }
        },
        required=["allergies"],
    )

    return NodeConfig(
        name="get_allergies",
        task_messages=[
            {
                "role": "system",
                "content": "Collect allergy information. Ask about any allergies they have. After recording allergies (or confirming none), proceed to medical conditions.",
            }
        ],
        functions=[record_allergies_func],
    )


def create_conditions_node() -> NodeConfig:
    """Create the medical conditions collection node."""
    record_conditions_func = FlowsFunctionSchema(
        name="record_conditions",
        handler=record_conditions,
        description="Record the user's medical conditions. Once confirmed, the next step is to collect visit reasons.",
        properties={
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The user's medical condition",
                        },
                    },
                    "required": ["name"],
                },
            }
        },
        required=["conditions"],
    )

    return NodeConfig(
        name="get_conditions",
        task_messages=[
            {
                "role": "system",
                "content": "Collect medical condition information. Ask about any medical conditions they have. After recording conditions (or confirming none), proceed to visit reasons.",
            }
        ],
        functions=[record_conditions_func],
    )


def create_visit_reasons_node() -> NodeConfig:
    """Create the visit reasons collection node."""
    record_visit_reasons_func = FlowsFunctionSchema(
        name="record_visit_reasons",
        handler=record_visit_reasons,
        description="Record the reasons for their visit. Once confirmed, the next step is to verify all information.",
        properties={
            "visit_reasons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The user's reason for visiting",
                        },
                    },
                    "required": ["name"],
                },
            }
        },
        required=["visit_reasons"],
    )

    return NodeConfig(
        name="get_visit_reasons",
        task_messages=[
            {
                "role": "system",
                "content": "Collect information about the reason for their visit. Ask what brings them to the doctor today. After recording their reasons, proceed to verification.",
            }
        ],
        functions=[record_visit_reasons_func],
    )


def create_verification_node() -> NodeConfig:
    """Create the information verification node with context reset and summary."""
    revise_information_func = FlowsFunctionSchema(
        name="revise_information",
        handler=revise_information,
        description="Return to prescriptions to revise information",
        properties={},
        required=[],
    )

    confirm_information_func = FlowsFunctionSchema(
        name="confirm_information",
        handler=confirm_information,
        description="Proceed with confirmed information",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="verify",
        task_messages=[
            {
                "role": "system",
                "content": """Review all collected information with the patient. Follow these steps:
1. Summarize their prescriptions, allergies, conditions, and visit reasons
2. Ask if everything is correct
3. Use the appropriate function based on their response

Be thorough in reviewing all details and wait for explicit confirmation.""",
            }
        ],
        context_strategy=ContextStrategyConfig(
            strategy=ContextStrategy.RESET_WITH_SUMMARY,
            summary_prompt=(
                "Summarize the patient intake conversation, including their birthday, "
                "prescriptions, allergies, medical conditions, and reasons for visiting. "
                "Focus on the specific medical information provided."
            ),
        ),
        functions=[revise_information_func, confirm_information_func],
    )


def create_confirmation_node() -> NodeConfig:
    """Create the final confirmation node."""
    complete_intake_func = FlowsFunctionSchema(
        name="complete_intake",
        handler=complete_intake,
        description="Complete the intake process",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "system",
                "content": "Once confirmed, thank them, then use the complete_intake function to end the conversation.",
            }
        ],
        functions=[complete_intake_func],
    )


def create_end_node() -> NodeConfig:
    """Create the final end node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "system",
                "content": "Thank them for their time and end the conversation.",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the patient intake bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
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
