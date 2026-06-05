#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Validate how programmatic TTS output lands in the LLM context.

This example exists to exercise (and let you inspect) the new
``append_text_to_context`` key on the built-in TTS actions. It walks a caller
through booking a dental appointment, deliberately mixing LLM-generated speech
with programmatic ``tts_say`` output and an ``end_conversation`` goodbye:

- ``greeting`` node: a ``tts_say`` *pre-action* speaks a recording disclaimer
  before the LLM greets the caller.
- ``scheduling`` node: a ``tts_say`` *pre-action* speaks a transitional line
  before the LLM asks for a day and time.
- ``confirmation`` node: a ``tts_say`` *post-action* speaks a reminder note
  after the LLM confirms the booking, then an ``end_conversation`` action
  speaks a goodbye and ends the call.

When the conversation ends, the full context is printed so you can verify that
the LLM and programmatic-TTS messages landed in the right order and that the
``append_text_to_context`` setting was honored.

Toggle the behavior from the command line::

    # Force programmatic TTS into the context
    uv run examples/tts_context_control.py --append-text-to-context true

    # Keep programmatic TTS out of the context
    uv run examples/tts_context_control.py --append-text-to-context false

    # Omit the key entirely and let the pipeline default apply (the default)
    uv run examples/tts_context_control.py --append-text-to-context unset

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai_responses (default), openai, anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import json
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
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
from pipecat.workers.runner import WorkerRunner
from utils import create_llm

from pipecat_flows import ActionConfig, FlowArgs, FlowManager, FlowsFunctionSchema, NodeConfig

load_dotenv(override=True)

# How the example sets "append_text_to_context" on its TTS actions. One of:
# - "true"/"false": set the key explicitly to that boolean
# - "unset": omit the key entirely so the pipeline's default behavior applies
#
# Overridden from the command line in __main__; defaults to "unset" so that
# importing this module (e.g. on Pipecat Cloud) keeps the out-of-the-box
# behavior.
APPEND_MODE = "unset"

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


# Action builders
def _maybe_append_text_to_context(action: ActionConfig) -> ActionConfig:
    """Apply the CLI-selected append_text_to_context setting to an action.

    When APPEND_MODE is "unset", the key is left off so the pipeline default
    applies; otherwise it's set to the corresponding boolean.
    """
    if APPEND_MODE != "unset":
        action["append_text_to_context"] = APPEND_MODE == "true"
    print("[pk] action: ", action)
    return action


def tts_say_action(text: str) -> ActionConfig:
    """Build a tts_say action honoring the CLI append_text_to_context setting."""
    return _maybe_append_text_to_context(ActionConfig(type="tts_say", text=text))


def end_conversation_action(text: str) -> ActionConfig:
    """Build an end_conversation action honoring the CLI append_text_to_context setting."""
    return _maybe_append_text_to_context(ActionConfig(type="end_conversation", text=text))


# Function handlers
async def record_name(args: FlowArgs, flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
    """Record the caller's name and move on to collecting the visit reason."""
    flow_manager.state["name"] = args["name"]
    return {"name": args["name"]}, create_visit_reason_node()


async def record_reason(args: FlowArgs, flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
    """Record the reason for the visit and move on to scheduling."""
    flow_manager.state["reason"] = args["reason"]
    return {"reason": args["reason"]}, create_scheduling_node()


async def book_slot(args: FlowArgs, flow_manager: FlowManager) -> tuple[dict, NodeConfig]:
    """Record the requested day and time, then confirm the appointment."""
    flow_manager.state["day"] = args["day"]
    flow_manager.state["time"] = args["time"]
    summary = (
        f"{flow_manager.state.get('name', 'the patient')} for a "
        f"{flow_manager.state.get('reason', 'visit')} on {args['day']} at {args['time']}"
    )
    return {"status": "booked"}, create_confirmation_node(summary)


# Function schemas
record_name_schema = FlowsFunctionSchema(
    name="record_name",
    description="Record the caller's full name",
    properties={"name": {"type": "string", "description": "The caller's full name"}},
    required=["name"],
    handler=record_name,
)

record_reason_schema = FlowsFunctionSchema(
    name="record_reason",
    description="Record the reason for the visit",
    properties={
        "reason": {
            "type": "string",
            "enum": ["cleaning", "checkup", "concern"],
            "description": "The type of appointment the caller wants",
        }
    },
    required=["reason"],
    handler=record_reason,
)

book_slot_schema = FlowsFunctionSchema(
    name="book_slot",
    description="Book the requested appointment day and time",
    properties={
        "day": {"type": "string", "description": "The requested day, e.g. 'Tuesday'"},
        "time": {"type": "string", "description": "The requested time, e.g. '3:00 PM'"},
    },
    required=["day", "time"],
    handler=book_slot,
)


# Node configurations
def create_greeting_node() -> NodeConfig:
    """Create the initial node: disclaimer pre-action, then greet and get a name."""
    return NodeConfig(
        name="greeting",
        role_message=(
            "You are a friendly scheduling assistant for Bright Smile Dental. "
            "Keep replies short and warm. This is a voice conversation, so avoid "
            "special characters and emojis."
        ),
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Greet the caller, briefly introduce yourself as the Bright Smile Dental "
                    "scheduling assistant, and ask for their full name. Do not mention recording; "
                    "that has already been handled."
                ),
            }
        ],
        functions=[record_name_schema],
        # tts_say PRE-action: a fixed compliance announcement that should be
        # spoken (and, optionally, recorded in context) before the LLM speaks.
        pre_actions=[
            tts_say_action(
                "Thanks for calling Bright Smile Dental. "
                "Please note this call may be recorded for quality purposes."
            )
        ],
    )


def create_visit_reason_node() -> NodeConfig:
    """Create the node that collects the reason for the visit."""
    return NodeConfig(
        name="visit_reason",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Ask whether they'd like to book a cleaning, a checkup, or to discuss a "
                    "specific concern."
                ),
            }
        ],
        functions=[record_reason_schema],
    )


def create_scheduling_node() -> NodeConfig:
    """Create the scheduling node: transitional pre-action, then ask for a day/time."""
    return NodeConfig(
        name="scheduling",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "Ask which day and time works best for them. The office is open "
                    "Monday through Friday, 9 AM to 5 PM."
                ),
            }
        ],
        functions=[book_slot_schema],
        # tts_say PRE-action: a fixed transitional line spoken before the LLM's
        # dynamic question.
        pre_actions=[tts_say_action("Great, let's get you scheduled.")],
    )


def create_confirmation_node(summary: str) -> NodeConfig:
    """Create the terminal node: confirm, then a tts_say post-action and goodbye."""
    return NodeConfig(
        name="confirmation",
        task_messages=[
            {
                "role": "developer",
                "content": (
                    f"Confirm the appointment back to the caller in one friendly sentence: "
                    f"{summary}. Do not say goodbye yet."
                ),
            }
        ],
        # No functions: this node just confirms and then runs its post-actions.
        post_actions=[
            # tts_say POST-action: a fixed reminder spoken after the LLM's
            # dynamic confirmation.
            tts_say_action(
                "You'll receive a text message reminder twenty-four hours before your appointment."
            ),
            # end_conversation with a spoken goodbye, then end the call.
            end_conversation_action("Thanks for choosing Bright Smile Dental. Goodbye!"),
        ],
    )


def _print_final_context(flow_manager: FlowManager) -> None:
    """Print the conversation context after the flow ends.

    This is the point of the example: inspect exactly which messages --
    LLM-generated and programmatic TTS -- landed in the context, and in what
    order, for the chosen append_text_to_context mode.
    """
    try:
        messages = flow_manager.get_current_context()
    except Exception as e:
        logger.error(f"Could not read final context: {e}")
        return

    print("\n" + "=" * 72)
    print(f"FINAL CONTEXT  (append_text_to_context mode: {APPEND_MODE})")
    print(f"{len(messages)} message(s)")
    print("=" * 72)
    print(json.dumps(messages, indent=2, default=str))
    print("=" * 72 + "\n")


# Main setup
async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the appointment-scheduling bot, then print the resulting context."""
    logger.info(f"Running with append_text_to_context mode: {APPEND_MODE}")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
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

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Initialize flow manager
    flow_manager = FlowManager(
        worker=worker,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the conversation with the greeting node
        await flow_manager.initialize(create_greeting_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()

    # Conversation has ended (via end_conversation or disconnect). Dump the
    # final context so the ordering of LLM and programmatic-TTS messages, and
    # whether the programmatic ones were recorded, can be inspected.
    _print_final_context(flow_manager)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    import argparse
    import sys

    # Parse our custom argument first, leaving the rest for the Pipecat runner.
    parser = argparse.ArgumentParser(description="TTS-to-context control example")
    parser.add_argument(
        "--append-text-to-context",
        choices=["true", "false", "unset"],
        default="unset",
        help=(
            "How to set the 'append_text_to_context' key on this example's tts_say and "
            "end_conversation actions: 'true'/'false' set it explicitly, 'unset' (default) "
            "omits the key so the pipeline's default behavior applies."
        ),
    )
    args, _ = parser.parse_known_args()

    # Store the selection where the node/action builders can read it.
    APPEND_MODE = args.append_text_to_context

    # Remove our custom arg (in either "--flag value" or "--flag=value" form) so
    # the Pipecat runner only sees its own arguments.
    cleaned = []
    skip_next = False
    for token in sys.argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--append-text-to-context":
            skip_next = True
            continue
        if token.startswith("--append-text-to-context="):
            continue
        cleaned.append(token)
    sys.argv = cleaned

    # Now run the standard runner
    from pipecat.runner.run import main

    main()
