#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Routing between Flow nodes with a classifier instead of the LLM.

Some node decisions are trivial: a verification step that only needs a yes/no,
a menu where the user picks one of a few known options. Paying for an LLM
inference to make that decision adds latency and cost you don't need. This
example shows how to route those decisions with a plain classifier (here a
regex/keyword match, but swap in whatever you like) and reserve the LLM only
for inputs the classifier can't confidently handle.

How it works:

- The "verify" node uses ``respond_immediately=False`` so entering it does NOT
  trigger an LLM completion. It speaks a canned question via a ``tts_say``
  pre-action and then waits for the user.
- In a normal pipeline, when the user finishes their turn the context
  aggregator emits an ``LLMContextFrame`` that the LLM service consumes and
  responds to. That is the LLM call we want to skip.
- ``ClassifierRouter`` is a custom FrameProcessor placed BETWEEN the user
  context aggregator and the LLM. When we're in the "verify" node it inspects
  the user's transcription and either:
    * routes deterministically (yes -> confirmed, no -> declined) and DROPS the
      ``LLMContextFrame`` so the LLM never runs, or
    * passes the frame through to the LLM when it can't classify the input,
      letting the LLM improvise an answer (the "off the rails" follow-up case).

The terminal nodes ("confirmed"/"declined") also avoid the LLM: they speak a
canned line and end the call using an ``end_conversation`` pre-action.

Requirements:
- CARTESIA_API_KEY
- GOOGLE_API_KEY

Run the example:
uv run classifier_routing.py
"""

import os
import re

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

from pipecat_flows import FlowManager, NodeConfig

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


# The classifier. This is the seam to replace with regex, a rules engine, or a
# small binary classifier model. It returns "yes", "no", or None ("not sure,
# let the LLM handle it").
# NOTE: these toy patterns are illustrative; real STT output (typographic
# apostrophes, dropped apostrophes, hedges like "I don't know") will fool them,
# so use a real classifier in production.
YES_PATTERN = re.compile(r"\b(yes|yeah|yep|yup|sure|correct|i am|i do)\b", re.IGNORECASE)
NO_PATTERN = re.compile(r"\b(no|nope|nah|i'm not|i am not|i don't)\b", re.IGNORECASE)


def classify(text: str) -> str | None:
    """Classify an utterance as a yes/no, or None if it isn't a clear answer."""
    # Check "no" first so "I'm not" isn't shadowed by a stray "i" match.
    if NO_PATTERN.search(text):
        return "no"
    if YES_PATTERN.search(text):
        return "yes"
    return None


def _latest_user_text(context: LLMContext) -> str:
    """Pull the most recent user message text out of the LLM context."""
    for message in reversed(context.messages):
        # Skip provider-specific (non-dict) messages; user turns are standard dicts.
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            # Some providers use a list of content parts; join the text parts.
            return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class ClassifierRouter(FrameProcessor):
    """Routes the "verify" node deterministically, skipping the LLM when it can.

    Placed between the user context aggregator and the LLM. For the "verify"
    node it classifies the user's answer and, on a confident yes/no, transitions
    to the matching node WITHOUT forwarding the ``LLMContextFrame`` (so the LLM
    never runs). Anything it can't classify is passed through to the LLM.
    """

    def __init__(self):
        super().__init__()
        # Set after the FlowManager is constructed in run_bot().
        self.flow_manager: FlowManager | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Only intercept the turn-end context frame while we're in the node
        # whose decision we want to make without the LLM.
        if (
            isinstance(frame, LLMContextFrame)
            and self.flow_manager is not None
            and self.flow_manager.current_node == "verify"
        ):
            answer = classify(_latest_user_text(frame.context))
            if answer == "yes":
                logger.info("Classifier routed 'yes' -> confirmed (no LLM call)")
                await self.flow_manager.set_node_from_config(create_confirmed_node())
                return  # Drop the frame: the LLM does not run for this turn.
            if answer == "no":
                logger.info("Classifier routed 'no' -> declined (no LLM call)")
                await self.flow_manager.set_node_from_config(create_declined_node())
                return  # Drop the frame: the LLM does not run for this turn.
            logger.info("Classifier unsure -> forwarding to the LLM")
            # Fall through to push the frame so the LLM can handle the follow-up.

        # SUPER IMPORTANT: always push every other frame.
        await self.push_frame(frame, direction)


# Flow nodes
def create_verify_node() -> NodeConfig:
    """Ask a yes/no question, then wait for the user without invoking the LLM.

    ``respond_immediately=False`` means no LLM completion runs when we enter the
    node; the canned question is spoken by the ``tts_say`` pre-action. The
    ``role_message``/``task_messages`` only matter on the LLM fallback path,
    when the classifier can't make sense of the user's reply.
    """
    return NodeConfig(
        name="verify",
        role_message=(
            "You are a friendly age-verification assistant. Keep replies to one "
            "short sentence. Your responses will be converted to audio, so avoid "
            "special characters and emojis."
        ),
        task_messages=[
            {
                "role": "developer",
                "content": (
                    "The user replied with something other than a clear yes or no. "
                    "Briefly and politely answer what they said, then ask again "
                    "whether they are over 18."
                ),
            }
        ],
        respond_immediately=False,
        pre_actions=[{"type": "tts_say", "text": "Are you over 18? Please say yes or no."}],
    )


def create_confirmed_node() -> NodeConfig:
    """Canned success: speak a line and end the call, no LLM involved."""
    return NodeConfig(
        name="confirmed",
        task_messages=[],
        respond_immediately=False,
        pre_actions=[{"type": "end_conversation", "text": "Great, you're verified. Goodbye!"}],
    )


def create_declined_node() -> NodeConfig:
    """Canned decline: speak a line and end the call, no LLM involved."""
    return NodeConfig(
        name="declined",
        task_messages=[],
        respond_immediately=False,
        pre_actions=[
            {
                "type": "end_conversation",
                "text": "No problem, you must be 18 or older. Goodbye!",
            }
        ],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY", ""))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        settings=CartesiaTTSService.Settings(
            voice="32b3f3c5-7171-46aa-abe7-b598964aa793",
        ),
    )
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY", ""))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            filter_incomplete_user_turns=True,
        ),
    )

    # Sits between the user aggregator and the LLM so it can decide, per turn,
    # whether the LLM runs at all.
    router = ClassifierRouter()

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            router,  # Classifier-based routing (may skip the LLM)
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
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

    # Initialize flow manager
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    # Give the router a handle to the flow manager so it can trigger transitions.
    router.flow_manager = flow_manager

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await flow_manager.initialize(create_verify_node())

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
