#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Podcast Interview Example.

This example demonstrates a podcast interview flow using Pipecat Flows.

The flow handles:
1. Introduction and guest introduction
2. Topic selection
3. Interview with multiple questions and follow-ups
4. Conclusion and wrap-up
5. Final thank you

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

# logger.info("Loading Local Smart Turn Analyzer V3...")
# from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

# logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from utils import create_llm, needs_stt_tts

from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowResult,
    FlowsFunctionSchema,
    NodeConfig,
)

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


# Type definitions
class ProceedToTopicResult(FlowResult):
    """Result type for proceed_to_topic function"""

    guest_summary: str


class StartInterviewResult(FlowResult):
    """Result type for start_interview function"""

    topic: str


def create_introduction_node() -> NodeConfig:
    """Create the Introduction node."""

    async def handle_proceed_to_topic(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[ProceedToTopicResult | None, NodeConfig]:
        """Handler for proceed_to_topic function"""
        guest_summary: str = args.get("guest_summary", "")
        return ProceedToTopicResult(guest_summary=guest_summary), create_topic_node()

    proceed_to_topic_func = FlowsFunctionSchema(
        name="proceed_to_topic",
        handler=handle_proceed_to_topic,
        description="Use after the guest has introduced themselves.",
        properties={
            "guest_summary": {
                "type": "string",
                "description": "A quick summary of who the guest is (name, role, area of expertise, etc.)",
            }
        },
        required=["guest_summary"],
    )
    return NodeConfig(
        name="introduction",
        role_messages=[
            {
                "role": "system",
                "content": "You are a warm, engaging podcast host with a natural conversational style. You're genuinely curious about your guests and skilled at making them feel comfortable while drawing out interesting insights. Your questions flow naturally, and you listen actively, building on what your guest shares.",
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": "Welcome the guest warmly and enthusiastically. Focus this exchange on getting to know who they are. Invite them to briefly introduce themselves—name, role, current focus, or anything fun they'd like to share. Ask one follow-up question if it helps clarify or highlight something interesting about them. Once you feel you have a clear introduction, use the proceed_to_topic function to move into topic selection.",
            }
        ],
        functions=[proceed_to_topic_func],
    )


def create_topic_node() -> NodeConfig:
    """Create the Topic Selection node."""

    async def handle_start_interview(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[StartInterviewResult | None, NodeConfig]:
        """Handler for start_interview function"""
        topic: str = args.get("topic", "")
        return StartInterviewResult(topic=topic), create_interview_node()

    start_interview_func = FlowsFunctionSchema(
        name="start_interview",
        handler=handle_start_interview,
        description="Use this when the guest has shared a clear topic they want to explore.",
        properties={
            "topic": {"type": "string", "description": "The topic the guest wants to discuss"}
        },
        required=["topic"],
    )
    return NodeConfig(
        name="topic",
        task_messages=[
            {
                "role": "system",
                "content": "Now that you know who the guest is, help them choose the topic they'd like to explore. Refer back to their introduction to personalize the transition. Ask what topic, story, or challenge they're excited to discuss today. Show genuine interest and, if needed, ask a clarifying question to make sure you understand the angle they want to take. Once the topic feels clear and specific enough to dive into, use the start_interview function.",
            }
        ],
        functions=[start_interview_func],
    )


def create_interview_node() -> NodeConfig:
    """Create the Interview node."""

    async def handle_next_question(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        """Handler for next_question function"""
        return None, create_interview_node()

    async def handle_wrap_up(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Handler for wrap_up function"""
        return None, create_conclusion_node()

    next_question_func = FlowsFunctionSchema(
        name="next_question",
        handler=handle_next_question,
        description="Use this after you've thoroughly explored the current aspect with multiple questions and follow-ups.",
        properties={},
        required=[],
    )
    wrap_up_func = FlowsFunctionSchema(
        name="wrap_up",
        handler=handle_wrap_up,
        description="Use this when you've gathered substantial insights and are ready to wrap up.",
        properties={},
        required=[],
    )
    return NodeConfig(
        name="interview",
        task_messages=[
            {
                "role": "system",
                "content": "You're now in the heart of the interview. Start by introducing the topic with enthusiasm, then dive deep into one key aspect at a time. Ask open-ended, thoughtful questions that invite storytelling and personal insights. Listen actively to responses and ask natural follow-up questions that build on what your guest shares—dig deeper into interesting points, ask for examples, or explore the 'why' behind their answers. Keep the conversation flowing naturally, like a genuine dialogue between friends. Once you've thoroughly explored an aspect (typically after 3-5 exchanges), use the next_question function to smoothly transition to the next key aspect. After covering 3 key aspects of the topic, use the wrap_up function to conclude the interview.",
            }
        ],
        functions=[next_question_func, wrap_up_func],
    )


def create_conclusion_node() -> NodeConfig:
    """Create the Conclusion node."""

    async def handle_end_interview(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[None, NodeConfig]:
        """Handler for end_interview function"""
        return None, create_final_node()

    end_interview_func = FlowsFunctionSchema(
        name="end_interview",
        handler=handle_end_interview,
        description="Use this after the guest has shared their final thoughts.",
        properties={},
        required=[],
    )
    return NodeConfig(
        name="conclusion",
        task_messages=[
            {
                "role": "system",
                "content": "Express genuine appreciation for the conversation and the insights your guest shared. Summarize 2-3 key takeaways or memorable points from your discussion in a warm, conversational way—this helps reinforce the value of the conversation. Then, ask your guest if they have any final thoughts, a last word, or anything else they'd like to add. Wait for their response, then use the end_interview function to wrap up.",
            }
        ],
        functions=[end_interview_func],
    )


def create_final_node() -> NodeConfig:
    """Create the Final node."""
    return NodeConfig(
        name="final",
        task_messages=[
            {
                "role": "system",
                "content": "Thank the guest one final time for joining you and for sharing their insights. End the conversation on a positive, warm note.",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY")) if needs_stt_tts() else None

    tts = (
        CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        )
        if needs_stt_tts()
        else None
    )

    # LLM service is created using the create_llm function from utils.py
    # Default is OpenAI; can be changed by setting LLM_PROVIDER environment variable
    llm = create_llm()

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
            ),
        ),
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        list(
            filter(
                None,
                [
                    transport.input(),
                    rtvi,
                    stt,
                    context_aggregator.user(),
                    llm,
                    tts,
                    transport.output(),
                    context_aggregator.assistant(),
                ],
            )
        )
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        await flow_manager.initialize(create_introduction_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            # turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            # turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
