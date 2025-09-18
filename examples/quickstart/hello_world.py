#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""A 'Hello-World' introduction to Pipecat Flows.

Requirements:
- CARTESIA_API_KEY
- GOOGLE_API_KEY

Run the example:
uv run hello_world.py
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
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_flows import (
    FlowArgs,
    FlowManager,
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


# Flow nodes
def create_initial_node() -> NodeConfig:
    """Create the initial node of the flow.

    Define the bot's role and task for the node as well as the function for it to call.
    The function call includes a handler which provides the function call result to
    Pipecat and then transitions to the next node.
    """
    record_favorite_color_func = FlowsFunctionSchema(
        name="record_favorite_color_func",
        description="Record the color the user said is their favorite.",
        required=["color"],
        handler=record_favorite_color_and_set_next_node,
        properties={"color": {"type": "string"}},
    )

    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": "You are an inquisitive child. Use very simple language. Ask simple questions. You must ALWAYS use one of the available functions to progress the conversation. Your responses will be converted to audio. Avoid outputting special characters and emojis.",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Say 'Hello world' and ask what is the user's favorite color.",
            }
        ],
        "functions": [record_favorite_color_func],
    }


async def record_favorite_color_and_set_next_node(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Function handler that records the color then sets the next node.

    Here "record" means print to the console, but any logic could go here;
    Write to a database, make an API call, etc.
    """
    print(f"Your favorite color is: {args['color']}")
    return args["color"], create_end_node()


def create_end_node() -> NodeConfig:
    """End the conversation.

    Flows transitions to this node when the user has answered the question.
    It thanks the user and ends the conversation using the `end_conversation`
    post-action.
    """
    return NodeConfig(
        name="create_end_node",
        task_messages=[
            {
                "role": "system",
                "content": "Thank the user for answering and end the conversation",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="32b3f3c5-7171-46aa-abe7-b598964aa793",
        text_filters=[MarkdownTextFilter()],
    )
    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-exp")

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # Initialize flow manager
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
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
