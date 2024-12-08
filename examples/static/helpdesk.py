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
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=False)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Type definitions
class TechnicalAnswerResult(FlowResult):
    answer: str

# Function handlers
async def look_up_answer(args: FlowArgs) -> TechnicalAnswerResult:
    """Look up the answer from Zendesk."""

    return {"answer": "To fix this problem, turn your computer off and on again."}



flow_config: FlowConfig = {
  "initial_node": "start",
  "nodes": {
    "confirm": {
      "messages": [
        {
          "role": "system",
          "content": "Read back the complete question details and ask the user for confirmation. Use the available functions: - Use complete_conversation when the user confirms - Use revise_question if they want to change something. Be friendly and clear when reading back the order details."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "complete_conversation",
            "description": "User confirms the question is correct",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "end"
          }
        }
      ]
    },
    "choose_technical_question": {
      "messages": [
        {
          "role": "system",
          "content": "You are handling a technical question. Use the available function: - Use the look_up_answer function when someone asks a question. - Use the end function ONLY when the user confirms you have answered their question. - Use the transfer_to_person function if the user wants to talk to a real person or you cannot provide an answer. Only use the end function after the user confirms their question is answered. Start by acknowledging the user's choice. Remember to be friendly and professional."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "look_up_answer",
            "handler": "look_up_answer",
            "description": "Look up the answer to the user's question in the Zendesk knowledge base.",
            "parameters": {
              "type": "object",
              "properties": {
                "question": {
                  "type": "string",
                  "description": "The user's question."
                }
              },
              "required": [
                "question"
              ]
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "confirm_conversation",
            "description": "Proceed to end conversation",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "confirm"
          }
        }
      ]
    },
    "end": {
      "messages": [
        {
          "role": "system",
          "content": "Concisely end the conversation—1-3 words is appropriate. Just say 'Bye' or something similarly short."
        }
      ],
      "functions": [],
      "post_actions": [
        {
          "type": "end_conversation"
        }
      ]
    },
    "start": {
      "messages": [
        {
          "role": "system",
          "content": "You are a customer service bot for Daily. For this step, ask them if they have a technical or a pricing question, and wait for them to use a function to choose. Start by greeting them. Be brief, friendly, and professional."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "choose_technical_question",
            "description": "User has a question related to the product and needs help resolving.",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "choose_technical_question"
          }
        },
        {
          "type": "function",
          "function": {
            "name": "choose_pricing_question",
            "description": "User has a question about pricing.",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "choose_pricing_question"
          }
        }
      ]
    },
    "choose_pricing_question": {
      "messages": [
        {
          "role": "system",
          "content": "You are handling a pricing question. Our pricing is the following: - We charge $0.02 per minute per participant. - The first 10000 minutes per month are free. - We offer a 10% discount if you pay yearly instead of monthly. Use this information to calculate the price. For example, if the user writes: \"How much would it cost to have 1000 monthly meetings that last an hour with two people on the call each time?\" Respond with:\"That would be 120000 minutes per month. You receive 10000 minutes for free each month, meaning you only pay for 110000 minutes. Your total cost would be $2200 per month, or $23,760 per year with our 10% yearly discount.\" Only use the end function after the user confirms their question is answered. Explain your logic in detail. If you're unsure about your answer, tell the user and encourage them to check www.daily.co/pricing for up-to-date pricing information. Start by acknowledging the user's choice. Remember to be friendly and professional."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "confirm_conversation",
            "description": "Proceed to end conversation",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "confirm"
          }
        }
      ]
    }
  }
}

async def main():
    """Main function to set up and run the helpdesk bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        # Initialize services
        transport = DailyTransport(
            room_url,
            None,
            "Helpdesk Bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are a customer support bot. You must ALWAYS use the available functions to progress the conversation. Your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.",
            }
        ]

        context = OpenAILLMContext(messages, flow_config["nodes"]["start"]["functions"])
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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager in static mode
        flow_manager = FlowManager(task=task, llm=llm, tts=tts, flow_config=flow_config)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            logger.debug("Initializing flow")
            await flow_manager.initialize(messages)
            logger.debug("Starting conversation")
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
