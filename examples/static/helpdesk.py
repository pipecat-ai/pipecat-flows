#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

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
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=False)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# Type definitions
class TechnicalAnswerResult(FlowResult):
    answer: str
    docs_link: Optional[str] = None
    code_example: Optional[str] = None

class PricingResult(FlowResult):
    base_price: float
    discounted_price: Optional[float] = None
    discount_percentage: Optional[float] = None
    monthly_estimate: float
    product_type: str

# Function handlers
async def look_up_answer(args: FlowArgs) -> TechnicalAnswerResult:
    """Look up technical answers from our knowledge base."""
    question = args["question"].lower()
    
    # Mock knowledge base with realistic Daily-specific answers
    answers = {
        "create room": {
            "answer": "To create a room via the API, make a POST request to /v1/rooms. This will create a room with default settings. You can customize properties like room name, expiry, and features using optional parameters. Would you like to see a code example?",
            "code_example": """curl -H "Authorization: Bearer YOUR_API_KEY" \\
    -H "Content-Type: application/json" \\
    -X POST https://api.daily.co/v1/rooms \\
    -d '{"properties":{"max_participants":10,"enable_chat":true}}'"""
        },
        "browser": {
            "answer": "We support all modern browsers including Chrome 75 and above, Safari 13.1 and above, Firefox 91 ESR and above, and Microsoft Edge 75 and above. For mobile, we support iOS Safari 13.1 and later, and Android Chrome. For the best experience, we recommend using the latest version of Chrome or Safari. Would you like more specific information about any of these browsers?"
        },
        "api key": {
            "answer": "You can find your API key in your Daily dashboard under the Developers section. Never share your API key publicly or commit it to version control. For security, we recommend using environment variables to store your API key. Would you like to know more about API key best practices?"
        }
    }

    # Find best matching answer
    for key, value in answers.items():
        if key in question:
            return value
    
    # Fallback response with docs link
    return {
        "answer": "I don't have specific information about that in my knowledge base. For the most accurate and up-to-date information, I recommend checking our documentation at docs.daily.co or speaking with our support team. Would you like me to connect you with support?",
        "docs_link": "https://docs.daily.co"
    }

async def look_up_pricing(args: FlowArgs) -> PricingResult:
    """Calculate pricing based on product type and volume."""
    product = args["product_type"]
    minutes = args["number_of_minutes"]
    
    # Base prices per minute
    prices = {
        "video": 0.004,
        "audio": 0.00099,
        "recording": 0.01349,
        "realtime_transcription": 0.0059,
        "postcall_transcription": 0.0043,
        "livestreaming": 0.03
    }
    
    # Calculate volume discount
    discount = 0
    if minutes > 50_000_000:
        discount = 0.63
    elif minutes > 25_000_000:
        discount = 0.45
    elif minutes > 10_000_000:
        discount = 0.35
    elif minutes > 1_000_000:
        discount = 0.25
    elif minutes > 500_000:
        discount = 0.15
    elif minutes > 100_000:
        discount = 0.07

    base_price = prices[product] * minutes
    discounted_price = base_price * (1 - discount) if discount > 0 else None

    return {
        "base_price": base_price,
        "discounted_price": discounted_price,
        "discount_percentage": discount * 100 if discount > 0 else None,
        "monthly_estimate": minutes,
        "product_type": product
    }

flow_config: FlowConfig = {
  "initial_node": "start",
  "nodes": {
    "start": {
      "messages": [
        {
          "role": "system",
          "content": "For this step, ask the user if they have a technical or a pricing question, and wait for them to use a function to choose. Start by greeting them and briefly mentioning that you can help with Daily's products."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "choose_technical_question",
            "description": "User has a technical question about Daily's APIs or implementation. Let's get started understanding their question.",
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
            "description": "User has a question about Daily's pricing. Let's get started understanding their question.",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "select_product"
          }
        }
      ]
    },
    "select_product": {
      "messages": [
        {
          "role": "system",
          "content": "Ask which product they're interested in pricing for. Briefly list the base prices: video ($0.004/min), audio ($0.00099/min), recording ($0.01349/min), real-time transcription ($0.0059/min), post-call transcription ($0.0043/min), or HLS livestreaming ($0.03/min). Wait for them to select a product type."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "look_up_pricing",
            "handler": "__function__:look_up_pricing",
            "description": "Calculate pricing based on product and volume.",
            "parameters": {
              "type": "object",
              "properties": {
                "product_type": {
                  "type": "string",
                  "enum": ["video", "audio", "recording", "realtime_transcription", "postcall_transcription", "livestreaming"],
                  "description": "Type of product to price"
                },
                "number_of_minutes": {
                  "type": "number",
                  "description": "Estimated minutes per month"
                }
              },
              "required": ["product_type", "number_of_minutes"]
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "confirm_pricing",
            "description": "User understands the pricing",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "confirm"
          }
        }
      ]
    },
    "choose_technical_question": {
      "messages": [
        {
          "role": "system",
          "content": "You are handling a technical question. Use the look_up_answer function when someone asks a question. If you can't answer their question or they want to talk to support, use transfer_to_person. Use confirm_conversation only when they're satisfied with the answer. Be friendly and professional. If you provide code examples, format them clearly."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "look_up_answer",
            "handler": "__function__:look_up_answer",
            "description": "Look up the answer to the user's technical question.",
            "parameters": {
              "type": "object",
              "properties": {
                "question": {
                  "type": "string",
                  "description": "The user's technical question"
                }
              },
              "required": ["question"]
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "transfer_to_person",
            "description": "Transfer to a support person",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "transfer"
          }
        },
        {
          "type": "function",
          "function": {
            "name": "confirm_conversation",
            "description": "User is satisfied with the answer",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "confirm"
          }
        }
      ]
    },
    "transfer": {
      "messages": [
        {
          "role": "system",
          "content": "Politely explain that you'll connect them with our support team. Mention that someone will reach out via email shortly. Then use complete_conversation to end the chat."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "complete_conversation",
            "description": "End the conversation after transfer message",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "end"
          }
        }
      ]
    },
    "confirm": {
      "messages": [
        {
          "role": "system",
          "content": "Confirm if they have any other questions. They can ask another technical question, check pricing for another product, or end the conversation. Be friendly and clear."
        }
      ],
      "functions": [
        {
          "type": "function",
          "function": {
            "name": "complete_conversation",
            "description": "User is done and wants to end the conversation",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "end"
          }
        },
        {
          "type": "function",
          "function": {
            "name": "new_technical_question",
            "description": "User has another technical question",
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
            "name": "new_pricing_question",
            "description": "User wants to check pricing for another product",
            "parameters": {
              "type": "object",
              "properties": {}
            },
            "transition_to": "select_product"
          }
        }
      ]
    },
    "end": {
      "messages": [
        {
          "role": "system",
          "content": "Thank them briefly and say goodbye (1-3 words is perfect)."
        }
      ],
      "functions": [],
      "post_actions": [
        {
          "type": "end_conversation"
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
            text_filter=MarkdownTextFilter(),
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": """You are a Daily support specialist. Daily provides real-time voice, video, and AI APIs for developers. 
                You must ALWAYS use the available functions to progress the conversation. Your responses will be converted to audio.
                Be brief, friendly, and professional. Focus on being helpful and accurate. If you're unsure about anything, 
                acknowledge that and offer to connect the user with our support team.""",
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
