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
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Patient Intake
#
# This configuration defines a medical intake system with the following states:
#
# 1. start
#    - Initial state where system verifies patient identity through birthday
#    - Functions:
#      * verify_birthday (node function to check DOB)
#      * get_prescriptions (transitions to prescription collection)
#    - Pre-action: Initial greeting from Jessica
#    - Expected flow: Greet -> Ask DOB -> Verify -> Transition to prescriptions
#
# 2. get_prescriptions
#    - Collects information about patient's current medications
#    - Functions:
#      * record_prescriptions (node function, collects medication name and dosage)
#      * get_allergies (transitions to allergy collection)
#    - Expected flow: Ask about prescriptions -> Record details -> Transition to allergies
#
# 3. get_allergies
#    - Collects information about patient's allergies
#    - Functions:
#      * record_allergies (node function, records allergy information)
#      * get_conditions (transitions to medical conditions)
#    - Expected flow: Ask about allergies -> Record details -> Transition to conditions
#
# 4. get_conditions
#    - Collects information about patient's medical conditions
#    - Functions:
#      * record_conditions (node function, records medical conditions)
#      * get_visit_reasons (transitions to visit reason collection)
#    - Expected flow: Ask about conditions -> Record details -> Transition to visit reasons
#
# 5. get_visit_reasons
#    - Collects information about why patient is visiting
#    - Functions:
#      * record_visit_reasons (node function, records visit reasons)
#      * verify_information (transitions to verification)
#    - Expected flow: Ask about visit reason -> Record details -> Transition to verification
#
# 6. verify_information
#    - Reviews all collected information with patient
#    - Functions:
#      * get_prescriptions (returns to prescriptions if changes needed)
#      * end (transitions to end after confirmation)
#    - Expected flow: Review all info -> Confirm accuracy -> End or revise
#
# 7. end
#    - Final state that closes the conversation
#    - No functions available
#    - Pre-action: Thank you message
#    - Post-action: Ends conversation


async def verify_birthday(args: FlowArgs) -> FlowResult:
    """Handler for birthday verification."""
    birthday = args["birthday"]
    # In a real app, this would verify against patient records
    is_valid = birthday == "1983-01-01"
    return {"status": "success", "verified": is_valid}


async def record_prescriptions(args: FlowArgs) -> FlowResult:
    """Handler for recording prescriptions."""
    prescriptions = args["prescriptions"]
    # In a real app, this would store in patient records
    return {"status": "success", "count": len(prescriptions)}


async def record_allergies(args: FlowArgs) -> FlowResult:
    """Handler for recording allergies."""
    allergies = args["allergies"]
    # In a real app, this would store in patient records
    return {"status": "success", "count": len(allergies)}


async def record_conditions(args: FlowArgs) -> FlowResult:
    """Handler for recording medical conditions."""
    conditions = args["conditions"]
    # In a real app, this would store in patient records
    return {"status": "success", "count": len(conditions)}


async def record_visit_reasons(args: FlowArgs) -> FlowResult:
    """Handler for recording visit reasons."""
    visit_reasons = args["visit_reasons"]
    # In a real app, this would store in patient records
    return {"status": "success", "count": len(visit_reasons)}


flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "messages": [
                {
                    "role": "system",
                    "content": "Start by introducing yourself to Chad Bailey, then ask for their date of birth, including the year. Once they provide their birthday, use verify_birthday to check it. If the birthday is correct (1983-01-01), use get_prescriptions to proceed.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "verify_birthday",
                        "handler": verify_birthday,
                        "description": "Verify the user has provided their correct birthday",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "birthday": {
                                    "type": "string",
                                    "description": "The user's birthdate (convert to YYYY-MM-DD format)",
                                }
                            },
                            "required": ["birthday"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_prescriptions",
                        "description": "Proceed to collecting prescriptions and ask the user what medications they're taking",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "get_prescriptions": {
            "messages": [
                {
                    "role": "system",
                    "content": "This step is for collecting a user's prescription. Ask them what presceriptions they're taking, including the dosage. Use the available functions:\n - Use record_prescriptions when the user lists their medications (must include both name and dosage)\n - Use get_allergies once all prescriptions are recorded\n\nAsk them what prescriptions they're currently taking, making sure to get both medication names and dosages. After they provide their prescriptions (or confirm they have none), acknowledge their response and proceed to allergies",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_prescriptions",
                        "handler": record_prescriptions,
                        "description": "Record the user's prescriptions",
                        "parameters": {
                            "type": "object",
                            "properties": {
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
                            "required": ["prescriptions"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_allergies",
                        "description": "Proceed to collecting allergies and ask the user if they have any allergies",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "get_allergies": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are collecting allergy information. Use the available functions:\n - Use record_allergies when the user lists their allergies (or confirms they have none)\n - Use get_conditions once allergies are recorded\n\nAsk them about any allergies they have. After they list their allergies (or confirm they have none), acknowledge their response and ask about any medical conditions they have.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_allergies",
                        "handler": record_allergies,
                        "description": "Record the user's allergies",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "allergies": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "What the user is allergic to",
                                            }
                                        },
                                        "required": ["name"],
                                    },
                                }
                            },
                            "required": ["allergies"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_conditions",
                        "description": "Proceed to collecting medical conditions and ask the user if they have any medical conditions",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "get_conditions": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are collecting medical condition information. Use the available functions:\n - Use record_conditions when the user lists their conditions (or confirms they have none)\n - Use get_visit_reasons once conditions are recorded\n\nAsk them about any medical conditions they have. After they list their conditions (or confirm they have none), acknowledge their response and ask about the reason for their visit today.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_conditions",
                        "handler": record_conditions,
                        "description": "Record the user's medical conditions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "conditions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "The user's medical condition",
                                            }
                                        },
                                        "required": ["name"],
                                    },
                                }
                            },
                            "required": ["conditions"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_visit_reasons",
                        "description": "Proceed to collecting visit reasons and ask the user why they're visiting",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "get_visit_reasons": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are collecting information about the reason for their visit. Use the available functions:\n - Use record_visit_reasons when they explain their reasons\n - Use verify_information once reasons are recorded\n\nAsk them what brings them to the doctor today. After they explain their reasons, acknowledge their response and let them know you'll review all the information they've provided.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "record_visit_reasons",
                        "handler": record_visit_reasons,
                        "description": "Record the reasons for their visit",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "visit_reasons": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "The user's reason for visiting",
                                            }
                                        },
                                        "required": ["name"],
                                    },
                                }
                            },
                            "required": ["visit_reasons"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "verify_information",
                        "description": "Proceed to information verification, repeat the information back to the user, and have them verify",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "verify_information": {
            "messages": [
                {
                    "role": "system",
                    "content": "Review all collected information. Summarize their prescriptions, allergies, conditions, and visit reasons. Use the available functions:\n - Use get_prescriptions if they need to make any changes\n - Use confirm_intake if they confirm everything is correct\n\nBe thorough in reviewing all details and ask for their explicit confirmation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_prescriptions",
                        "description": "Return to prescriptions to revise information",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "confirm_intake",
                        "description": "Confirm the intake information and proceed",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "confirm_intake": {
            "messages": [
                {
                    "role": "system",
                    "content": "The intake information is confirmed. Thank them for providing their information, let them know what to expect next, and use end to complete the conversation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "End the conversation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "pre_actions": [
                {
                    "type": "tts_say",
                    "text": "Perfect! I've recorded all your information for your visit.",
                }
            ],
        },
        "end": {
            "messages": [
                {
                    "role": "system",
                    "content": "Thank them for their time and end the conversation.",
                }
            ],
            "functions": [],
            "pre_actions": [
                {
                    "type": "tts_say",
                    "text": "Thank you for providing your information! We'll see you soon for your visit.",
                }
            ],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}


async def main():
    """Main function to set up and run the patient intake bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Patient Intake Bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-asteria-en")
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Get initial tools from the first node
        initial_tools = flow_config["nodes"]["start"]["functions"]

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are Jessica, an agent for Tri-County Health Services. You must ALWAYS use one of the available functions to progress the conversation. Be professional but friendly.",
            }
        ]

        context = OpenAILLMContext(messages, initial_tools)
        context_aggregator = llm.create_context_aggregator(context)

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

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager with LLM
        flow_manager = FlowManager(task, llm, tts, flow_config)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            # Initialize the flow processor
            await flow_manager.initialize(messages)
            # Kick off the conversation using the context aggregator
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
