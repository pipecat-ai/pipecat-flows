#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A dynamic food ordering flow example for Pipecat Flows.

This example demonstrates a food ordering system using dynamic flows where
conversation paths are determined at runtime. The flow handles:

1. Initial greeting and food type selection (pizza or sushi)
2. Order details collection based on food type
3. Order confirmation and revision
4. Order completion

Multi-LLM Support:
Set LLM_PROVIDER environment variable to choose your LLM provider.
Supported: openai (default), anthropic, google, aws

Requirements:
- CARTESIA_API_KEY (for TTS)
- DEEPGRAM_API_KEY (for STT)
- DAILY_API_KEY (for transport)
- LLM API key (varies by provider - see env.example)
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
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
from pipecat.services.llm_service import FunctionCallParams

from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from utils import create_llm

from pipecat_flows import (
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
class PizzaOrderResult(FlowResult):
    size: str
    type: str
    price: float


class SushiOrderResult(FlowResult):
    count: int
    type: str
    price: float


# Pre-action handlers
async def check_kitchen_status(action: dict, flow_manager: FlowManager) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


# Node creation functions
def create_initial_node() -> NodeConfig:
    """Create the initial node for food type selection."""

    async def choose_pizza(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to pizza order selection."""
        return None, create_pizza_node()

    async def choose_sushi(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to sushi order selection."""
        return None, create_sushi_node()

    choose_pizza_func = FlowsFunctionSchema(
        name="choose_pizza",
        handler=choose_pizza,
        description="User wants to order pizza. Let's get that order started.",
        properties={},
        required=[],
    )

    choose_sushi_func = FlowsFunctionSchema(
        name="choose_sushi",
        handler=choose_sushi,
        description="User wants to order sushi. Let's get that order started.",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="initial",
        role_messages=[
            {
                "role": "system",
                "content": "You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.",
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": "For this step, ask the user if they want pizza or sushi, and wait for them to use a function to choose. Start off by greeting them. Be friendly and casual; you're taking an order for food over the phone.",
            }
        ],
        pre_actions=[
            {
                "type": "function",
                "handler": check_kitchen_status,
            },
        ],
        functions=[choose_pizza_func, choose_sushi_func],
    )


def create_pizza_node() -> NodeConfig:
    """Create the pizza ordering node."""

    async def select_pizza_order(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[PizzaOrderResult, NodeConfig]:
        """Handle pizza size and type selection."""
        size = args["size"]
        pizza_type = args["type"]

        # Simple pricing
        base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
        price = base_price[size]

        result = PizzaOrderResult(size=size, type=pizza_type, price=price)

        # Store order details in flow state
        flow_manager.state["order"] = {
            "type": "pizza",
            "size": size,
            "pizza_type": pizza_type,
            "price": price,
        }

        return result, create_confirmation_node()

    select_pizza_func = FlowsFunctionSchema(
        name="select_pizza_order",
        handler=select_pizza_order,
        description="Record the pizza order details",
        properties={
            "size": {
                "type": "string",
                "enum": ["small", "medium", "large"],
                "description": "Size of the pizza",
            },
            "type": {
                "type": "string",
                "enum": ["pepperoni", "cheese", "supreme", "vegetarian"],
                "description": "Type of pizza",
            },
        },
        required=["size", "type"],
    )

    return NodeConfig(
        name="choose_pizza",
        task_messages=[
            {
                "role": "system",
                "content": """You are handling a pizza order. Use the available functions:
- Use select_pizza_order when the user specifies both size AND type

Pricing:
- Small: $10
- Medium: $15
- Large: $20

Remember to be friendly and casual.""",
            }
        ],
        functions=[select_pizza_func],
    )


def create_sushi_node() -> NodeConfig:
    """Create the sushi ordering node."""

    async def select_sushi_order(
        args: FlowArgs, flow_manager: FlowManager
    ) -> tuple[SushiOrderResult, NodeConfig]:
        """Handle sushi roll count and type selection."""
        count = args["count"]
        roll_type = args["type"]

        # Simple pricing: $8 per roll
        price = count * 8.00

        result = SushiOrderResult(count=count, type=roll_type, price=price)

        # Store order details in flow state
        flow_manager.state["order"] = {
            "type": "sushi",
            "count": count,
            "roll_type": roll_type,
            "price": price,
        }

        return result, create_confirmation_node()

    select_sushi_func = FlowsFunctionSchema(
        name="select_sushi_order",
        handler=select_sushi_order,
        description="Record the sushi order details",
        properties={
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of rolls to order",
            },
            "type": {
                "type": "string",
                "enum": ["california", "spicy tuna", "rainbow", "dragon"],
                "description": "Type of sushi roll",
            },
        },
        required=["count", "type"],
    )

    return NodeConfig(
        name="choose_sushi",
        task_messages=[
            {
                "role": "system",
                "content": """You are handling a sushi order. Use the available functions:
- Use select_sushi_order when the user specifies both count AND type

Pricing:
- $8 per roll

Remember to be friendly and casual.""",
            }
        ],
        functions=[select_sushi_func],
    )


def create_confirmation_node() -> NodeConfig:
    """Create the order confirmation node."""

    async def complete_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to end state."""
        return None, create_end_node()

    async def revise_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
        """Transition to start for order revision."""
        return None, create_initial_node()

    complete_order_func = FlowsFunctionSchema(
        name="complete_order",
        handler=complete_order,
        description="User confirms the order is correct",
        properties={},
        required=[],
    )

    revise_order_func = FlowsFunctionSchema(
        name="revise_order",
        handler=revise_order,
        description="User wants to make changes to their order",
        properties={},
        required=[],
    )

    return NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "system",
                "content": """Read back the complete order details to the user and ask if they want anything else or if they want to make changes. Use the available functions:
- Use complete_order when the user confirms that the order is correct and no changes are needed
- Use revise_order if they want to change something

Be friendly and clear when reading back the order details.""",
            }
        ],
        functions=[complete_order_func, revise_order_func],
    )


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return NodeConfig(
        name="end",
        task_messages=[
            {
                "role": "system",
                "content": "Thank the user for their order and end the conversation politely and concisely.",
            }
        ],
        post_actions=[{"type": "end_conversation"}],
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the food ordering bot."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        text_filters=[MarkdownTextFilter()],
    )
    # LLM service is created using the create_llm function from utils.py
    # Default is OpenAI; can be changed by setting LLM_PROVIDER environment variable
    llm = create_llm()


   


    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

  

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

    # Define default tools that will be available at every node
    # Example: A help function available throughout the conversation
    async def get_help_with_time(args: FlowArgs, flow_manager: FlowManager) -> FlowResult:
        """Provide help information about current Time  to the user."""
        current_time = datetime.now() + timedelta(minutes=15)
        return {
            "status": "success",
            "message": f"You can order pizza or sushi. The Current time is {current_time}.",
        }

    default_tools = [
        FlowsFunctionSchema(
            name="get_help",
            handler=get_help_with_time,
            description="Get help information about the current time",
            properties={},
            required=[],
        ),
    ]

    # Initialize flow manager in dynamic mode with default tools
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
        default_tools=default_tools,  # Pass default tools here
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





#
# Copyright (c) 2024-2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

# """Enhanced food ordering with GLOBAL FUNCTIONS available at every node.

# This example demonstrates:
# 1. Global functions (speak_to_manager, check_order_status, etc.) available everywhere
# 2. Node-specific functions (choose_pizza, select_pizza_order, etc.)
# 3. Real-world restaurant ordering scenario
# """

# import os
# from datetime import datetime
# from dotenv import load_dotenv
# from loguru import logger
# from pipecat.audio.vad.silero import SileroVADAnalyzer
# from pipecat.pipeline.pipeline import Pipeline
# from pipecat.pipeline.runner import PipelineRunner
# from pipecat.pipeline.task import PipelineParams, PipelineTask
# from pipecat.processors.aggregators.llm_context import LLMContext
# from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
# from pipecat.runner.types import RunnerArguments
# from pipecat.runner.utils import create_transport
# from pipecat.services.cartesia.tts import CartesiaTTSService
# from pipecat.services.deepgram.stt import DeepgramSTTService
# from pipecat.transports.base_transport import BaseTransport, TransportParams
# from pipecat.transports.daily.transport import DailyParams
# from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
# from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
# from utils import create_llm

# from pipecat_flows import (
#     FlowArgs,
#     FlowManager,
#     FlowResult,
#     FlowsFunctionSchema,
#     NodeConfig,
# )

# load_dotenv(override=True)

# transport_params = {
#     "daily": lambda: DailyParams(
#         audio_in_enabled=True,
#         audio_out_enabled=True,
#         vad_analyzer=SileroVADAnalyzer(),
#     ),
#     "twilio": lambda: FastAPIWebsocketParams(
#         audio_in_enabled=True,
#         audio_out_enabled=True,
#         vad_analyzer=SileroVADAnalyzer(),
#     ),
#     "webrtc": lambda: TransportParams(
#         audio_in_enabled=True,
#         audio_out_enabled=True,
#         vad_analyzer=SileroVADAnalyzer(),
#     ),
# }


# # ==================== TYPE DEFINITIONS ====================
# class PizzaOrderResult(FlowResult):
#     size: str
#     type: str
#     price: float


# class SushiOrderResult(FlowResult):
#     count: int
#     type: str
#     price: float


# # ==================== GLOBAL FUNCTIONS ====================
# # These functions are available at EVERY node in the conversation



# async def check_order_status(
#     args: FlowArgs, flow_manager: FlowManager
# ) -> FlowResult:
#     """
#     Check current order status and details.
    
#     This is a non-edge function (doesn't cause transitions).
#     The LLM can call this to see what's been ordered so far.
#     """
#     order = flow_manager.state.get("order", {})
#     special_instructions = flow_manager.state.get("special_instructions", [])
    
#     if not order:
#         return {
#             "status": "success",
#             "message": "No order placed yet",
#             "order": None,
#         }
    
#     return {
#         "status": "success",
#         "message": "Current order retrieved",
#         "order": order,
#         "special_instructions": special_instructions,
#         "total_price": order.get("price", 0.0),
#     }


# async def add_special_instructions(
#     args: FlowArgs, flow_manager: FlowManager
# ) -> FlowResult:
#     """
#     Add special cooking or delivery instructions.
    
#     Examples: "No olives", "Extra spicy", "Ring doorbell twice"
#     """
#     instruction = args["instruction"]
    
#     # Initialize list if needed
#     if "special_instructions" not in flow_manager.state:
#         flow_manager.state["special_instructions"] = []
    
#     # Add instruction
#     flow_manager.state["special_instructions"].append(instruction)
    
#     logger.info(f"📝 Special instruction added: {instruction}")
    
#     return {
#         "status": "success",
#         "instruction_added": instruction,
#         "all_instructions": flow_manager.state["special_instructions"],
#     }


# async def cancel_order(
#     args: FlowArgs, flow_manager: FlowManager
# ) -> tuple[FlowResult, NodeConfig]:
#     """
#     Cancel the entire order and restart.
    
#     Available at any time for customer convenience.
#     """
#     reason = args.get("reason", "customer request")
    
#     logger.warning(f"❌ Order cancelled: {reason}")
    
#     # Log the cancellation
#     cancelled_order = flow_manager.state.get("order", {})
#     flow_manager.state["last_cancelled_order"] = cancelled_order
#     flow_manager.state["cancellation_reason"] = reason
    
#     # Clear current order
#     flow_manager.state["order"] = {}
#     flow_manager.state["special_instructions"] = []
    
#     return {
#         "status": "cancelled",
#         "reason": reason,
#     }, create_initial_node()


# async def request_delivery_estimate(
#     args: FlowArgs, flow_manager: FlowManager
# ) -> FlowResult:
#     """
#     Get estimated delivery time for the current order.
#     """
#     order = flow_manager.state.get("order")
    
#     if not order:
#         return {
#             "status": "error",
#             "message": "No order placed yet. Please place an order first.",
#         }
    
#     # Mock delivery time calculation (30-45 mins)
#     order_type = order.get("type", "food")
#     base_time = 30 if order_type == "pizza" else 40
    
#     logger.info(f"🚚 Delivery estimate requested for {order_type}")
    
#     return {
#         "status": "success",
#         "estimated_time_minutes": base_time,
#         "estimated_time_range": f"{base_time}-{base_time + 15} minutes",
#         "order_type": order_type,
#     }


# # Create the global functions list
# GLOBAL_FUNCTION_SCHEMAS = [
#     FlowsFunctionSchema(
#         name="check_order_status",
#         handler=check_order_status,
#         description="Check what items are currently in the order, including price and special instructions.",
#         properties={},
#         required=[],
#     ),
#     FlowsFunctionSchema(
#         name="add_special_instructions",
#         handler=add_special_instructions,
#         description="Add special instructions for the kitchen or delivery. Examples: 'No onions', 'Extra crispy', 'Leave at door'",
#         properties={
#             "instruction": {
#                 "type": "string",
#                 "description": "The special instruction to add",
#             }
#         },
#         required=["instruction"],
#     ),
#     FlowsFunctionSchema(
#         name="cancel_order",
#         handler=cancel_order,
#         description="Cancel the entire order and start over. Only use if customer explicitly wants to cancel.",
#         properties={
#             "reason": {
#                 "type": "string",
#                 "description": "Why the order is being cancelled",
#             }
#         },
#         required=[],
#     ),
#     FlowsFunctionSchema(
#         name="request_delivery_estimate",
#         handler=request_delivery_estimate,
#         description="Get estimated delivery time for the current order.",
#         properties={},
#         required=[],
#     ),
# ]


# # ==================== HELPER FUNCTION ====================
# def create_node_with_globals(
#     name: str,
#     task_messages: list,
#     node_functions: list = None,
#     **kwargs,
# ) -> NodeConfig:
#     """
#     Helper to create a node with global functions automatically included.
    
#     Args:
#         name: Node identifier
#         task_messages: Task-specific system messages
#         node_functions: Node-specific functions (will be combined with globals)
#         **kwargs: Additional NodeConfig parameters
    
#     Returns:
#         NodeConfig with both global and node-specific functions
#     """
#     # Combine global + node-specific functions
#     all_functions = GLOBAL_FUNCTION_SCHEMAS.copy()
#     if node_functions:
#         all_functions.extend(node_functions)
    
#     return NodeConfig(
#         name=name,
#         task_messages=task_messages,
#         functions=all_functions,
#         **kwargs,
#     )


# # ==================== PRE-ACTION HANDLERS ====================
# async def check_kitchen_status(action: dict, flow_manager: FlowManager) -> None:
#     """Check if kitchen is open and log status."""
#     logger.info("✅ Checking kitchen status - Kitchen is OPEN")
#     flow_manager.state["kitchen_status"] = "open"


# # ==================== NODE CREATION FUNCTIONS ====================
# def create_initial_node() -> NodeConfig:
#     """Create the initial node for food type selection."""

#     async def choose_pizza(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
#         """Transition to pizza order selection."""
#         logger.info("🍕 Customer chose pizza")
#         return None, create_pizza_node()

#     async def choose_sushi(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
#         """Transition to sushi order selection."""
#         logger.info("🍣 Customer chose sushi")
#         return None, create_sushi_node()

#     choose_pizza_func = FlowsFunctionSchema(
#         name="choose_pizza",
#         handler=choose_pizza,
#         description="User wants to order pizza. Let's get that order started.",
#         properties={},
#         required=[],
#     )

#     choose_sushi_func = FlowsFunctionSchema(
#         name="choose_sushi",
#         handler=choose_sushi,
#         description="User wants to order sushi. Let's get that order started.",
#         properties={},
#         required=[],
#     )

#     return create_node_with_globals(
#         name="initial",
#         role_messages=[
#             {
#                 "role": "system",
#                 "content": """You are an order-taking assistant for a restaurant. 

# IMPORTANT: You have access to global functions that are ALWAYS available:
# - speak_to_manager: If customer is upset or asks for manager
# - check_order_status: To see what's been ordered
# - add_special_instructions: For special requests
# - cancel_order: If customer wants to cancel
# - request_delivery_estimate: For delivery time

# You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.""",
#             }
#         ],
#         task_messages=[
#             {
#                 "role": "system",
#                 "content": "For this step, greet the customer warmly and ask if they want pizza or sushi. Wait for them to choose using the available functions. Be friendly and casual; you're taking an order for food over the phone.",
#             }
#         ],
#         pre_actions=[
#             {
#                 "type": "function",
#                 "handler": check_kitchen_status,
#             },
#         ],
#         node_functions=[choose_pizza_func, choose_sushi_func],
#     )


# def create_pizza_node() -> NodeConfig:
#     """Create the pizza ordering node."""

#     async def select_pizza_order(
#         args: FlowArgs, flow_manager: FlowManager
#     ) -> tuple[PizzaOrderResult, NodeConfig]:
#         """Handle pizza size and type selection."""
#         size = args["size"]
#         pizza_type = args["type"]

#         # Simple pricing
#         base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
#         price = base_price[size]

#         result = PizzaOrderResult(size=size, type=pizza_type, price=price)

#         # Store order details in flow state
#         flow_manager.state["order"] = {
#             "type": "pizza",
#             "size": size,
#             "pizza_type": pizza_type,
#             "price": price,
#         }

#         logger.info(f"🍕 Pizza ordered: {size} {pizza_type} - ${price}")

#         return result, create_confirmation_node()

#     select_pizza_func = FlowsFunctionSchema(
#         name="select_pizza_order",
#         handler=select_pizza_order,
#         description="Record the pizza order details when customer specifies BOTH size and type",
#         properties={
#             "size": {
#                 "type": "string",
#                 "enum": ["small", "medium", "large"],
#                 "description": "Size of the pizza",
#             },
#             "type": {
#                 "type": "string",
#                 "enum": ["pepperoni", "cheese", "supreme", "vegetarian"],
#                 "description": "Type of pizza",
#             },
#         },
#         required=["size", "type"],
#     )

#     return create_node_with_globals(
#         name="choose_pizza",
#         task_messages=[
#             {
#                 "role": "system",
#                 "content": """You are handling a pizza order. Use the available functions:
# - Use select_pizza_order when the user specifies both size AND type

# Pricing:
# - Small: $10
# - Medium: $15
# - Large: $20

# Remember: Global functions like speak_to_manager, add_special_instructions, and check_order_status are ALWAYS available if the customer needs them.

# Be friendly and casual.""",
#             }
#         ],
#         node_functions=[select_pizza_func],
#     )


# def create_sushi_node() -> NodeConfig:
#     """Create the sushi ordering node."""

#     async def select_sushi_order(
#         args: FlowArgs, flow_manager: FlowManager
#     ) -> tuple[SushiOrderResult, NodeConfig]:
#         """Handle sushi roll count and type selection."""
#         count = args["count"]
#         roll_type = args["type"]

#         # Simple pricing: $8 per roll
#         price = count * 8.00

#         result = SushiOrderResult(count=count, type=roll_type, price=price)

#         # Store order details in flow state
#         flow_manager.state["order"] = {
#             "type": "sushi",
#             "count": count,
#             "roll_type": roll_type,
#             "price": price,
#         }

#         logger.info(f"🍣 Sushi ordered: {count}x {roll_type} - ${price}")

#         return result, create_confirmation_node()

#     select_sushi_func = FlowsFunctionSchema(
#         name="select_sushi_order",
#         handler=select_sushi_order,
#         description="Record the sushi order details when customer specifies BOTH count and type",
#         properties={
#             "count": {
#                 "type": "integer",
#                 "minimum": 1,
#                 "maximum": 10,
#                 "description": "Number of rolls to order",
#             },
#             "type": {
#                 "type": "string",
#                 "enum": ["california", "spicy tuna", "rainbow", "dragon"],
#                 "description": "Type of sushi roll",
#             },
#         },
#         required=["count", "type"],
#     )

#     return create_node_with_globals(
#         name="choose_sushi",
#         task_messages=[
#             {
#                 "role": "system",
#                 "content": """You are handling a sushi order. Use the available functions:
# - Use select_sushi_order when the user specifies both count AND type

# Pricing:
# - $8 per roll

# Remember: Global functions like speak_to_manager, add_special_instructions, and check_order_status are ALWAYS available.

# Be friendly and casual.""",
#             }
#         ],
#         node_functions=[select_sushi_func],
#     )


# def create_confirmation_node() -> NodeConfig:
#     """Create the order confirmation node."""

#     async def complete_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
#         """Transition to end state."""
#         logger.info("✅ Order confirmed by customer")
#         return None, create_end_node()

#     async def revise_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
#         """Transition to start for order revision."""
#         logger.info("🔄 Customer wants to revise order")
#         return None, create_initial_node()

#     complete_order_func = FlowsFunctionSchema(
#         name="complete_order",
#         handler=complete_order,
#         description="User confirms the order is correct and wants to complete it",
#         properties={},
#         required=[],
#     )

#     revise_order_func = FlowsFunctionSchema(
#         name="revise_order",
#         handler=revise_order,
#         description="User wants to make changes to their order",
#         properties={},
#         required=[],
#     )

#     return create_node_with_globals(
#         name="confirm",
#         task_messages=[
#             {
#                 "role": "system",
#                 "content": """Read back the complete order details to the user clearly and ask if everything is correct. 

# Use the available functions:
# - Use complete_order when the user confirms the order is correct
# - Use revise_order if they want to change something

# Remember: Global functions are still available. Customer can:
# - Add special instructions (add_special_instructions)
# - Request delivery estimate (request_delivery_estimate)
# - Check order status (check_order_status)
# - Speak to manager (speak_to_manager)

# Be friendly and clear when reading back the order details.""",
#             }
#         ],
#         node_functions=[complete_order_func, revise_order_func],
#     )


# def create_manager_node() -> NodeConfig:
#     """Create the manager escalation node."""

#     async def return_to_order(args: FlowArgs, flow_manager: FlowManager) -> tuple[None, NodeConfig]:
#         """Return to order after speaking with manager."""
#         logger.info("🔙 Returning from manager to order flow")
        
#         # Check if there's an existing order
#         if flow_manager.state.get("order"):
#             return None, create_confirmation_node()
#         else:
#             return None, create_initial_node()

#     return_to_order_func = FlowsFunctionSchema(
#         name="return_to_order",
#         handler=return_to_order,
#         description="Manager has resolved the issue and customer wants to continue ordering",
#         properties={},
#         required=[],
#     )

#     return create_node_with_globals(
#         name="manager",
#         task_messages=[
#             {
#                 "role": "system",
#                 "content": """You are now acting as the restaurant manager. The customer has been escalated to you.

# Their concern was: Check the flow_manager.state for 'escalation_reason'.

# Be empathetic, apologize if appropriate, and work to resolve their issue. When resolved, use return_to_order to continue their order.

# Global functions like cancel_order are still available if needed.""",
#             }
#         ],
#         node_functions=[return_to_order_func],
#     )


# def create_end_node() -> NodeConfig:
#     """Create the final node."""
#     return create_node_with_globals(
#         name="end",
#         task_messages=[
#             {
#                 "role": "system",
#                 "content": """Thank the user for their order. 

# Mention:
# - Their order total (check flow_manager.state['order']['price'])
# - Estimated delivery time (30-45 minutes)
# - Any special instructions they added

# End the conversation politely and concisely.""",
#             }
#         ],
#         post_actions=[{"type": "end_conversation"}],
#         node_functions=[],  # No node-specific functions, but globals still available!
#     )


# # ==================== MAIN BOT LOGIC ====================
# async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
#     """Run the enhanced food ordering bot with global functions."""
#     stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
#     tts = CartesiaTTSService(
#         api_key=os.getenv("CARTESIA_API_KEY"),
#         voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
#         text_filters=[MarkdownTextFilter()],
#     )
#     llm = create_llm()

#     context = LLMContext()
#     context_aggregator = LLMContextAggregatorPair(context)

#     pipeline = Pipeline(
#         [
#             transport.input(),
#             stt,
#             context_aggregator.user(),
#             llm,
#             tts,
#             transport.output(),
#             context_aggregator.assistant(),
#         ]
#     )

#     task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

#     flow_manager = FlowManager(
#         task=task,
#         llm=llm,
#         context_aggregator=context_aggregator,
#         transport=transport,
#     )

#     @transport.event_handler("on_client_connected")
#     async def on_client_connected(transport, client):
#         logger.info("🎉 Client connected - Starting order flow")
#         await flow_manager.initialize(create_initial_node())

#     @transport.event_handler("on_client_disconnected")
#     async def on_client_disconnected(transport, client):
#         logger.info("👋 Client disconnected")
#         await task.cancel()

#     runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
#     await runner.run(task)


# async def bot(runner_args: RunnerArguments):
#     """Main bot entry point compatible with Pipecat Cloud."""
#     transport = await create_transport(runner_args, transport_params)
#     await run_bot(transport, runner_args)


# if __name__ == "__main__":
#     from pipecat.runner.run import main

#     main()