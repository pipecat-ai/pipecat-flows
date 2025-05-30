#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Pipecat Flows.

This package provides a framework for building structured conversations in Pipecat.
The FlowManager can handle both static and dynamic conversation flows:

1. Static Flows:
   - Configuration-driven conversations with predefined paths
   - Entire flow structure defined upfront
   - Example:
        from pipecat_flows import FlowArgs, FlowResult

        async def collect_name(args: FlowArgs) -> FlowResult:
            name = args["name"]
            return {"status": "success", "name": name}

        flow_config = {
            "initial_node": "greeting",
            "nodes": {
                "greeting": {
                    "messages": [...],
                    "functions": [{
                        "type": "function",
                        "function": {
                            "name": "collect_name",
                            "handler": collect_name,
                            "description": "...",
                            "parameters": {...},
                            "transition_to": "next_step"
                        }
                    }]
                }
            }
        }
        flow_manager = FlowManager(task, llm, flow_config=flow_config)

2. Dynamic Flows:
   - Runtime-determined conversations
   - Nodes created or modified during execution
   - Example:
        from pipecat_flows import FlowArgs, FlowResult

        async def collect_age(args: FlowArgs) -> FlowResult:
            age = args["age"]
            return {"status": "success", "age": age}

        async def handle_transitions(function_name: str, args: Dict, flow_manager):
            if function_name == "collect_age":
                await flow_manager.set_node("next_step", create_next_node())

        flow_manager = FlowManager(task, llm, transition_callback=handle_transitions)
"""

from .exceptions import (
    ActionError,
    FlowError,
    FlowInitializationError,
    FlowTransitionError,
    InvalidFunctionError,
)
from .manager import FlowManager
from .types import (
    ConsolidatedFunctionResult,
    ContextStrategy,
    ContextStrategyConfig,
    DirectFunction,
    FlowArgs,
    FlowConfig,
    FlowFunctionHandler,
    FlowResult,
    FlowsFunctionSchema,
    LegacyFunctionHandler,
    NodeConfig,
)

__all__ = [
    # Flow Manager
    "FlowManager",
    # Types
    "ContextStrategy",
    "ContextStrategyConfig",
    "FlowArgs",
    "FlowConfig",
    "FlowFunctionHandler",
    "FlowResult",
    "ConsolidatedFunctionResult",
    "FlowsFunctionSchema",
    "LegacyFunctionHandler",
    "DirectFunction",
    "NodeConfig",
    # Exceptions
    "FlowError",
    "FlowInitializationError",
    "FlowTransitionError",
    "InvalidFunctionError",
    "ActionError",
]
