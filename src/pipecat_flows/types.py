#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type definitions for the conversation flow system.

This module defines the core types used throughout the flow system:
- FlowResult: Function return type
- FlowArgs: Function argument type (includes flow_manager for state access)
- NodeConfig: Node configuration type (supports $state references in content)
- FlowConfig: Complete flow configuration type

These types provide structure and validation for flow configurations,
function interactions, and state management.
"""

from typing import Any, Dict, List, TypedDict


class FlowResult(TypedDict, total=False):
    """Base type for function results.

    Example:
        {
            "status": "success",
            "data": {"processed": True},
            "error": None  # Optional error message
        }
    """

    status: str
    error: str


FlowArgs = Dict[str, Any]
"""Type alias for function handler arguments.

Example:
    {
        "user_name": "John",
        "age": 25,
        "preferences": {"color": "blue"},
        "flow_manager": FlowManager  # Available in all function handlers for state management
    }
"""


class NodeConfigRequired(TypedDict):
    """Required fields for node configuration."""

    task_messages: List[dict]
    functions: List[dict]


class NodeConfig(NodeConfigRequired, total=False):
    """Configuration for a single node in the flow.

    Required fields:
        task_messages: List of message dicts defining the current node's objectives
        functions: List of function definitions in provider-specific format

    Optional fields:
        role_messages: List of message dicts defining the bot's role/personality
        pre_actions: Actions to execute before LLM inference
        post_actions: Actions to execute after LLM inference

    Example:
        {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant..."
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Ask the user for their name..."
                }
            ],
            "functions": [...],
            "pre_actions": [...],
            "post_actions": [...]
        }
    """

    role_messages: List[Dict[str, Any]]
    pre_actions: List[Dict[str, Any]]
    post_actions: List[Dict[str, Any]]

    role_messages: List[Dict[str, Any]]
    pre_actions: List[Dict[str, Any]]
    post_actions: List[Dict[str, Any]]


class FlowConfig(TypedDict):
    """Configuration for the entire conversation flow.

    Attributes:
        initial_node: Name of the starting node
        nodes: Dictionary mapping node names to their configurations

    Example:
        {
            "initial_node": "greeting",
            "nodes": {
                "greeting": {
                    "role_messages": [...],
                    "task_messages": [...],
                    "functions": [...],
                    "pre_actions": [...]
                },
                "process_order": {
                    "task_messages": [...],
                    "functions": [...],
                    "post_actions": [...]
                }
            }
        }
    """

    initial_node: str
    nodes: Dict[str, NodeConfig]
