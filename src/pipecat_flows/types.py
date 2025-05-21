#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type definitions for the conversation flow system.

This module defines the core types used throughout the flow system:
- FlowResult: Function return type
- FlowArgs: Function argument type
- NodeConfig: Node configuration type
- FlowConfig: Complete flow configuration type

These types provide structure and validation for flow configurations
and function interactions.
"""

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema

T = TypeVar("T")
TransitionHandler = Callable[[Dict[str, T], "FlowManager"], Awaitable[None]]
"""Type for transition handler functions.

Args:
    args: Dictionary of arguments from the function call
    flow_manager: Reference to the FlowManager instance

Returns:
    None: Handlers are expected to update state and set next node
"""


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
        "preferences": {"color": "blue"}
    }
"""

LegacyFunctionHandler = Callable[[FlowArgs], Awaitable[FlowResult]]
"""Legacy function handler that only receives arguments.

Args:
    args: Dictionary of arguments from the function call

Returns:
    FlowResult: Result of the function execution
"""

FlowFunctionHandler = Callable[[FlowArgs, "FlowManager"], Awaitable[FlowResult]]
"""Modern function handler that receives both arguments and flow_manager.

Args:
    args: Dictionary of arguments from the function call
    flow_manager: Reference to the FlowManager instance

Returns:
    FlowResult: Result of the function execution
"""

FunctionHandler = Union[LegacyFunctionHandler, FlowFunctionHandler]
"""Union type for function handlers supporting both legacy and modern patterns."""


LegacyActionHandler = Callable[[Dict[str, Any]], Awaitable[None]]
"""Legacy action handler type that only receives the action dictionary.

Args:
    action: Dictionary containing action configuration and parameters

Example:
    async def simple_handler(action: dict):
        await notify(action["text"])
"""

FlowActionHandler = Callable[[Dict[str, Any], "FlowManager"], Awaitable[None]]
"""Modern action handler type that receives both action and flow_manager.

Args:
    action: Dictionary containing action configuration and parameters
    flow_manager: Reference to the FlowManager instance

Example:
    async def advanced_handler(action: dict, flow_manager: FlowManager):
        await flow_manager.transport.notify(action["text"])
"""


class ActionConfigRequired(TypedDict):
    """Required fields for action configuration."""

    type: str


class ActionConfig(ActionConfigRequired, total=False):
    """Configuration for an action.

    Required:
        type: Action type identifier (e.g. "tts_say", "notify_slack")

    Optional:
        handler: Callable to handle the action
        text: Text for tts_say action
        Additional fields are allowed and passed to the handler
    """

    handler: Union[LegacyActionHandler, FlowActionHandler]
    text: str


class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Attributes:
        APPEND: Append new messages to existing context (default)
        RESET: Reset context with new messages only
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary
    """

    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"


@dataclass
class ContextStrategyConfig:
    """Configuration for context management.

    Attributes:
        strategy: Strategy to use for context management
        summary_prompt: Required prompt text when using RESET_WITH_SUMMARY
    """

    strategy: ContextStrategy
    summary_prompt: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.strategy == ContextStrategy.RESET_WITH_SUMMARY and not self.summary_prompt:
            raise ValueError("summary_prompt is required when using RESET_WITH_SUMMARY strategy")


@dataclass
class FlowsFunctionSchema:
    """Function schema with Flows-specific properties.

    This class provides similar functionality to FunctionSchema with additional
    fields for Pipecat Flows integration.

    Attributes:
        name: Name of the function
        description: Description of the function
        properties: Dictionary defining properties types and descriptions
        required: List of required parameters
        handler: Function handler to process the function call
        transition_to: Target node to transition to after function execution
        transition_callback: Callback function for dynamic transitions
    """

    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str]
    handler: Optional[FunctionHandler] = None
    transition_to: Optional[str] = None
    transition_callback: Optional[Callable] = None

    def __post_init__(self):
        """Validate the schema configuration."""
        if self.transition_to and self.transition_callback:
            raise ValueError("Cannot specify both transition_to and transition_callback")

    def to_function_schema(self) -> FunctionSchema:
        """Convert to a standard FunctionSchema for use with LLMs.

        Returns:
            FunctionSchema without flow-specific fields
        """
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=self.properties,
            required=self.required,
        )


class FlowsFunction:
    def __init__(self, function: Callable):
        self.function = function
        self._initialize_metadata()

    def _initialize_metadata(self):
        # Get function name
        self.name = self.function.__name__

        # Get function description
        # TODO: should ignore args and return type, right? Just the top-level docstring?
        self.description = inspect.getdoc(self.function) or ""

        # Get function properties as JSON schema
        # TODO: also get whether each property is required
        # TODO: is there a way to get "args" from doc string and use it to fill in descriptions?
        self.properties = self._get_parameters_as_jsonschema(self.function)

    # TODO: maybe to better support things like enums, check if each type is a pydantic type and use its convert-to-jsonschema function
    def _get_parameters_as_jsonschema(self, func: Callable) -> Dict[str, Any]:
        """
        Get function parameters as a dictionary of JSON schemas.

        Args:
            func: Function to get parameters from

        Returns:
            A dictionary mapping each function parameter to its JSON schema
        """

        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties = {}

        # TODO: use param or ignore it
        for name, param in sig.parameters.items():
            # Ignore 'self' parameter
            if name == "self":
                continue

            type_hint = hints.get(name)

            # Convert type hint to JSON schema
            properties[name] = self._typehint_to_jsonschema(type_hint)

        return properties

    # TODO: test this way more, throwing crazy types at it
    def _typehint_to_jsonschema(self, type_hint: Any) -> Dict[str, Any]:
        """
        Convert a Python type hint to a JSON Schema.

        Args:
            hint: A Python type hint

        Returns:
            A dictionary representing the JSON Schema
        """
        if type_hint is None:
            return {}

        # Handle basic types
        if type_hint is type(None):
            return {"type": "null"}
        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif type_hint is dict or type_hint is Dict:
            return {"type": "object"}
        elif type_hint is list or type_hint is List:
            return {"type": "array"}

        # Get origin and arguments for complex types
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle Optional/Union types
        if origin is Union:
            # Check if this is an Optional (Union with None)
            has_none = type(None) in args
            non_none_args = [arg for arg in args if arg is not type(None)]

            if has_none and len(non_none_args) == 1:
                # This is an Optional[X]
                schema = self._typehint_to_jsonschema(non_none_args[0])
                schema["nullable"] = True
                return schema
            else:
                # This is a general Union
                return {"anyOf": [self._typehint_to_jsonschema(arg) for arg in args]}

        # Handle List, Tuple, Set with specific item types
        if origin in (list, List, tuple, Tuple, set, Set) and args:
            return {"type": "array", "items": self._typehint_to_jsonschema(args[0])}

        # Handle Dict with specific key/value types
        if origin in (dict, Dict) and len(args) == 2:
            # For JSON Schema, keys must be strings
            return {"type": "object", "additionalProperties": self._typehint_to_jsonschema(args[1])}

        # Handle TypedDict
        if hasattr(type_hint, "__annotations__"):
            properties = {}
            required = []

            for field_name, field_type in get_type_hints(type_hint).items():
                properties[field_name] = self._typehint_to_jsonschema(field_type)
                # Check if field is required (this is a simplification, might need adjustment)
                if not getattr(type_hint, "__total__", True) or not isinstance(
                    field_type, Optional
                ):
                    required.append(field_name)

            schema = {"type": "object", "properties": properties}

            if required:
                schema["required"] = required

            return schema

        # Default to any type if we can't determine the specific schema
        return {}


class NodeConfigRequired(TypedDict):
    """Required fields for node configuration."""

    task_messages: List[dict]


class NodeConfig(NodeConfigRequired, total=False):
    """Configuration for a single node in the flow.

    Required fields:
        task_messages: List of message dicts defining the current node's objectives

    Optional fields:
        role_messages: List of message dicts defining the bot's role/personality
        functions: List of function definitions in provider-specific format, FunctionSchema,
            or FlowsFunctionSchema
        pre_actions: Actions to execute before LLM inference
        post_actions: Actions to execute after LLM inference
        context_strategy: Strategy for updating context during transitions
        respond_immediately: Whether to run LLM inference as soon as the node is set (default: True)

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
            "post_actions": [...],
            "context_strategy": ContextStrategyConfig(strategy=ContextStrategy.APPEND)
        }
    """

    role_messages: List[Dict[str, Any]]
    functions: List[Union[Dict[str, Any], FlowsFunctionSchema]]
    pre_actions: List[ActionConfig]
    post_actions: List[ActionConfig]
    context_strategy: ContextStrategyConfig
    respond_immediately: bool


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
