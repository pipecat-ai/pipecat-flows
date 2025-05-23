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
import types
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
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

import docstring_parser
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema

from pipecat_flows.exceptions import InvalidFunctionError

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

NamedNodeConfig = tuple[str, "NodeConfig"]
"""Type alias for a node configuration with its name."""

UnifiedFunctionResult = Tuple[
    Optional[FlowResult], Optional["NodeConfig"] | Optional[NamedNodeConfig]
]
"""Return type for "unified" functions that do either or both of handling some processing as well as specifying the next node."""

LegacyFunctionHandler = Callable[[FlowArgs], Awaitable[FlowResult | UnifiedFunctionResult]]
"""Legacy function handler that only receives arguments.

Args:
    args: Dictionary of arguments from the function call

Returns:
    FlowResult: Result of the function execution
"""

FlowFunctionHandler = Callable[
    [FlowArgs, "FlowManager"], Awaitable[FlowResult | UnifiedFunctionResult]
]
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


class FlowsDirectFunction:
    def __init__(self, function: Callable):
        self.function = function
        self._initialize_metadata()

    @staticmethod
    def validate_function(function: Callable) -> None:
        if not inspect.iscoroutinefunction(function):
            raise InvalidFunctionError(f"Direct function {function.__name__} must be async")
        params = list(inspect.signature(function).parameters.items())
        if len(params) == 0:
            raise InvalidFunctionError(
                f"Direct function {function.__name__} must have at least one parameter (flow_manager)"
            )
        last_param_name = params[-1][0]
        if last_param_name != "flow_manager":
            raise InvalidFunctionError(
                f"Direct function {function.__name__} last parameter must be named 'flow_manager'"
            )

    async def invoke(
        self, args: Mapping[str, Any], flow_manager: "FlowManager"
    ) -> UnifiedFunctionResult:
        # print(f"[pk] Invoking function {self.name} with args: {args}")
        return await self.function(**args, flow_manager=flow_manager)

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

    def _initialize_metadata(self):
        # Get function name
        self.name = self.function.__name__

        # Parse docstring for description and parameters
        docstring = docstring_parser.parse(inspect.getdoc(self.function))

        # Get function description
        self.description = (docstring.description or "").strip()

        # Get function parameters as JSON schemas, and the list of required parameters
        self.properties, self.required = self._get_parameters_as_jsonschema(
            self.function, docstring.params
        )

    # TODO: maybe to better support things like enums, check if each type is a pydantic type and use its convert-to-jsonschema function
    def _get_parameters_as_jsonschema(
        self, func: Callable, docstring_params: List[docstring_parser.DocstringParam]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Get function parameters as a dictionary of JSON schemas and a list of required parameters.
        Ignore the last parameter, as it's expected to be the flow_manager.

        Args:
            func: Function to get parameters from

        Returns:
            A tuple containing:
                - A dictionary mapping each function parameter to its JSON schema
                - A list of required parameter names
        """

        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            # Ignore 'self' parameter
            if name == "self":
                continue

            # Ignore the last parameter, which is expected to be the flow_manager
            param_names = [n for n in sig.parameters]
            is_last_param = name == param_names[-1]
            if is_last_param:
                continue

            type_hint = hints.get(name)

            # Convert type hint to JSON schema
            properties[name] = self._typehint_to_jsonschema(type_hint)

            # Add whether the parameter is required
            # If the parameter has no default value, it's required
            if param.default is inspect.Parameter.empty:
                required.append(name)

            # Add parameter description from docstring
            for doc_param in docstring_params:
                if doc_param.arg_name == name:
                    properties[name]["description"] = doc_param.description or ""

        return properties, required

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
        if origin is Union or origin is types.UnionType:
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

            # NOTE: this does not yet support some fields being required and others not, which could happen when:
            # - the base class is a TypedDict with required fields (total=True or not specified) and the derived class has optional fields (total=False)
            # - Python 3.11+ NotRequired is used
            all_fields_required = getattr(type_hint, "__total__", True)

            for field_name, field_type in get_type_hints(type_hint).items():
                properties[field_name] = self._typehint_to_jsonschema(field_type)
                if all_fields_required:
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
    functions: List[Union[Dict[str, Any], FlowsFunctionSchema, FlowsDirectFunction]]
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
