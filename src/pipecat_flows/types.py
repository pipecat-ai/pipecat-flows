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
- FlowsFunctionSchema: A uniform schema for function calls in flows

These types provide structure and validation for flow configurations
and function interactions.
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from pipecat.adapters.schemas.direct_function import BaseDirectFunctionWrapper
from pipecat.adapters.schemas.function_schema import FunctionSchema

from pipecat_flows.exceptions import InvalidFunctionError

if TYPE_CHECKING:
    from pipecat_flows.manager import FlowManager

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

    Parameters:
        status: Status of the function execution.
        error: Optional error message if execution failed.

    Example::

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

Example::

    {
        "user_name": "John",
        "age": 25,
        "preferences": {"color": "blue"}
    }
"""

ConsolidatedFunctionResult = Tuple[Optional[FlowResult], Optional[Union["NodeConfig", str]]]
"""Return type for "consolidated" functions.

Return type for "consolidated" functions that do either or both of:
- doing some work
- specifying the next node to transition to after the work is done, specified as either:
    - a NodeConfig (for dynamic flows)
    - a node name (for static flows)
"""

LegacyFunctionHandler = Callable[[FlowArgs], Awaitable[FlowResult | ConsolidatedFunctionResult]]
"""Legacy function handler that only receives arguments.

Args:
    args: Dictionary of arguments from the function call.

Returns:
    FlowResult: Result of the function execution.
"""

FlowFunctionHandler = Callable[
    [FlowArgs, "FlowManager"], Awaitable[FlowResult | ConsolidatedFunctionResult]
]
"""Modern function handler that receives both arguments and flow_manager.

Args:
    args: Dictionary of arguments from the function call.
    flow_manager: Reference to the FlowManager instance.

Returns:
    FlowResult: Result of the function execution.
"""


FunctionHandler = Union[LegacyFunctionHandler, FlowFunctionHandler]
"""Union type for function handlers supporting both legacy and modern patterns."""


class FlowsDirectFunction(Protocol):
    """Protocol for "direct" functions with automatic metadata extraction.

    "Direct" functions have their definition automatically extracted from the function
    signature and docstring. This can be used in NodeConfigs directly, in lieu of a
    FlowsFunctionSchema or function definition dict.
    """

    def __call__(
        self, flow_manager: "FlowManager", **kwargs: Any
    ) -> Awaitable[ConsolidatedFunctionResult]:
        """Execute the direct function.

        Args:
            flow_manager: Reference to the FlowManager instance.
            **kwargs: Additional keyword arguments.

        Returns:
            Result of the function execution, which can include both a FlowResult
            and the next node to transition to.
        """
        ...


LegacyActionHandler = Callable[[Dict[str, Any]], Awaitable[None]]
"""Legacy action handler type that only receives the action dictionary.

Args:
    action: Dictionary containing action configuration and parameters.

Example::

    async def simple_handler(action: dict):
        await notify(action["text"])
"""

FlowActionHandler = Callable[[Dict[str, Any], "FlowManager"], Awaitable[None]]
"""Modern action handler type that receives both action and flow_manager.

Args:
    action: Dictionary containing action configuration and parameters.
    flow_manager: Reference to the FlowManager instance.

Example::

    async def advanced_handler(action: dict, flow_manager: FlowManager):
        await flow_manager.transport.notify(action["text"])
"""


class ActionConfigRequired(TypedDict):
    """Required fields for action configuration.

    Parameters:
        type: Action type identifier.
    """

    type: str


class ActionConfig(ActionConfigRequired, total=False):
    """Configuration for an action.

    Parameters:
        type: Action type identifier (e.g. "tts_say", "notify_slack").
        handler: Callable to handle the action.
        text: Text for tts_say action.

    Note:
        Additional fields are allowed and passed to the handler.
    """

    handler: Union[LegacyActionHandler, FlowActionHandler]
    text: str


class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Parameters:
        APPEND: Append new messages to existing context (default).
        RESET: Reset context with new messages only.
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary.
    """

    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"


@dataclass
class ContextStrategyConfig:
    """Configuration for context management.

    Parameters:
        strategy: Strategy to use for context management.
        summary_prompt: Required prompt text when using RESET_WITH_SUMMARY.
    """

    strategy: ContextStrategy
    summary_prompt: Optional[str] = None

    def __post_init__(self):
        """Validate configuration.

        Raises:
            ValueError: If summary_prompt is missing when using RESET_WITH_SUMMARY.
        """
        if self.strategy == ContextStrategy.RESET_WITH_SUMMARY and not self.summary_prompt:
            raise ValueError("summary_prompt is required when using RESET_WITH_SUMMARY strategy")


@dataclass
class FlowsFunctionSchema:
    """Function schema with Flows-specific properties.

    This class extends standard function schemas with additional fields for
    Pipecat Flows integration including handler assignment and transition logic.

    Parameters:
        name: Name of the function.
        description: Description of the function.
        properties: Dictionary defining parameter types and descriptions.
        required: List of required parameter names.
        handler: Function handler to process the function call.
        transition_to: Target node to transition to after function execution.

            .. deprecated:: 0.0.18
                Use a "consolidated" handler that returns a tuple (result, next_node) instead.

        transition_callback: Callback function for dynamic transitions.

            .. deprecated:: 0.0.18
                Use a "consolidated" handler that returns a tuple (result, next_node) instead.
    """

    name: str
    description: str
    properties: Dict[str, Any]
    required: List[str]
    handler: Optional[FunctionHandler] = None
    transition_to: Optional[str] = None
    transition_callback: Optional[Callable] = None

    def __post_init__(self):
        """Validate the schema configuration.

        Raises:
            ValueError: If both transition_to and transition_callback are specified.
        """
        if self.transition_to and self.transition_callback:
            raise ValueError("Cannot specify both transition_to and transition_callback")

    def to_function_schema(self) -> FunctionSchema:
        """Convert to a standard FunctionSchema for use with LLMs.

        Returns:
            FunctionSchema without flow-specific fields.
        """
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=self.properties,
            required=self.required,
        )


class FlowsDirectFunctionWrapper(BaseDirectFunctionWrapper):
    """Wrapper around a FlowsDirectFunction for metadata extraction and invocation.

    The wrapper:

    - extracts metadata from the function signature and docstring
    - generates a corresponding FunctionSchema
    - helps with function invocation
    """

    @classmethod
    def special_first_param_name(cls) -> str:
        """Get the special first parameter name for Flows direct functions.

        Returns:
            The string "flow_manager" which is expected as the first parameter.
        """
        return "flow_manager"

    @classmethod
    def validate_function(cls, function: Callable) -> None:
        """Validate the function signature and docstring.

        Args:
            function: The function to validate.

        Raises:
            InvalidFunctionError: If the function does not meet the requirements.
        """
        try:
            super().validate_function(function)
        except Exception as e:
            raise InvalidFunctionError(str(e)) from e

    async def invoke(self, args: Mapping[str, Any], flow_manager: "FlowManager"):
        """Invoke the wrapped function with the provided arguments.

        Args:
            args: Arguments to pass to the function.
            flow_manager: FlowManager instance for function execution context.

        Returns:
            The result of the function call.
        """
        return await self.function(flow_manager=flow_manager, **args)


class NodeConfigRequired(TypedDict):
    """Required fields for node configuration.

    Parameters:
        task_messages: List of message dicts defining the current node's objectives.
    """

    task_messages: List[dict]


class NodeConfig(NodeConfigRequired, total=False):
    """Configuration for a single node in the flow.

    Parameters:
        task_messages: List of message dicts defining the current node's objectives.
        name: Name of the node, useful for debug logging when returning a next node
            from a "consolidated" function.
        role_messages: List of message dicts defining the bot's role/personality.
        functions: List of function definitions in provider-specific format,
            FunctionSchema, or FlowsFunctionSchema; or a "direct function" whose
            definition is automatically extracted.
        pre_actions: Actions to execute before LLM inference.
        post_actions: Actions to execute after LLM inference.
        context_strategy: Strategy for updating context during transitions.
        respond_immediately: Whether to run LLM inference as soon as the node is
            set (default: True).

    Example::

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
            "context_strategy": ContextStrategyConfig(strategy=ContextStrategy.APPEND),
            "respond_immediately": true,
        }
    """

    name: str
    role_messages: List[Dict[str, Any]]
    functions: List[Union[Dict[str, Any], FlowsFunctionSchema, FlowsDirectFunction]]
    pre_actions: List[ActionConfig]
    post_actions: List[ActionConfig]
    context_strategy: ContextStrategyConfig
    respond_immediately: bool


def get_or_generate_node_name(node_config: NodeConfig) -> str:
    """Get the node name from configuration or generate a UUID if not set.

    Args:
        node_config: Node configuration dictionary.

    Returns:
        Node name from config or generated UUID string.
    """
    return node_config.get("name", str(uuid.uuid4()))


class FlowConfig(TypedDict):
    """Configuration for the entire conversation flow.

    Note:
        FlowConfig applies to static flows only.

    Parameters:
        initial_node: Name of the starting node.
        nodes: Dictionary mapping node names to their configurations.

    Example::

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
