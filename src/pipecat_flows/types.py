#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type definitions for the conversation flow system.

This module defines the core types used throughout the flow system:

- FlowResult: Function return type
- FlowArgs: Function argument type
- NodeConfig: Node configuration type
- FlowsFunctionSchema: A uniform schema for function calls in flows

These types provide structure and validation for flow configurations
and function interactions.
"""

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Required,
    TypedDict,
)

from pipecat.adapters.schemas.direct_function import BaseDirectFunctionWrapper
from pipecat.adapters.schemas.function_schema import FunctionSchema

from pipecat_flows.exceptions import InvalidFunctionError

if TYPE_CHECKING:
    from pipecat_flows.manager import FlowManager


class FlowResult(TypedDict, total=False):
    """Optional convention TypedDict for ``status``/``error`` results.

    .. deprecated:: 1.x.0
        ``FlowResult`` is no longer required or referenced by any handler type
        in the library, and Pipecat's upstream function-call-result contract is
        ``Any``. Define your own ``TypedDict`` if you want a structured result,
        or return any JSON-serializable value. ``FlowResult`` will be removed
        in 2.0.0.

    Parameters:
        status: Status of the function execution.
        error: Optional error message if execution failed.
    """

    status: str
    error: str


FlowArgs = Mapping[str, Any]
"""Type alias for function handler arguments.

Args are intended as read-only input to handlers. The concrete runtime type
is a ``dict``, but the annotation uses ``Mapping`` so handlers treat args as
input rather than scratch space, and so it aligns with Pipecat's typing of
``FunctionCallParams.arguments``.

Example::

    {
        "user_name": "John",
        "age": 25,
        "preferences": {"color": "blue"}
    }
"""

ConsolidatedFunctionResult = tuple[Any, "NodeConfig | None"]
"""Return type for "consolidated" functions.

Return type for "consolidated" functions that do either or both of:
- doing some work
- specifying the next node to transition to after the work is done

The first tuple element is the function-call result delivered to the LLM.
Any JSON-serializable value is accepted (matching Pipecat's upstream
``FunctionCallResultCallback`` contract). Pass ``None`` to signal a
transition-only handler; FlowManager substitutes an acknowledgement result.
"""

ZeroArgFunctionHandler = Callable[[], Awaitable[Any]]
"""Function handler that takes no arguments.

.. deprecated:: 1.x.0
    Use :data:`FlowFunctionHandler` (``(args, flow_manager)``) instead. Will be
    removed in 2.0.0.

Returns:
    Any JSON-serializable value, or a :data:`ConsolidatedFunctionResult`
    tuple to also specify the next node.
"""

LegacyFunctionHandler = Callable[[FlowArgs], Awaitable[Any]]
"""Legacy function handler that only receives arguments.

.. deprecated:: 1.x.0
    Use :data:`FlowFunctionHandler` (``(args, flow_manager)``) instead. Will be
    removed in 2.0.0.

Args:
    args: Dictionary of arguments from the function call.

Returns:
    Any JSON-serializable value, or a :data:`ConsolidatedFunctionResult`
    tuple to also specify the next node.
"""

FlowFunctionHandler = Callable[[FlowArgs, "FlowManager"], Awaitable[Any]]
"""Modern function handler that receives both arguments and flow_manager.

Args:
    args: Dictionary of arguments from the function call.
    flow_manager: Reference to the FlowManager instance.

Returns:
    Any JSON-serializable value, or a :data:`ConsolidatedFunctionResult`
    tuple to also specify the next node.
"""


FunctionHandler = ZeroArgFunctionHandler | LegacyFunctionHandler | FlowFunctionHandler
"""Union type for function handlers supporting 0-arg, legacy, and modern patterns."""


FlowsDirectFunction = Callable[..., Awaitable[ConsolidatedFunctionResult]]
"""Type alias for "direct" functions with automatic metadata extraction.

"Direct" functions have their definition automatically extracted from the
function signature and docstring. This can be used in :data:`NodeConfig`
directly, in lieu of a :class:`FlowsFunctionSchema` or function definition
dict.

Expected shape:

.. code-block:: python

    async def f(flow_manager: FlowManager, **params) -> ConsolidatedFunctionResult:
        ...

where ``**params`` are any named parameters described by the function's
docstring.

This is defined as ``Callable[..., ...]`` rather than a Protocol because
Python's Protocol system cannot express "any concrete named-parameter list"
against ``**kwargs: Any`` — a function with named params like ``llm: str``
is not structurally compatible with a ``**kwargs: Any`` protocol signature.
Runtime validation of the expected shape happens in
:meth:`FlowsDirectFunctionWrapper.validate_function`.
"""


LegacyActionHandler = Callable[[dict[str, Any]], Awaitable[None]]
"""Legacy action handler type that only receives the action dictionary.

.. deprecated:: 1.x.0
    Use :data:`FlowActionHandler` (``(action, flow_manager)``) instead. Will be
    removed in 2.0.0.

Args:
    action: Dictionary containing action configuration and parameters.

Example::

    async def simple_handler(action: dict):
        await notify(action["text"])
"""

FlowActionHandler = Callable[[dict[str, Any], "FlowManager"], Awaitable[None]]
"""Modern action handler type that receives both action and flow_manager.

Args:
    action: Dictionary containing action configuration and parameters.
    flow_manager: Reference to the FlowManager instance.

Example::

    async def advanced_handler(action: dict, flow_manager: FlowManager):
        await flow_manager.transport.notify(action["text"])
"""


class ActionConfig(TypedDict, total=False):
    """Configuration for an action.

    Parameters:
        type: Action type identifier (e.g. "tts_say", "notify_slack").
        handler: Callable to handle the action.
        text: Text for tts_say action.

    Note:
        Additional fields are allowed and passed to the handler.
    """

    type: Required[str]
    handler: LegacyActionHandler | FlowActionHandler
    text: str


class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Parameters:
        APPEND: Append new messages to existing context (default).
        RESET: Reset context with new messages only.
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary.

            .. deprecated:: 1.0.0
                Use Pipecat's native context summarization instead. To trigger
                on-demand summarization during a node transition, push an
                ``LLMSummarizeContextFrame`` in a pre-action. See
                https://docs.pipecat.ai/guides/fundamentals/context-summarization
                Will be removed in 2.0.0.
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

            .. deprecated:: 1.0.0
                Deprecated along with RESET_WITH_SUMMARY. Use
                ``LLMContextSummaryConfig.summarization_prompt`` instead.
                Will be removed in 2.0.0.
    """

    strategy: ContextStrategy
    summary_prompt: str | None = None

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
        cancel_on_interruption: Whether to cancel this function call when an
            interruption occurs. Defaults to False.
        timeout_secs: Optional per-tool timeout in seconds, overriding the global
            ``function_call_timeout_secs``. Defaults to None (use global timeout).
    """

    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    handler: FunctionHandler | None = None
    cancel_on_interruption: bool = False
    timeout_secs: float | None = None

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


def flows_direct_function(
    *, cancel_on_interruption: bool = False, timeout_secs: float | None = None
) -> Callable[[Callable], Callable]:
    """Decorator to attach additional metadata to a Pipecat direct function.

    This metadata can be used, for example, to store the additional arguments
    that should be used when registering the function with the Pipecat service.

    Args:
        cancel_on_interruption: Whether to cancel the function call when the user
            interrupts. Defaults to False.
        timeout_secs: Optional per-tool timeout in seconds, overriding the global
            ``function_call_timeout_secs``. Defaults to None (use global timeout).

    Returns:
        A decorator that attaches the metadata to the function.

    Example::

        @flows_direct_function(cancel_on_interruption=False, timeout_secs=30)
        async def long_running_task(flow_manager: FlowManager, query: str):
            '''Perform a long-running task that should not be cancelled on interruption.'''
            # ... implementation
            return {"status": "complete"}, None
    """

    def decorator(func: Callable) -> Callable:
        func._flows_cancel_on_interruption = cancel_on_interruption
        func._flows_timeout_secs = timeout_secs
        return func

    return decorator


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

    def _initialize_metadata(self):
        """Initialize metadata from function signature, docstring, and decorator."""
        super()._initialize_metadata()
        # Read Flows-specific metadata from decorator (falling back to fields'
        # defaults for backward compatibility)
        self.cancel_on_interruption = getattr(self.function, "_flows_cancel_on_interruption", False)
        self.timeout_secs = getattr(self.function, "_flows_timeout_secs", None)

    async def invoke(self, args: Mapping[str, Any], flow_manager: "FlowManager"):
        """Invoke the wrapped function with the provided arguments.

        Args:
            args: Arguments to pass to the function.
            flow_manager: FlowManager instance for function execution context.

        Returns:
            The result of the function call.
        """
        return await self.function(flow_manager=flow_manager, **args)


class NodeConfig(TypedDict, total=False):
    """Configuration for a single node in the flow.

    Parameters:
        task_messages: List of message dicts defining the current node's objectives.
        name: Name of the node, useful for debug logging when returning a next node
            from a "consolidated" function.
        role_message: The bot's role/personality as a plain string, sent as the
            LLM's system instruction via ``LLMUpdateSettingsFrame``. When
            provided, the system instruction persists across node transitions
            until a new node explicitly sets ``role_message`` again.
        role_messages: Deprecated list-of-dicts format for the bot's role/personality.

            .. deprecated:: 0.0.24
                Use ``role_message`` (str) instead. Will be removed in 2.0.0.

        functions: List of FlowsFunctionSchema definitions or direct functions
            whose definitions are automatically extracted from their signatures.
        pre_actions: Actions to execute before LLM inference.
        post_actions: Actions to execute after LLM inference.
        context_strategy: Strategy for updating context during transitions.
        respond_immediately: Whether to run LLM inference as soon as the node is
            set (default: True).

    Example::

        {
            "role_message": "You are a helpful assistant...",
            "task_messages": [
                {
                    "role": "developer",
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

    task_messages: Required[list[dict]]
    name: str
    role_message: str
    role_messages: list[dict[str, Any]]
    functions: list[FlowsFunctionSchema | FlowsDirectFunction]
    pre_actions: list[ActionConfig]
    post_actions: list[ActionConfig]
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
