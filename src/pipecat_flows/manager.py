#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Core conversation flow management system.

This module provides the FlowManager class which orchestrates conversations
across different LLM providers. It supports:
- Static flows with predefined paths
- Dynamic flows with runtime-determined transitions
- State management and transitions
- Function registration and execution
- Action handling
- Cross-provider compatibility

The flow manager coordinates all aspects of a conversation, including:
- LLM context management
- Function registration
- State transitions
- Action execution
- Error handling
"""

import asyncio
import inspect
import sys
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Union, cast

from loguru import logger
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
)
from pipecat.pipeline.task import PipelineTask
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import BaseTransport

from .actions import ActionError, ActionManager
from .adapters import create_adapter
from .exceptions import (
    FlowError,
    FlowInitializationError,
    FlowTransitionError,
    InvalidFunctionError,
)
from .types import (
    ActionConfig,
    ConsolidatedFunctionResult,
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowConfig,
    FlowResult,
    FlowsDirectFunction,
    FlowsFunctionSchema,
    FunctionHandler,
    NodeConfig,
    get_or_generate_node_name,
)

if TYPE_CHECKING:
    from pipecat.services.anthropic.llm import AnthropicLLMService
    from pipecat.services.google.llm import GoogleLLMService
    from pipecat.services.openai.llm import OpenAILLMService

    LLMService = Union[OpenAILLMService, AnthropicLLMService, GoogleLLMService]
else:
    LLMService = Any


class FlowManager:
    """Manages conversation flows, supporting both static and dynamic configurations.

    The FlowManager orchestrates conversation flows by managing state transitions,
    function registration, and message handling across different LLM providers.

    Attributes:
        task: Pipeline task for frame queueing
        llm: LLM service instance (OpenAI, Anthropic, or Google)
        tts: Optional TTS service for voice actions
        state: Shared state dictionary across nodes
        current_node: Currently active node identifier
        initialized: Whether the manager has been initialized
        nodes: Node configurations for static flows
        current_functions: Currently registered function names
    """

    def __init__(
        self,
        *,
        task: PipelineTask,
        llm: LLMService,
        context_aggregator: Any,
        tts: Optional[Any] = None,
        flow_config: Optional[FlowConfig] = None,
        context_strategy: Optional[ContextStrategyConfig] = None,
        transport: Optional[BaseTransport] = None,
    ):
        """Initialize the flow manager.

        Args:
            task: PipelineTask instance for queueing frames
            llm: LLM service instance (e.g., OpenAI, Anthropic)
            context_aggregator: Context aggregator for updating user context
            tts: Optional TTS service for voice actions
            flow_config: Optional static flow configuration. If provided,
                operates in static mode with predefined nodes
            context_strategy: Optional context strategy configuration
            transport: Optional transport

        Raises:
            ValueError: If any transition handler is not a valid async callable

        Deprecated:
            0.0.18: The `tts` parameter is deprecated and will be removed in a future version.
        """
        if tts is not None:
            warnings.warn(
                "The 'tts' parameter is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.task = task
        self.llm = llm
        self.action_manager = ActionManager(task, flow_manager=self)
        self.adapter = create_adapter(llm)
        self.initialized = False
        self._context_aggregator = context_aggregator
        self._pending_transition: Optional[Dict[str, Any]] = None
        self._context_strategy = context_strategy or ContextStrategyConfig(
            strategy=ContextStrategy.APPEND
        )
        self.transport = transport

        # Set up static or dynamic mode
        if flow_config:
            self.nodes = flow_config["nodes"]
            self.initial_node = flow_config["initial_node"]
            logger.debug("Initialized in static mode")
        else:
            self.nodes = {}
            self.initial_node = None
            logger.debug("Initialized in dynamic mode")

        self.state: Dict[str, Any] = {}  # Shared state across nodes
        self.current_functions: Set[str] = set()  # Track registered functions
        self.current_node: Optional[str] = None

        self._showed_deprecation_warning_for_transition_fields = False
        self._showed_deprecation_warning_for_set_node = False

    def _validate_transition_callback(self, name: str, callback: Any) -> None:
        """Validate a transition callback.

        Args:
            name: Name of the function the callback is for
            callback: The callback to validate

        Raises:
            ValueError: If callback is not a valid async callable
        """
        if not callable(callback):
            raise ValueError(f"Transition callback for {name} must be callable")
        if not inspect.iscoroutinefunction(callback):
            raise ValueError(f"Transition callback for {name} must be async")

    async def initialize(self, initial_node: Optional[NodeConfig] = None) -> None:
        """Initialize the flow manager."""
        if self.initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return

        try:
            self.initialized = True
            logger.debug(f"Initialized {self.__class__.__name__}")

            # Set initial node
            node_name = None
            node = None
            if self.initial_node:
                # Static flow: self.initial_node is expected to be there
                node_name = self.initial_node
                node = self.nodes[self.initial_node]
                if not node:
                    raise ValueError(
                        f"Initial node '{self.initial_node}' not found in static flow configuration"
                    )
            else:
                # Dynamic flow: initial_node argument may have been provided (otherwise initial node
                # will be set later via set_node())
                if initial_node:
                    node_name = get_or_generate_node_name(initial_node)
                    node = initial_node
            if node_name:
                logger.debug(f"Setting initial node: {node_name}")
                await self._set_node(node_name, node)

        except Exception as e:
            self.initialized = False
            raise FlowInitializationError(f"Failed to initialize flow: {str(e)}") from e

    def get_current_context(self) -> List[dict]:
        """Get the current conversation context.

        Returns:
            List of messages in the current context, including system messages,
            user messages, and assistant responses.

        Raises:
            FlowError: If context aggregator is not available
        """
        if not self._context_aggregator:
            raise FlowError("No context aggregator available")

        return self._context_aggregator.user()._context.messages

    def register_action(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say")
            handler: Async or sync function that handles the action

        Example:
            async def custom_notification(action: dict):
                text = action.get("text", "")
                await notify_user(text)

            flow_manager.register_action("notify", custom_notification)
        """
        self.action_manager._register_action(action_type, handler)

    def _register_action_from_config(self, action: ActionConfig) -> None:
        """Register an action handler from action configuration.

        Args:
            action: Action configuration dictionary containing type and optional handler

        Raises:
            ActionError: If action type is not registered and no valid handler provided
        """
        action_type = action.get("type")
        handler = action.get("handler")

        # Register action if not already registered
        if action_type and action_type not in self.action_manager.action_handlers:
            # Register handler if provided
            if handler and callable(handler):
                self.register_action(action_type, handler)
                logger.debug(f"Registered action handler from config: {action_type}")
            # Raise error if no handler provided and not a built-in action
            elif action_type not in ["tts_say", "end_conversation"]:
                raise ActionError(
                    f"Action '{action_type}' not registered. "
                    "Provide handler in action config or register manually."
                )

    async def _call_handler(
        self, handler: FunctionHandler, args: FlowArgs
    ) -> FlowResult | ConsolidatedFunctionResult:
        """Call handler with appropriate parameters based on its signature.

        Detects whether the handler can accept a flow_manager parameter and
        calls it accordingly to maintain backward compatibility with legacy handlers.

        Args:
            handler: The function handler to call (either legacy or modern format)
            args: Arguments dictionary from the function call

        Returns:
            FlowResult: The result returned by the handler
        """
        # Get the function signature
        sig = inspect.signature(handler)

        # Calculate effective parameter count
        effective_param_count = len(sig.parameters)

        # Handle different function signatures
        if effective_param_count == 0:
            # Function takes no args
            return await handler()
        elif effective_param_count == 1:
            # Legacy handler with just args
            return await handler(args)
        else:
            # Modern handler with args and flow_manager
            return await handler(args, self)

    async def _create_transition_func(
        self,
        name: str,
        handler: Optional[Callable | FlowsDirectFunction],
        transition_to: Optional[str],
        transition_callback: Optional[Callable] = None,
    ) -> Callable:
        """Create a transition function for the given name and handler.

        Args:
            name: Name of the function being registered
            handler: Optional function to process data
            transition_to: Optional node to transition to (static flows)
            transition_callback: Optional callback for dynamic transitions

        Returns:
            Callable: Async function that handles the tool invocation

        Raises:
            ValueError: If both transition_to and transition_callback are specified
        """
        if transition_to and transition_callback:
            raise ValueError(
                f"Function {name} cannot have both transition_to and transition_callback"
            )

        # Validate transition callback if provided
        if transition_callback:
            self._validate_transition_callback(name, transition_callback)

        async def transition_func(params: FunctionCallParams) -> None:
            """Inner function that handles the actual tool invocation."""
            try:
                logger.debug(f"Function called: {name}")

                # Execute handler if present
                is_transition_only_function = False
                acknowledged_result = {"status": "acknowledged"}
                if handler:
                    # Invoke the handler with the provided arguments
                    if isinstance(handler, FlowsDirectFunction):
                        handler_response = await handler.invoke(params.arguments, self)
                    else:
                        handler_response = await self._call_handler(handler, params.arguments)
                    # Support both "consolidated" handlers that return (result, next_node) and handlers
                    # that return just the result.
                    if isinstance(handler_response, tuple):
                        result, next_node = handler_response
                        if result is None:
                            result = acknowledged_result
                            is_transition_only_function = True
                    else:
                        result = handler_response
                        next_node = None
                        # FlowsDirectFunctions should always be "consolidated" functions that return a tuple
                        if isinstance(handler, FlowsDirectFunction):
                            raise InvalidFunctionError(
                                f"Direct function {name} expected to return a tuple (result, next_node) but got {type(result)}"
                            )
                else:
                    result = acknowledged_result
                    next_node = None
                    is_transition_only_function = True

                logger.debug(
                    f"{'Transition-only function called for' if is_transition_only_function else 'Function handler completed for'} {name}"
                )

                # Determine if this is an edge function
                is_edge_function = (
                    bool(next_node) or bool(transition_to) or bool(transition_callback)
                )

                if is_edge_function:
                    # Store transition info for coordinated execution
                    transition_info = {
                        "next_node": next_node,
                        "transition_to": transition_to,
                        "transition_callback": transition_callback,
                        "function_name": name,
                        "arguments": params.arguments,
                        "result": result,
                    }
                    self._pending_transition = transition_info

                    properties = FunctionCallResultProperties(
                        run_llm=False,  # Don't run LLM until transition completes
                        on_context_updated=self._check_and_execute_transition,
                    )
                else:
                    # Node function - run LLM immediately
                    properties = FunctionCallResultProperties(
                        run_llm=True,
                        on_context_updated=None,
                    )

                await params.result_callback(result, properties=properties)

            except Exception as e:
                logger.error(f"Error in transition function {name}: {str(e)}")
                error_result = {"status": "error", "error": str(e)}
                await params.result_callback(error_result)

        return transition_func

    async def _check_and_execute_transition(self) -> None:
        """Check if all functions are complete and execute transition if so."""
        if not self._pending_transition:
            return

        # Check if all function calls are complete using Pipecat's state
        assistant_aggregator = self._context_aggregator.assistant()
        if not assistant_aggregator.has_function_calls_in_progress:
            # All functions complete, execute transition
            transition_info = self._pending_transition
            self._pending_transition = None

            await self._execute_transition(transition_info)

    async def _execute_transition(self, transition_info: Dict[str, Any]) -> None:
        """Execute the stored transition."""
        next_node = transition_info.get("next_node")
        transition_to = transition_info.get("transition_to")
        transition_callback = transition_info.get("transition_callback")
        function_name = transition_info.get("function_name")
        arguments = transition_info.get("arguments")
        result = transition_info.get("result")

        try:
            if next_node:  # Function-returned next node (consolidated function)
                if isinstance(next_node, str):  # Static flow
                    node_name = next_node
                    node = self.nodes[next_node]
                else:  # Dynamic flow
                    node_name = get_or_generate_node_name(next_node)
                    node = next_node
                logger.debug(f"Transition to function-returned node: {node_name}")
                await self._set_node(node_name, node)
            elif transition_to:  # Static flow (deprecated)
                logger.debug(f"Static transition to: {transition_to}")
                await self._set_node(transition_to, self.nodes[transition_to])
            elif transition_callback:  # Dynamic flow (deprecated)
                logger.debug(f"Dynamic transition for: {function_name}")
                # Check callback signature
                sig = inspect.signature(transition_callback)
                if len(sig.parameters) == 2:
                    # Old style: (args, flow_manager)
                    await transition_callback(arguments, self)
                else:
                    # New style: (args, result, flow_manager)
                    await transition_callback(arguments, result, self)
        except Exception as e:
            logger.error(f"Error executing transition: {str(e)}")
            raise

    def _lookup_function(self, func_name: str) -> Callable:
        """Look up a function by name in the main module.

        Args:
            func_name: Name of the function to look up

        Returns:
            Callable: The found function

        Raises:
            FlowError: If function is not found
        """
        main_module = sys.modules["__main__"]
        handler = getattr(main_module, func_name, None)

        if handler is not None:
            logger.debug(f"Found function '{func_name}' in main module")
            return handler

        error_message = (
            f"Function '{func_name}' not found in main module.\n"
            "Ensure the function is defined in your main script "
            "or imported into it."
        )

        raise FlowError(error_message)

    async def _register_function(
        self,
        name: str,
        new_functions: Set[str],
        handler: Optional[Callable | FlowsDirectFunction],
        transition_to: Optional[str] = None,
        transition_callback: Optional[Callable] = None,
    ) -> None:
        """Register a function with the LLM if not already registered.

        Args:
            name: Name of the function to register
            handler: A callable function handler, a FlowsDirectFunction, or a string.
                    If string starts with '__function__:', extracts the function name after the prefix.
            transition_to: Optional node to transition to (static flows)
            transition_callback: Optional transition callback (dynamic flows)
            new_functions: Set to track newly registered functions for this node

        Raises:
            FlowError: If function registration fails
        """
        if name not in self.current_functions:
            try:
                # Handle special token format (e.g. "__function__:function_name")
                if isinstance(handler, str) and handler.startswith("__function__:"):
                    func_name = handler.split(":")[1]
                    handler = self._lookup_function(func_name)

                # Create transition function
                transition_func = await self._create_transition_func(
                    name, handler, transition_to, transition_callback
                )

                # Register function with LLM
                self.llm.register_function(
                    name,
                    transition_func,
                )

                new_functions.add(name)
                logger.debug(f"Registered function: {name}")
            except Exception as e:
                logger.error(f"Failed to register function {name}: {str(e)}")
                raise FlowError(f"Function registration failed: {str(e)}") from e

    async def set_node_from_config(self, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Args:
            node_config: Configuration for the new node

        Raises:
            FlowTransitionError: If manager not initialized
            FlowError: If node setup fails
        """
        await self._set_node(get_or_generate_node_name(node_config), node_config)

    async def set_node(self, node_id: str, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Args:
            node_id: Identifier for the new node
            node_config: Configuration for the new node

        Raises:
            FlowTransitionError: If manager not initialized
            FlowError: If node setup fails
        """
        if not self._showed_deprecation_warning_for_set_node:
            self._showed_deprecation_warning_for_set_node = True
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    """`set_node()` is deprecated and will be removed in a future version. Instead, do the following for dynamic flows: 
- Prefer "consolidated" or "direct" functions that return a tuple (result, next_node) over deprecated `transition_callback`s
- Pass your initial node to `FlowManager.initialize()`
- If you really need to set a node explicitly, use `set_node_from_config()`
In all of these cases, you can provide a `name` in your new node's config for debug logging purposes.""",
                    DeprecationWarning,
                    stacklevel=2,
                )
        await self._set_node(node_id, node_config)

    async def _set_node(self, node_id: str, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Handles the complete node transition process in the following order:
        1. Execute pre-actions (if any)
        2. Set up messages (role and task)
        3. Register node functions
        4. Update LLM context with messages and tools
        5. Update state (current node and functions)
        6. Trigger LLM completion with new context
        7. Execute post-actions (if any)

        Args:
            node_id: Identifier for the new node
            node_config: Complete configuration for the node

        Raises:
            FlowTransitionError: If manager not initialized
            FlowError: If node setup fails
        """
        if not self.initialized:
            raise FlowTransitionError(f"{self.__class__.__name__} must be initialized first")

        try:
            # Clear any pending transition state when starting a new node
            # This ensures clean state regardless of how we arrived here:
            # - Normal transition flow (already cleared in _check_and_execute_transition)
            # - Direct calls to set_node/set_node_from_config
            self._pending_transition = None

            self._validate_node_config(node_id, node_config)
            logger.debug(f"Setting node: {node_id}")

            # Clear any deferred post-actions from previous node
            self.action_manager.clear_deferred_post_actions()

            # Register action handlers from config
            for action_list in [
                node_config.get("pre_actions", []),
                node_config.get("post_actions", []),
            ]:
                for action in action_list:
                    self._register_action_from_config(action)

            # Execute pre-actions if any
            if pre_actions := node_config.get("pre_actions"):
                await self._execute_actions(pre_actions=pre_actions)

            # Combine role and task messages
            messages = []
            if role_messages := node_config.get("role_messages"):
                messages.extend(role_messages)
            messages.extend(node_config["task_messages"])

            # Register functions and prepare tools
            tools: List[FlowsFunctionSchema | FlowsDirectFunction] = []
            new_functions: Set[str] = set()

            # Get functions list with default empty list if not provided
            functions_list = node_config.get("functions", [])

            async def register_function_schema(schema: FlowsFunctionSchema):
                """Helper to register a single FlowsFunctionSchema."""
                tools.append(schema)
                await self._register_function(
                    name=schema.name,
                    new_functions=new_functions,
                    handler=schema.handler,
                    transition_to=schema.transition_to,
                    transition_callback=schema.transition_callback,
                )

            async def register_direct_function(func):
                """Helper to register a single direct function."""
                direct_function = FlowsDirectFunction(function=func)
                tools.append(direct_function)
                await self._register_function(
                    name=direct_function.name,
                    new_functions=new_functions,
                    handler=direct_function,
                    transition_to=None,
                    transition_callback=None,
                )

            for func_config in functions_list:
                # Handle direct functions
                if callable(func_config):
                    await register_direct_function(func_config)
                # Handle Gemini's nested function declarations as a special case
                elif (
                    not isinstance(func_config, FlowsFunctionSchema)
                    and "function_declarations" in func_config
                ):
                    for declaration in func_config["function_declarations"]:
                        # Convert each declaration to FlowsFunctionSchema and process it
                        schema = self.adapter.convert_to_function_schema(
                            {"function_declarations": [declaration]}
                        )
                        await register_function_schema(schema)
                # Convert to FlowsFunctionSchema if needed and process it
                else:
                    schema = (
                        func_config
                        if isinstance(func_config, FlowsFunctionSchema)
                        else self.adapter.convert_to_function_schema(func_config)
                    )
                    await register_function_schema(schema)

            # Create ToolsSchema with standard function schemas
            standard_functions = []
            for tool in tools:
                # Convert FlowsFunctionSchema to standard FunctionSchema for the LLM
                standard_functions.append(tool.to_function_schema())

            # Use provider adapter to format tools, passing original configs for Gemini adapter
            formatted_tools = self.adapter.format_functions(
                standard_functions, original_configs=functions_list
            )

            # Update LLM context
            await self._update_llm_context(
                messages, formatted_tools, strategy=node_config.get("context_strategy")
            )
            logger.debug("Updated LLM context")

            # Update state
            self.current_node = node_id
            self.current_functions = new_functions

            # Trigger completion with new context
            respond_immediately = node_config.get("respond_immediately", True)
            if self._context_aggregator and respond_immediately:
                await self.task.queue_frames([self._context_aggregator.user().get_context_frame()])

            # Execute post-actions if any
            if post_actions := node_config.get("post_actions"):
                if respond_immediately:
                    await self._execute_actions(post_actions=post_actions)
                else:
                    # Schedule post-actions for execution after first LLM response in this node
                    self._schedule_deferred_post_actions(post_actions=post_actions)

            logger.debug(f"Successfully set node: {node_id}")

        except Exception as e:
            logger.error(f"Error setting node {node_id}: {str(e)}")
            raise FlowError(f"Failed to set node {node_id}: {str(e)}") from e

    def _schedule_deferred_post_actions(self, post_actions: List[ActionConfig]) -> None:
        self.action_manager.schedule_deferred_post_actions(post_actions=post_actions)

    async def _create_conversation_summary(
        self, summary_prompt: str, messages: List[dict]
    ) -> Optional[str]:
        """Generate a conversation summary from messages."""
        return await self.adapter.generate_summary(self.llm, summary_prompt, messages)

    async def _update_llm_context(
        self,
        messages: List[dict],
        functions: List[dict],
        strategy: Optional[ContextStrategyConfig] = None,
    ) -> None:
        """Update LLM context with new messages and functions.

        Args:
            messages: New messages to add to context
            functions: New functions to make available
            strategy: Optional context update configuration

        Raises:
            FlowError: If context update fails
        """
        try:
            update_config = strategy or self._context_strategy

            if (
                update_config.strategy == ContextStrategy.RESET_WITH_SUMMARY
                and self._context_aggregator
                and self._context_aggregator.user()._context.messages
            ):
                # We know summary_prompt exists because of __post_init__ validation in ContextStrategyConfig
                summary_prompt = cast(str, update_config.summary_prompt)
                try:
                    # Try to get summary with 5 second timeout
                    summary = await asyncio.wait_for(
                        self._create_conversation_summary(
                            summary_prompt,
                            self._context_aggregator.user()._context.messages,
                        ),
                        timeout=5.0,
                    )

                    if summary:
                        summary_message = self.adapter.format_summary_message(summary)
                        messages.insert(0, summary_message)
                        logger.debug("Added conversation summary to context")
                    else:
                        # Fall back to RESET strategy if summary fails
                        logger.warning("Failed to generate summary, falling back to RESET strategy")
                        update_config.strategy = ContextStrategy.RESET

                except asyncio.TimeoutError:
                    logger.warning("Summary generation timed out, falling back to RESET strategy")
                    update_config.strategy = ContextStrategy.RESET

            # For first node or RESET/RESET_WITH_SUMMARY strategy, use update frame
            frame_type = (
                LLMMessagesUpdateFrame
                if self.current_node is None
                or update_config.strategy
                in [ContextStrategy.RESET, ContextStrategy.RESET_WITH_SUMMARY]
                else LLMMessagesAppendFrame
            )

            await self.task.queue_frames(
                [frame_type(messages=messages), LLMSetToolsFrame(tools=functions)]
            )

            logger.debug(
                f"Updated LLM context using {frame_type.__name__} with strategy {update_config.strategy}"
            )

        except Exception as e:
            logger.error(f"Failed to update LLM context: {str(e)}")
            raise FlowError(f"Context update failed: {str(e)}") from e

    async def _execute_actions(
        self,
        pre_actions: Optional[List[ActionConfig]] = None,
        post_actions: Optional[List[ActionConfig]] = None,
    ) -> None:
        """Execute pre and post actions.

        Args:
            pre_actions: Actions to execute before context update
            post_actions: Actions to execute after context update
        """
        if pre_actions:
            await self.action_manager.execute_actions(pre_actions)
        if post_actions:
            await self.action_manager.execute_actions(post_actions)

    def _validate_node_config(self, node_id: str, config: NodeConfig) -> None:
        """Validate the configuration of a conversation node.

        This method ensures that:
        1. Required fields (task_messages) are present
        2. Functions have valid configurations based on their type:
        - FlowsFunctionSchema objects have proper handler/transition fields
        - Dictionary format functions have valid handler/transition entries
        - Direct functions are valid according to the FlowsDirectFunctions validation
        3. Edge functions (matching node names) are allowed without handlers/transitions

        Args:
            node_id: Identifier for the node being validated
            config: Complete node configuration to validate

        Raises:
            ValueError: If configuration is invalid or missing required fields
        """
        # Check required fields
        if "task_messages" not in config:
            raise ValueError(f"Node '{node_id}' missing required 'task_messages' field")

        # Get functions list with default empty list if not provided
        functions_list = config.get("functions", [])

        # Validate each function configuration if there are any
        for func in functions_list:
            # If the function is callable, validate using FlowsDirectFunction
            if callable(func):
                FlowsDirectFunction.validate_function(func)
                continue

            # Extract function name using adapter (handles all formats)
            try:
                name = self.adapter.get_function_name(func)
            except Exception as e:
                raise ValueError(f"Function in node '{node_id}' has invalid format: {str(e)}")

            # Skip validation for edge functions (matching node names)
            if name in self.nodes:
                continue

            # Check for handler, transition_to, and transition_callback depending on format
            if isinstance(func, FlowsFunctionSchema):
                # For FlowsFunctionSchema, we can access the fields directly
                has_handler = func.handler is not None
                has_transition_to = func.transition_to is not None
                has_transition_callback = func.transition_callback is not None
            else:
                # For dictionary formats, use the provider-specific format checks
                # OpenAI format
                if "function" in func:
                    has_handler = "handler" in func["function"]
                    has_transition_to = "transition_to" in func["function"]
                    has_transition_callback = "transition_callback" in func["function"]
                # Anthropic format
                elif "name" in func and "input_schema" in func:
                    has_handler = "handler" in func
                    has_transition_to = "transition_to" in func
                    has_transition_callback = "transition_callback" in func
                # Gemini format
                elif "function_declarations" in func and func["function_declarations"]:
                    decl = func["function_declarations"][0]
                    has_handler = "handler" in decl
                    has_transition_to = "transition_to" in decl
                    has_transition_callback = "transition_callback" in decl
                else:
                    # Unknown format, report error
                    raise ValueError(
                        f"Unknown function format for function '{name}' in node '{node_id}'"
                    )

            # Warn if the function has no handler or transitions
            if not has_handler and not has_transition_to and not has_transition_callback:
                logger.warning(
                    f"Function '{name}' in node '{node_id}' has neither handler, transition_to, nor transition_callback"
                )

            # Warn about usage of deprecated transition_to and transition_callback
            if (
                has_transition_to
                or has_transition_callback
                and not self._showed_deprecation_warning_for_transition_fields
            ):
                self._showed_deprecation_warning_for_transition_fields = True
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        '`transition_to` and `transition_callback` are deprecated and will be removed in a future version. Use a "consolidated" `handler` that returns a tuple (result, next_node) instead.',
                        DeprecationWarning,
                        stacklevel=2,
                    )
