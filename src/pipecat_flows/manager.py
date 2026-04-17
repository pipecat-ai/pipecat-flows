#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Core conversation flow management system.

This module provides the FlowManager class which orchestrates
conversations across different LLM providers. It supports:

- Flows with runtime-determined transitions
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
import warnings
from collections.abc import Callable
from typing import Any, cast

from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext, NotGiven
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.transports.base_transport import BaseTransport

from pipecat_flows.actions import ActionError, ActionManager
from pipecat_flows.adapters import LLMAdapter
from pipecat_flows.exceptions import (
    FlowError,
    FlowInitializationError,
    FlowTransitionError,
    InvalidFunctionError,
)
from pipecat_flows.types import (
    ActionConfig,
    ConsolidatedFunctionResult,
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowResult,
    FlowsDirectFunction,
    FlowsDirectFunctionWrapper,
    FlowsFunctionSchema,
    FunctionHandler,
    NodeConfig,
    get_or_generate_node_name,
)


class FlowManager:
    """Manages conversation flows.

    The FlowManager orchestrates conversation flows by managing state transitions,
    function registration, and message handling across different LLM providers,
    with comprehensive action handling and error management.

    The manager coordinates all aspects of a conversation including LLM context
    management, function registration, state transitions, and action execution.
    """

    def __init__(
        self,
        *,
        task: PipelineTask,
        llm: LLMService | LLMSwitcher,
        context_aggregator: Any,
        context_strategy: ContextStrategyConfig | None = None,
        transport: BaseTransport | None = None,
        global_functions: list[FlowsFunctionSchema | FlowsDirectFunction] | None = None,
    ):
        """Initialize the flow manager.

        Args:
            task: PipelineTask instance for queueing frames.
            llm: LLM service or LLMSwitcher.
            context_aggregator: Context aggregator for updating user context.
            context_strategy: Context strategy configuration for managing conversation
                context during transitions.
            transport: Transport instance for communication.
            global_functions: Optional list of FlowsFunctionSchemas or FlowsDirectFunctions
                that will be available at every node. These functions are registered once
                during initialization and automatically included alongside node-specific
                functions.
        """
        self._task = task
        self._llm = llm
        self._action_manager = ActionManager(task, flow_manager=self)
        self._adapter = LLMAdapter()
        self._initialized = False
        self._context_aggregator = context_aggregator
        self._pending_transition: dict[str, Any] | None = None
        self._context_strategy = context_strategy or ContextStrategyConfig(
            strategy=ContextStrategy.APPEND
        )
        self._transport = transport
        self._global_functions = global_functions or []

        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None

        self._showed_deprecation_warning_for_role_messages = False
        self._showed_deprecation_warning_for_reset_with_summary = False

    @property
    def state(self) -> dict[str, Any]:
        """Access the shared state dictionary across nodes.

        This property provides access to a persistent dictionary that maintains
        data across node transitions. It can be used to store and retrieve
        conversation state, user preferences, or any other data that needs
        to persist throughout the flow.

        Returns:
            Dict[str, Any]: The shared state dictionary that can be used for
                reading and writing state data.

        Examples:
            Setting state::

                flow_manager.state["user_name"] = "Alice"
                flow_manager.state["age"] = 25

            Getting state::

                name = flow_manager.state.get("user_name", "Unknown")
                age = flow_manager.state["age"]

            Checking for state::

                if "user_preferences" in flow_manager.state:
                    preferences = flow_manager.state["user_preferences"]
        """
        return self._state

    @property
    def transport(self) -> BaseTransport | None:
        """Access the transport instance used for communication.

        This property provides access to the transport instance that handles
        communication with the client (e.g., DailyTransport for Daily rooms).
        The transport can be used to interact with participants, manage
        audio/video settings, or access platform-specific features.

        Returns:
            Optional[BaseTransport]: The transport instance if provided during
                initialization, None otherwise.

        Examples:
            Accessing transport in action handlers::

                async def mute_participant(action: dict, flow_manager: FlowManager):
                    transport = flow_manager.transport
                    if transport and hasattr(transport, 'update_participant'):
                        await transport.update_participant(participant_id, {"canSnd": False})

            Working with Daily transport features::

                async def get_room_info(action: dict, flow_manager: FlowManager):
                    transport = flow_manager.transport
                    if isinstance(transport, DailyTransport):
                        participants = transport.participants()
                        return {"participant_count": len(participants)}
        """
        return self._transport

    @property
    def current_node(self) -> str | None:
        """Access the identifier of the currently active conversation node.

        This property provides access to the current node name/identifier in the
        conversation flow. It can be used to make decisions based on the current
        state of the conversation, implement conditional logic, or for debugging
        and logging purposes.

        Returns:
            Optional[str]: The identifier of the current node if a node is active,
                None if no node has been set or before initialization.

        Examples:
            Conditional logic based on current node::

                async def participant_joined(action: dict, flow_manager: FlowManager):
                    current = flow_manager.current_node
                    if current == "transferring_to_human_agent":
                        await start_human_agent_interaction(flow_manager)
                    elif current == "collecting_payment":
                        await setup_secure_session(flow_manager)

            Logging and debugging::

                async def log_conversation_state(action: dict, flow_manager: FlowManager):
                    node = flow_manager.current_node
                    logger.info(f"Current conversation node: {node}")
                    return {"current_node": node}
        """
        return self._current_node

    @property
    def task(self) -> PipelineTask:
        """Access the pipeline task instance for frame queueing.

        This property provides access to the PipelineTask instance used by the
        FlowManager. The task can be used to queue custom frames directly into
        the pipeline, enabling advanced flow control and custom frame injection.

        Returns:
            PipelineTask: The pipeline task instance used for frame processing
                and queueing operations.

        Examples:
            Queueing frames in handlers::

                async def send_custom_notification(action: dict, flow_manager: FlowManager):
                    from pipecat.frames.frames import TTSUpdateSettingsFrame

                    # Queue a TTS settings update frame
                    await task.queue_frame(
                        TTSUpdateSettingsFrame(settings={"voice": "your-new-voice-id"})
                    )
        """
        return self._task

    async def initialize(self, initial_node: NodeConfig | None = None) -> None:
        """Initialize the flow manager.

        Args:
            initial_node: Optional initial node configuration. If provided,
                the flow will start at this node immediately.

        Raises:
            FlowInitializationError: If initialization fails.

        Examples:
            Initialize with an initial node::

                flow_manager = FlowManager(
                    ... # Initialization parameters
                )
                await flow_manager.initialize(create_initial_node())

            Initialize without an initial node (set later via set_node_from_config)::

                flow_manager = FlowManager(
                    ... # Initialization parameters
                )
                await flow_manager.initialize()
        """
        if self._initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return

        try:
            self._initialized = True
            logger.debug(f"Initialized {self.__class__.__name__}")

            # Set initial node if provided (otherwise initial node
            # will be set later via set_node_from_config())
            if initial_node:
                node_name = get_or_generate_node_name(initial_node)
                logger.debug(f"Setting initial node: {node_name}")
                await self._set_node(node_name, initial_node)

        except Exception as e:
            self._initialized = False
            raise FlowInitializationError(f"Failed to initialize flow: {str(e)}") from e

    def get_current_context(self) -> list[dict]:
        """Get the current conversation context.

        Returns:
            List of messages in the current context, including system messages,
            user messages, and assistant responses.

        Raises:
            FlowError: If context aggregator is not available.
        """
        if not self._context_aggregator:
            raise FlowError("No context aggregator available")

        context = self._context_aggregator.user()._context

        return context.get_messages()

    def register_action(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say").
            handler: Async or sync function that handles the action.

        Example::

            async def custom_notification(action: dict):
                text = action.get("text", "")
                await notify_user(text)

            flow_manager.register_action("notify", custom_notification)
        """
        self._action_manager._register_action(action_type, handler)

    def _register_action_from_config(self, action: ActionConfig) -> None:
        """Register an action handler from action configuration.

        Args:
            action: Action configuration dictionary containing type and optional handler.

        Raises:
            ActionError: If action type is not registered and no valid handler provided.
        """
        action_type = action.get("type")
        handler = action.get("handler")

        # Register action if not already registered
        if action_type and action_type not in self._action_manager._action_handlers:
            # Register handler if provided
            if handler and callable(handler):
                self.register_action(action_type, handler)
                logger.debug(f"Registered action handler from config: {action_type}")
            else:
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
            handler: The function handler to call (either legacy or modern format).
            args: Arguments dictionary from the function call.

        Returns:
            The result returned by the handler.
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
        handler: Callable | FlowsDirectFunctionWrapper | None,
    ) -> Callable:
        """Create a transition function for the given name and handler.

        Args:
            name: Name of the function being registered.
            handler: Optional function to process data.

        Returns:
            Async function that handles the tool invocation.
        """

        async def transition_func(params: FunctionCallParams) -> None:
            """Inner function that handles the actual tool invocation."""
            try:
                logger.debug(f"Function called: {name}")

                # Execute handler if present
                is_transition_only_function = False
                acknowledged_result = {"status": "acknowledged"}
                if handler:
                    # Invoke the handler with the provided arguments
                    if isinstance(handler, FlowsDirectFunctionWrapper):
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
                        if isinstance(handler, FlowsDirectFunctionWrapper):
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
                is_edge_function = bool(next_node)

                if is_edge_function:
                    # Store transition info for coordinated execution
                    transition_info = {
                        "next_node": next_node,
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

    async def _execute_transition(self, transition_info: dict[str, Any]) -> None:
        """Execute the stored transition."""
        next_node = transition_info.get("next_node")

        try:
            if next_node:
                node_name = get_or_generate_node_name(next_node)
                logger.debug(f"Transition to function-returned node: {node_name}")
                await self._set_node(node_name, next_node)
        except Exception as e:
            logger.error(f"Error executing transition: {str(e)}")
            raise

    async def _register_function(
        self,
        name: str,
        new_functions: set[str],
        handler: Callable | FlowsDirectFunctionWrapper | None,
        *,
        cancel_on_interruption: bool = True,
        timeout_secs: float | None = None,
    ) -> None:
        """Register a function with the LLM if not already registered.

        Args:
            name: Name of the function to register
            handler: A callable function handler or a FlowsDirectFunction.
            new_functions: Set to track newly registered functions for this node
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. Defaults to True.
            timeout_secs: Optional per-tool timeout in seconds, overriding the global
                ``function_call_timeout_secs``. Defaults to None (use global timeout).

        Raises:
            FlowError: If function registration fails
        """
        if name not in self._current_functions:
            try:
                # Create transition function
                transition_func = await self._create_transition_func(name, handler)

                # Register function with LLM (or LLMSwitcher)
                kwargs = {}
                if timeout_secs is not None:
                    kwargs["timeout_secs"] = timeout_secs
                self._llm.register_function(
                    name,
                    transition_func,
                    cancel_on_interruption=cancel_on_interruption,
                    **kwargs,
                )

                new_functions.add(name)
                logger.debug(f"Registered function: {name}")
            except Exception as e:
                logger.error(f"Failed to register function {name}: {str(e)}")
                raise FlowError(f"Function registration failed: {str(e)}") from e

    async def set_node_from_config(self, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Used to manually transition between nodes in a flow.

        Args:
            node_config: Configuration for the new node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        await self._set_node(get_or_generate_node_name(node_config), node_config)

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
            node_id: Identifier for the new node.
            node_config: Complete configuration for the node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        if not self._initialized:
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
            self._action_manager.clear_deferred_post_actions()

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

            # Register functions and prepare tools
            tools: list[FlowsFunctionSchema | FlowsDirectFunctionWrapper] = []
            new_functions: set[str] = set()

            # Get functions list with default empty list if not provided
            functions_list = node_config.get("functions", [])

            # Mix in global functions that should be available at every node
            functions_list = self._global_functions + functions_list

            async def register_function_schema(schema: FlowsFunctionSchema):
                """Helper to register a single FlowsFunctionSchema."""
                tools.append(schema)
                await self._register_function(
                    name=schema.name,
                    new_functions=new_functions,
                    handler=schema.handler,
                    cancel_on_interruption=schema.cancel_on_interruption,
                    timeout_secs=schema.timeout_secs,
                )

            async def register_direct_function(func):
                """Helper to register a single direct function."""
                direct_function = FlowsDirectFunctionWrapper(function=func)
                tools.append(direct_function)
                await self._register_function(
                    name=direct_function.name,
                    new_functions=new_functions,
                    handler=direct_function,
                    cancel_on_interruption=direct_function.cancel_on_interruption,
                    timeout_secs=direct_function.timeout_secs,
                )

            for func_config in functions_list:
                if callable(func_config):
                    await register_direct_function(func_config)
                elif isinstance(func_config, FlowsFunctionSchema):
                    await register_function_schema(func_config)
                else:
                    raise InvalidFunctionError(
                        f"Invalid function format in node '{node_id}'. "
                        "Use FlowsFunctionSchema or direct functions."
                    )

            # Create ToolsSchema with standard function schemas
            standard_functions = []
            for tool in tools:
                # Convert FlowsFunctionSchema to standard FunctionSchema for the LLM
                standard_functions.append(tool.to_function_schema())

            formatted_tools = self._adapter.format_functions(standard_functions)

            role_message = node_config.get("role_message")
            role_messages = node_config.get("role_messages")

            if role_message and role_messages:
                logger.warning(
                    "Both 'role_message' and 'role_messages' specified; using 'role_message'"
                )

            if role_messages and not role_message:
                if not self._showed_deprecation_warning_for_role_messages:
                    self._showed_deprecation_warning_for_role_messages = True
                    warnings.warn(
                        "'role_messages' is deprecated and will be removed in 2.0.0. "
                        "Use 'role_message' (singular, str) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            # Update LLM context
            await self._update_llm_context(
                role_message=role_message,
                role_messages=role_messages if not role_message else None,
                task_messages=node_config["task_messages"],
                functions=formatted_tools,
                strategy=node_config.get("context_strategy"),
            )
            logger.debug("Updated LLM context")

            # Update state
            self._current_node = node_id
            self._current_functions = new_functions

            # Trigger completion with new context
            respond_immediately = node_config.get("respond_immediately", True)
            if respond_immediately:
                await self._task.queue_frames([LLMRunFrame()])

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

    def _schedule_deferred_post_actions(self, post_actions: list[ActionConfig]) -> None:
        self._action_manager.schedule_deferred_post_actions(post_actions=post_actions)

    async def _create_conversation_summary(
        self, summary_prompt: str, context: LLMContext
    ) -> str | None:
        """Generate a conversation summary from a given context."""
        return await self._adapter.generate_summary(self._llm, summary_prompt, context)

    async def _update_llm_context(
        self,
        role_message: str | None,
        role_messages: list[dict] | None,
        task_messages: list[dict],
        functions: ToolsSchema | NotGiven,
        strategy: ContextStrategyConfig | None = None,
    ) -> None:
        """Update LLM context with new messages and functions.

        If ``role_message`` is provided, it is sent as an
        ``LLMUpdateSettingsFrame`` (system instruction on the LLM itself).

        If ``role_messages`` (deprecated) is provided, the messages are
        prepended to the conversation context alongside ``task_messages``.

        Args:
            role_message: Optional role/personality string sent as the LLM
                system instruction via ``LLMUpdateSettingsFrame``.
            role_messages: Deprecated list-of-dicts prepended to context
                messages for backward compatibility.
            task_messages: Task messages to add to context.
            functions: New functions to make available.
            strategy: Optional context update configuration.

        Raises:
            FlowError: If context update fails.
        """
        try:
            frames = []

            # New path: role_message as LLM system instruction (persists until changed)
            if role_message:
                frames.append(
                    LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=role_message))
                )

            messages = []

            # Legacy path: role_messages prepended to context messages
            if role_messages:
                messages.extend(role_messages)

            update_config = strategy or self._context_strategy

            if update_config.strategy == ContextStrategy.RESET_WITH_SUMMARY:
                if not self._showed_deprecation_warning_for_reset_with_summary:
                    self._showed_deprecation_warning_for_reset_with_summary = True
                    warnings.warn(
                        "RESET_WITH_SUMMARY is deprecated and will be removed in 2.0.0. "
                        "Use Pipecat's native context summarization instead. To trigger "
                        "on-demand summarization during a node transition, push an "
                        "LLMSummarizeContextFrame in a pre-action. See "
                        "https://docs.pipecat.ai/guides/fundamentals/context-summarization",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            if (
                update_config.strategy == ContextStrategy.RESET_WITH_SUMMARY
                and self._context_aggregator
                and self._context_aggregator.user()._context
            ):
                # We know summary_prompt exists because of __post_init__ validation in ContextStrategyConfig
                summary_prompt = cast(str, update_config.summary_prompt)
                try:
                    # Try to get summary with 5 second timeout
                    summary = await asyncio.wait_for(
                        self._create_conversation_summary(
                            summary_prompt,
                            self._context_aggregator.user()._context,
                        ),
                        timeout=5.0,
                    )

                    if summary:
                        summary_message = self._adapter.format_summary_message(summary)
                        messages.append(summary_message)
                        logger.debug(f"Added conversation summary to context: {summary_message}")
                    else:
                        # Fall back to APPEND strategy if summary fails
                        logger.warning(
                            "Failed to generate summary, falling back to APPEND strategy"
                        )
                        update_config.strategy = ContextStrategy.APPEND

                except TimeoutError:
                    logger.warning("Summary generation timed out, falling back to APPEND strategy")
                    update_config.strategy = ContextStrategy.APPEND

            # Add task messages
            messages.extend(task_messages)

            # For first node or RESET/RESET_WITH_SUMMARY strategy, use update frame
            frame_type = (
                LLMMessagesUpdateFrame
                if self._current_node is None
                or update_config.strategy
                in [ContextStrategy.RESET, ContextStrategy.RESET_WITH_SUMMARY]
                else LLMMessagesAppendFrame
            )

            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))

            await self._task.queue_frames(frames)

            logger.debug(
                f"Updated LLM context using {frame_type.__name__} with strategy {update_config.strategy}"
            )

        except Exception as e:
            logger.error(f"Failed to update LLM context: {str(e)}")
            raise FlowError(f"Context update failed: {str(e)}") from e

    async def _execute_actions(
        self,
        pre_actions: list[ActionConfig] | None = None,
        post_actions: list[ActionConfig] | None = None,
    ) -> None:
        """Execute pre and post actions.

        Args:
            pre_actions: Actions to execute before context update.
            post_actions: Actions to execute after context update.
        """
        if pre_actions:
            await self._action_manager.execute_actions(pre_actions)
        if post_actions:
            await self._action_manager.execute_actions(post_actions)

    def _validate_node_config(self, node_id: str, config: NodeConfig) -> None:
        """Validate the configuration of a conversation node.

        This method ensures that:
        1. Required fields (task_messages) are present
        2. Functions have valid configurations based on their type:
        - FlowsFunctionSchema objects have proper handler/transition fields
        - Direct functions are valid according to the FlowsDirectFunctions validation
        3. Edge functions (matching node names) are allowed without handlers/transitions

        Args:
            node_id: Identifier for the node being validated.
            config: Complete node configuration to validate.

        Raises:
            FlowError: If required fields are missing.
            InvalidFunctionError: If function format is invalid.
        """
        # Check required fields
        if "task_messages" not in config:
            raise FlowError(f"Node '{node_id}' missing required 'task_messages' field")

        # Get functions list with default empty list if not provided
        functions_list = config.get("functions", [])

        # Validate each function configuration if there are any
        for func in functions_list:
            if callable(func):
                FlowsDirectFunctionWrapper.validate_function(func)
            elif isinstance(func, FlowsFunctionSchema):
                if not func.handler:
                    logger.warning(f"Function '{func.name}' in node '{node_id}' has no handler")
            else:
                raise InvalidFunctionError(
                    f"Invalid function format in node '{node_id}'. "
                    "Use FlowsFunctionSchema or direct functions."
                )
