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

import copy
import inspect
import sys
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Set, Union

from loguru import logger
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
)
from pipecat.pipeline.task import PipelineTask

from .actions import ActionManager
from .adapters import create_adapter
from .exceptions import FlowError, FlowInitializationError, FlowTransitionError
from .types import FlowArgs, FlowConfig, FlowResult, NodeConfig

if TYPE_CHECKING:
    from pipecat.services.anthropic import AnthropicLLMService
    from pipecat.services.google import GoogleLLMService
    from pipecat.services.openai import OpenAILLMService
    from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService

    LLMService = Union[OpenAILLMService, AnthropicLLMService, GoogleLLMService, GeminiMultimodalLiveLLMService]
else:
    LLMService = Any


class FlowManager:
    """Manages conversation flows, supporting both static and dynamic configurations.

    The FlowManager orchestrates conversation flows by managing state transitions,
    function registration, and message handling across different LLM providers.

    Attributes:
        task (PipelineTask): Pipeline task for frame queueing
        llm: LLM service instance (OpenAI, Anthropic, or Google)
        tts (Optional[Any]): Text-to-speech service for voice actions
        state (Dict[str, Any]): Shared state dictionary across nodes
        current_node (Optional[str]): Currently active node identifier
        initialized (bool): Whether the manager has been initialized
        nodes (Dict[str, Dict]): Node configurations for static flows
        current_functions (Set[str]): Currently registered function names
    """

    def __init__(
        self,
        *,
        task: PipelineTask,
        llm: LLMService,
        tts: Optional[Any] = None,
        flow_config: Optional[FlowConfig] = None,
        transition_callback: Optional[
            Callable[[str, Dict[str, Any], "FlowManager"], Awaitable[None]]
        ] = None,
    ):
        """Initialize the flow manager.

        Args:
            task: PipelineTask instance for queueing frames
            llm: LLM service instance (e.g., OpenAI, Anthropic)
            tts: Optional TTS service for voice actions
            flow_config: Optional static flow configuration. If provided,
                       operates in static mode with predefined nodes
            transition_callback: Optional callback for handling transitions.
                               Required for dynamic flows, ignored for static flows
                               in favor of static transitions
        """
        self.task = task
        self.llm = llm
        self.tts = tts
        self.action_manager = ActionManager(task, tts)
        self.adapter = create_adapter(llm)
        self.initialized = False

        # Set up static or dynamic mode
        if flow_config:
            self.nodes = flow_config["nodes"]
            self.initial_node = flow_config["initial_node"]
            self.transition_callback = self._handle_static_transition
            logger.debug("Initialized in static mode")
        else:
            self.nodes = {}
            self.initial_node = None
            self.transition_callback = transition_callback
            logger.debug("Initialized in dynamic mode")

        self.state: Dict[str, Any] = {}  # Shared state across nodes
        self.current_functions: Set[str] = set()  # Track registered functions
        self.current_node: Optional[str] = None

    async def initialize(self) -> None:
        """Initialize the flow manager."""
        if self.initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return

        try:
            self.initialized = True
            logger.debug(f"Initialized {self.__class__.__name__}")

            # If in static mode, set initial node
            if self.initial_node:
                logger.debug(f"Setting initial node: {self.initial_node}")
                await self.set_node(self.initial_node, self.nodes[self.initial_node])

        except Exception as e:
            self.initialized = False
            raise FlowInitializationError(f"Failed to initialize flow: {str(e)}") from e

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

    async def _call_handler(self, handler: Callable, args: FlowArgs) -> FlowResult:
        """Call handler with or without args based on its signature.

        Args:
            handler: The function to call
            args: Arguments dictionary

        Returns:
            Dict[str, Any]: Handler result
        """
        sig = inspect.signature(handler)
        if "args" in sig.parameters:
            return await handler(args)
        return await handler()

    async def _handle_static_transition(
        self,
        function_name: str,
        args: Dict[str, Any],
        flow_manager: "FlowManager",
    ) -> None:
        """Handle transitions for static flows."""
        if function_name in self.nodes:
            logger.debug(f"Static transition to node: {function_name}")
            await self.set_node(function_name, self.nodes[function_name])

    async def _create_transition_func(
        self, name: str, handler: Optional[Callable], transition_to: Optional[str]
    ) -> Callable:
        """Create a transition function for the given name and handler.

        This method creates a function that will be called when the LLM invokes a tool.
        It handles both data processing (via handlers) and state transitions.

        Args:
            name: Name of the function being registered
            handler: Optional function to process data (for node functions)
            transition_to: Optional node to transition to after processing

        Returns:
            Callable: Async function that handles the tool invocation
        """

        async def transition_func(
            function_name: str,
            tool_call_id: str,
            args: Dict[str, Any],
            llm: Any,
            context: Any,
            result_callback: Callable,
        ) -> None:
            """Inner function that handles the actual tool invocation.

            Args:
                function_name: Name of the called function
                tool_call_id: Unique identifier for this tool call
                args: Arguments passed to the function
                llm: LLM service instance
                context: Current conversation context
                result_callback: Function to call with results
            """
            try:
                if handler:
                    # Execute handler for node functions (e.g., data processing)
                    result = await self._call_handler(handler, args)
                    await result_callback(result)
                    logger.debug(f"Handler completed for {name}")
                else:
                    # Acknowledge edge functions without handlers
                    await result_callback({"status": "acknowledged"})
                    logger.debug(f"Edge function called: {name}")

                # Handle transitions based on flow type
                if transition_to and self.nodes:  # Static flow with transition_to
                    logger.debug(f"Static transition via transition_to: {transition_to}")
                    await self._handle_static_transition(transition_to, args, self)
                elif self.transition_callback and not self.nodes:  # Dynamic flow
                    logger.debug(f"Executing dynamic transition for {name}")
                    await self.transition_callback(function_name, args, self)

            except Exception as e:
                logger.error(f"Error in transition function {name}: {str(e)}")
                error_result = {"status": "error", "error": str(e)}
                await result_callback(error_result)

        return transition_func

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
        handler: Optional[Callable],
        transition_to: Optional[str],
        new_functions: Set[str],
    ) -> None:
        """Register a function with the LLM if not already registered.

        Args:
            name: Name of the function to register with the LLM
            handler: Either a callable function or a string. If string starts with
                    '__function__:', extracts the function name after the prefix
            transition_to: Optional node name to transition to after function execution
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

                self.llm.register_function(
                    name, await self._create_transition_func(name, handler, transition_to)
                )
                new_functions.add(name)
                logger.debug(f"Registered function: {name}")
            except Exception as e:
                logger.error(f"Failed to register function {name}: {str(e)}")
                raise FlowError(f"Function registration failed: {str(e)}") from e

    def _remove_handlers(self, tool_config: Dict[str, Any]) -> None:
        """Remove handlers from tool configuration.

        Args:
            tool_config: Function configuration to clean
        """
        if "function" in tool_config and "handler" in tool_config["function"]:
            del tool_config["function"]["handler"]
        elif "handler" in tool_config:
            del tool_config["handler"]
        elif "function_declarations" in tool_config:
            for decl in tool_config["function_declarations"]:
                if "handler" in decl:
                    del decl["handler"]

    def _remove_transition_info(self, tool_config: Dict[str, Any]) -> None:
        """Remove transition information from tool configuration."""
        if "function" in tool_config and "transition_to" in tool_config["function"]:
            del tool_config["function"]["transition_to"]
        elif "transition_to" in tool_config:
            del tool_config["transition_to"]
        elif "function_declarations" in tool_config:
            for decl in tool_config["function_declarations"]:
                if "transition_to" in decl:
                    del decl["transition_to"]

    async def set_node(self, node_id: str, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Handles the complete node transition process including:
        - Executing pre-actions
        - Registering node functions
        - Updating LLM context
        - Executing post-actions
        - Updating state

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
            self._validate_node_config(node_id, node_config)
            logger.debug(f"Setting node: {node_id}")

            # Execute pre-actions if any
            if pre_actions := node_config.get("pre_actions"):
                await self._execute_actions(pre_actions=pre_actions)

            # Combine role and task messages
            messages = []
            if role_messages := node_config.get("role_messages"):
                messages.extend(role_messages)
            messages.extend(node_config["task_messages"])

            # Register functions and prepare tools
            tools = []
            new_functions: Set[str] = set()

            for func_config in node_config["functions"]:
                # Handle Gemini's nested function declarations
                if "function_declarations" in func_config:
                    for declaration in func_config["function_declarations"]:
                        name = declaration["name"]
                        handler = declaration.get("handler")
                        transition_to = declaration.get("transition_to")
                        logger.debug(f"Processing function: {name}")
                        await self._register_function(name, handler, transition_to, new_functions)
                else:
                    name = self.adapter.get_function_name(func_config)
                    logger.debug(f"Processing function: {name}")

                    # Extract handler and transition info based on format
                    if "function" in func_config:
                        handler = func_config["function"].get("handler")
                        transition_to = func_config["function"].get("transition_to")
                    else:
                        handler = func_config.get("handler")
                        transition_to = func_config.get("transition_to")

                    await self._register_function(name, handler, transition_to, new_functions)

                # Create tool config (after removing handler and transition_to)
                tool_config = copy.deepcopy(func_config)
                self._remove_handlers(tool_config)
                self._remove_transition_info(tool_config)
                tools.append(tool_config)

            # Let adapter format tools for provider
            formatted_tools = self.adapter.format_functions(tools)

            # Update LLM context
            await self._update_llm_context(messages, formatted_tools)
            logger.debug("Updated LLM context")

            # Execute post-actions if any
            if post_actions := node_config.get("post_actions"):
                await self._execute_actions(post_actions=post_actions)

            # Update state
            self.current_node = node_id
            self.current_functions = new_functions

            logger.debug(f"Successfully set node: {node_id}")

        except Exception as e:
            logger.error(f"Error setting node {node_id}: {str(e)}")
            raise FlowError(f"Failed to set node {node_id}: {str(e)}") from e

    async def _update_llm_context(self, messages: List[dict], functions: List[dict]) -> None:
        """Update LLM context with new messages and functions.

        Args:
            messages: New messages to add to context
            functions: New functions to make available

        Raises:
            FlowError: If context update fails
        """
        try:
            # Determine frame type based on whether this is the first node
            frame_type = (
                LLMMessagesUpdateFrame if self.current_node is None else LLMMessagesAppendFrame
            )

            await self.task.queue_frames(
                [frame_type(messages=messages), LLMSetToolsFrame(tools=functions)]
            )

            logger.debug(f"Updated LLM context using {frame_type.__name__}")
        except Exception as e:
            logger.error(f"Failed to update LLM context: {str(e)}")
            raise FlowError(f"Context update failed: {str(e)}") from e

    async def _execute_actions(
        self, pre_actions: Optional[List[dict]] = None, post_actions: Optional[List[dict]] = None
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
        1. Required fields (task_messages, functions) are present
        2. Functions have valid configurations based on their type:
        - Node functions must have either a handler or transition_to
        - Edge functions (matching node names) are allowed without handlers
        3. Function configurations match the LLM provider's format

        Args:
            node_id: Identifier for the node being validated
            config: Complete node configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required fields
        if "task_messages" not in config:
            raise ValueError(f"Node '{node_id}' missing required 'task_messages' field")
        if "functions" not in config:
            raise ValueError(f"Node '{node_id}' missing required 'functions' field")

        # Validate each function configuration
        for func in config["functions"]:
            try:
                name = self.adapter.get_function_name(func)
            except KeyError:
                raise ValueError(f"Function in node '{node_id}' missing name field")

            # Skip validation for edge functions (matching node names)
            if name in self.nodes:
                continue

            # Check for handler in provider-specific formats
            has_handler = (
                ("function" in func and "handler" in func["function"])  # OpenAI format
                or "handler" in func  # Anthropic format
                or (  # Gemini format
                    "function_declarations" in func
                    and func["function_declarations"]
                    and "handler" in func["function_declarations"][0]
                )
            )

            # Check for transition_to in provider-specific formats
            has_transition = (
                ("function" in func and "transition_to" in func["function"])
                or "transition_to" in func
                or (
                    "function_declarations" in func
                    and func["function_declarations"]
                    and "transition_to" in func["function_declarations"][0]
                )
            )

            # Warn if function has neither handler nor transition_to
            # End nodes may have neither, so just warn rather than error
            if not has_handler and not has_transition:
                logger.warning(
                    f"Function '{name}' in node '{node_id}' has neither handler nor transition_to"
                )
