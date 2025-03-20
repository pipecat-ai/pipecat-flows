#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Action management system for conversation flows.

This module provides the ActionManager class which handles execution of actions
during conversation state transitions. It supports:
- Built-in actions (TTS, conversation ending)
- Custom action registration
- Synchronous and asynchronous handlers
- Pre and post-transition actions
- Error handling and validation

Actions are used to perform side effects during conversations, such as:
- Text-to-speech output
- Database updates
- External API calls
- Custom integrations
"""

import asyncio
from dataclasses import dataclass
import inspect
from typing import Callable, Dict, List, Optional

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import ControlFrame

from .exceptions import ActionError
from .types import ActionConfig, FlowActionHandler


@dataclass
class FunctionActionFrame(ControlFrame):
    action: dict
    function: FlowActionHandler


class ActionManager:
    """Manages the registration and execution of flow actions.

    Actions are executed during state transitions and can include:
    - Text-to-speech output
    - Database updates
    - External API calls
    - Custom user-defined actions

    Built-in actions:
    - tts_say: Speak text using TTS
    - end_conversation: End the current conversation

    Custom actions can be registered using register_action().
    """

    def __init__(self, task: PipelineTask, flow_manager: "FlowManager", tts=None):
        """Initialize the action manager.

        Args:
            task: PipelineTask instance used to queue frames
            flow_manager: FlowManager instance that this ActionManager is part of
            tts: Optional TTS service for voice actions
        """
        self.action_handlers: Dict[str, Callable] = {}
        self.task = task
        self._flow_manager = flow_manager
        self.tts = tts
        self.function_finished_event = asyncio.Event()

        # Register built-in actions
        self._register_action("tts_say", self._handle_tts_action)
        self._register_action("end_conversation", self._handle_end_action)
        self._register_action("function", self._handle_function_action)

        # Wire up function actions
        task.set_reached_downstream_filter((FunctionActionFrame,))
        @task.event_handler("on_frame_reached_downstream")
        async def on_frame_reached_downstream(task, frame):
            if isinstance(frame, FunctionActionFrame):
                await frame.function(frame.action, flow_manager)
                self.function_finished_event.set()

    def _register_action(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say")
            handler: Async or sync function that handles the action

        Raises:
            ValueError: If handler is not callable
        """
        if not callable(handler):
            raise ValueError("Action handler must be callable")
        self.action_handlers[action_type] = handler
        logger.debug(f"Registered handler for action type: {action_type}")

    async def execute_actions(self, actions: Optional[List[ActionConfig]]) -> None:
        """Execute a list of actions.

        Args:
            actions: List of action configurations to execute

        Raises:
            ActionError: If action execution fails

        Note:
            Each action must have a 'type' field matching a registered handler
        """
        if not actions:
            return

        for action in actions:
            action_type = action.get("type")
            if not action_type:
                raise ActionError("Action missing required 'type' field")

            handler = self.action_handlers.get(action_type)
            if not handler:
                raise ActionError(f"No handler registered for action type: {action_type}")

            try:
                # Determine if handler can accept flow_manager argument by inspecting its signature
                # Handlers can either take (action) or (action, flow_manager)
                try:
                    handler_positional_arg_count = handler.__code__.co_argcount
                    if inspect.ismethod(handler) and handler_positional_arg_count > 0:
                        # adjust for `self` being the first arg
                        handler_positional_arg_count -= 1
                    can_handle_flow_manager_arg = (
                        handler_positional_arg_count > 1 or handler.__code__.co_flags & 0x04
                    )
                except AttributeError:
                    logger.warning(
                        f"Unable to determine handler signature for action type '{action_type}', "
                        "falling back to legacy single-parameter call"
                    )
                    can_handle_flow_manager_arg = False

                # Invoke handler appropriately, with async and flow_manager arg as needed
                if can_handle_flow_manager_arg:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(action, self._flow_manager)
                    else:
                        handler(action, self._flow_manager)
                else:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(action)
                    else:
                        handler(action)
                logger.debug(f"Successfully executed action: {action_type}")
            except Exception as e:
                raise ActionError(f"Failed to execute action {action_type}: {str(e)}") from e

    async def _handle_tts_action(self, action: dict) -> None:
        """Built-in handler for TTS actions.

        Args:
            action: Action configuration containing 'text' to speak
        """
        if not self.tts:
            logger.warning("TTS action called but no TTS service provided")
            return

        text = action.get("text")
        if not text:
            logger.error("TTS action missing 'text' field")
            return

        try:
            await self.tts.say(text)
            # TODO: Update to TTSSpeakFrame once Pipecat is fixed
            # await self.task.queue_frame(TTSSpeakFrame(text=action["text"]))
        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def _handle_end_action(self, action: dict) -> None:
        """Built-in handler for ending the conversation.

        This handler queues an EndFrame to terminate the conversation. If the action
        includes a 'text' key, it will queue that text to be spoken before ending.

        Args:
            action: Dictionary containing the action configuration.
                Optional 'text' key for a goodbye message.
        """
        if action.get("text"):  # Optional goodbye message
            await self.task.queue_frame(TTSSpeakFrame(text=action["text"]))
        await self.task.queue_frame(EndFrame())

    async def _handle_function_action(self, action: dict) -> None:
        """Built-in handler for queuing functions to run "inline" in the pipeline (i.e. when the pipeline is done with all the work queued before it).

        This handler queues a FunctionFrame.
        It expects a 'handler' key in the action, containing the function to execute.

        Args:
            action: Dictionary containing the action configuration.
                Required 'handler' key containing the function to execute.
        """
        handler = action.get("handler")
        if not handler:
            logger.error("Function action missing 'handler' field")
            return
        # the reason we're queueing a frame here is to ensure it happens after bot turn is over in 
        # post_actions
        await self.task.queue_frame(FunctionActionFrame(action=action, function=handler))
        await self.function_finished_event.wait()
        self.function_finished_event.clear()
