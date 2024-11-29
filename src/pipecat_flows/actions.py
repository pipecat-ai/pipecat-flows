#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from pipecat.frames.frames import (
    EndFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.task import PipelineTask

from .exceptions import ActionError


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

    def __init__(self, task: PipelineTask, tts=None):
        """Initialize the action manager.

        Args:
            task: PipelineTask instance used to queue frames
            tts: Optional TTS service for voice actions
        """
        self.action_handlers: Dict[str, Callable] = {}
        self.task = task
        self.tts = tts

        # Register built-in actions
        self._register_action("tts_say", self._handle_tts_action)
        self._register_action("end_conversation", self._handle_end_action)

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

    async def execute_actions(self, actions: Optional[List[Dict[str, Any]]]) -> None:
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
