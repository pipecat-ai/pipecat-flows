#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test suite for ActionManager functionality.

This module tests the ActionManager class which handles execution of actions
during conversation flows. Tests cover:
- Built-in actions (TTS, end conversation)
- Custom action registration and execution
- Error handling and validation
- Action sequencing
- TTS service integration
- Frame queueing

The tests use unittest.IsolatedAsyncioTestCase for async support and include
mocked dependencies for PipelineTask and TTS service.
"""

import unittest
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from pipecat.frames.frames import EndFrame, TTSSpeakFrame

from pipecat_flows.actions import ActionManager
from pipecat_flows.exceptions import ActionError


class TestActionManager(unittest.IsolatedAsyncioTestCase):
    """Test suite for ActionManager class.

    Tests functionality of ActionManager including:
    - Built-in action handlers:
        - TTS speech synthesis
        - Conversation ending
    - Custom action registration
    - Action execution sequencing
    - Error handling:
        - Missing TTS service
        - Invalid actions
        - Failed handlers
    - Multiple action execution
    - Frame queueing validation

    Each test uses mocked dependencies to verify:
    - Correct frame generation
    - Proper service calls
    - Error handling behavior
    - Action sequencing
    """

    def setUp(self):
        """Set up test fixtures before each test.

        Creates:
        - Mock PipelineTask for frame queueing
        - Mock TTS service for speech synthesis
        - ActionManager instance with mocked dependencies
        """
        self.mock_task = AsyncMock()
        self.mock_task.queue_frame = AsyncMock()
        self.mock_task.event_handler = Mock()

        self.mock_tts = AsyncMock()
        self.mock_tts.say = AsyncMock()

        self.mock_flow_manager = AsyncMock()

        self.action_manager = ActionManager(self.mock_task, self.mock_flow_manager, self.mock_tts)

    async def test_initialization(self):
        """Test ActionManager initialization and default handlers."""
        # Verify built-in action handlers are registered
        self.assertIn("tts_say", self.action_manager.action_handlers)
        self.assertIn("end_conversation", self.action_manager.action_handlers)

        # Test initialization without TTS service
        action_manager_no_tts = ActionManager(self.mock_task, self.mock_flow_manager, None)
        self.assertIsNone(action_manager_no_tts.tts)

    async def test_tts_action(self):
        """Test basic TTS action execution."""
        action = {"type": "tts_say", "text": "Hello"}
        await self.action_manager.execute_actions([action])

        # Verify TTS service was called with correct text
        self.mock_tts.say.assert_called_once_with("Hello")

    @patch("loguru.logger.error")
    async def test_tts_action_no_text(self, mock_logger):
        """Test TTS action with missing text field."""
        action = {"type": "tts_say"}  # Missing text field

        # The implementation logs error but doesn't raise
        await self.action_manager.execute_actions([action])

        # Verify error was logged
        mock_logger.assert_called_with("TTS action missing 'text' field")

        # Verify TTS service was not called
        self.mock_tts.say.assert_not_called()

    @patch("loguru.logger.warning")
    async def test_tts_action_no_service(self, mock_logger):
        """Test TTS action when no TTS service is provided."""
        action_manager = ActionManager(self.mock_task, None)
        action = {"type": "tts_say", "text": "Hello"}

        # Should log warning but not raise error
        await action_manager.execute_actions([action])

        # Verify warning was logged
        mock_logger.assert_called_with("TTS action called but no TTS service provided")

        # Verify no frames were queued
        self.mock_task.queue_frame.assert_not_called()

    async def test_end_conversation_action(self):
        """Test basic end conversation action."""
        action = {"type": "end_conversation"}
        await self.action_manager.execute_actions([action])

        # Verify EndFrame was queued
        self.mock_task.queue_frame.assert_called_once()
        frame = self.mock_task.queue_frame.call_args[0][0]
        self.assertIsInstance(frame, EndFrame)

    async def test_end_conversation_with_goodbye(self):
        """Test end conversation action with goodbye message."""
        action = {"type": "end_conversation", "text": "Goodbye!"}
        await self.action_manager.execute_actions([action])

        # Verify both frames were queued in correct order
        self.assertEqual(self.mock_task.queue_frame.call_count, 2)

        # Verify TTSSpeakFrame
        first_frame = self.mock_task.queue_frame.call_args_list[0][0][0]
        self.assertIsInstance(first_frame, TTSSpeakFrame)
        self.assertEqual(first_frame.text, "Goodbye!")

        # Verify EndFrame
        second_frame = self.mock_task.queue_frame.call_args_list[1][0][0]
        self.assertIsInstance(second_frame, EndFrame)

    async def test_action_handler_signatures(self):
        """Test both legacy and modern action handler signatures."""

        # Test legacy single-parameter handler
        async def legacy_handler(action: dict):
            self.assertEqual(action["data"], "legacy")

        self.action_manager._register_action("legacy", legacy_handler)
        await self.action_manager.execute_actions([{"type": "legacy", "data": "legacy"}])

        # Test modern two-parameter handler
        async def modern_handler(action: dict, flow_manager: Any):
            self.assertEqual(action["data"], "modern")
            self.assertEqual(flow_manager, self.mock_flow_manager)

        self.action_manager._register_action("modern", modern_handler)
        await self.action_manager.execute_actions([{"type": "modern", "data": "modern"}])

    async def test_invalid_action(self):
        """Test handling invalid actions."""
        # Test missing type
        with self.assertRaises(ActionError) as context:
            await self.action_manager.execute_actions([{}])
        self.assertIn("missing required 'type' field", str(context.exception))

        # Test unknown action type
        with self.assertRaises(ActionError) as context:
            await self.action_manager.execute_actions([{"type": "invalid"}])
        self.assertIn("No handler registered", str(context.exception))

    async def test_multiple_actions(self):
        """Test executing multiple actions in sequence."""
        actions = [
            {"type": "tts_say", "text": "First"},
            {"type": "tts_say", "text": "Second"},
        ]
        await self.action_manager.execute_actions(actions)

        # Verify TTS was called twice in correct order
        self.assertEqual(self.mock_tts.say.call_count, 2)
        expected_calls = [unittest.mock.call("First"), unittest.mock.call("Second")]
        self.assertEqual(self.mock_tts.say.call_args_list, expected_calls)

    def test_register_invalid_handler(self):
        """Test registering invalid action handlers."""
        # Test non-callable handler
        with self.assertRaises(ValueError) as context:
            self.action_manager._register_action("invalid", "not_callable")
        self.assertIn("must be callable", str(context.exception))

        # Test None handler
        with self.assertRaises(ValueError) as context:
            self.action_manager._register_action("invalid", None)
        self.assertIn("must be callable", str(context.exception))

    async def test_none_or_empty_actions(self):
        """Test handling None or empty action lists."""
        # Test None actions
        await self.action_manager.execute_actions(None)
        self.mock_task.queue_frame.assert_not_called()
        self.mock_tts.say.assert_not_called()

        # Test empty list
        await self.action_manager.execute_actions([])
        self.mock_task.queue_frame.assert_not_called()
        self.mock_tts.say.assert_not_called()

    @patch("loguru.logger.error")
    async def test_action_error_handling(self, mock_logger):
        """Test error handling during action execution."""
        # Configure TTS mock to raise an error
        self.mock_tts.say.side_effect = Exception("TTS error")

        action = {"type": "tts_say", "text": "Hello"}
        await self.action_manager.execute_actions([action])

        # Verify error was logged
        mock_logger.assert_called_with("TTS error: TTS error")

        # Verify action was still marked as executed (doesn't raise)
        self.mock_tts.say.assert_called_once()

    async def test_action_execution_error_handling(self):
        """Test error handling during action execution."""
        action_manager = ActionManager(self.mock_task, self.mock_tts)

        # Test action with missing handler
        with self.assertRaises(ActionError):
            await action_manager.execute_actions([{"type": "nonexistent_action"}])

        # Test action handler that raises an exception
        async def failing_handler(action):
            raise Exception("Handler error")

        action_manager._register_action("failing_action", failing_handler)

        with self.assertRaises(ActionError):
            await action_manager.execute_actions([{"type": "failing_action"}])
